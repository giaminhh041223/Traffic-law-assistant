"""QLoRA fine-tuning for the generator LLM — teach it the citation discipline.

Why QLoRA?
    A 7-B-param instruction model has ~7 GB of fp16 weights — full fine-tune
    needs 4–6× that in GPU RAM (optimiser states + gradients + activations) and
    overfits trivially on a few hundred legal QA pairs.

    QLoRA (Dettmers et al., NeurIPS 2023) sidesteps both problems:
        * The base model is loaded in 4-bit (≈ 4 GB for a 7-B model).
        * Only small low-rank adapter matrices are trained (≈ 0.1 % of params).
        * 8-bit paged AdamW keeps optimiser states tiny via CPU offloading.
    The fine-tune fits on a single 16 GB GPU and the trained adapter is a
    ~50 MB safetensors file — easy to version and to swap at inference.

What we train it for:
    Vietnamese legal QA pairs that ALREADY follow our citation format:
        {
          "system": "<the same SYSTEM_PROMPT_VI>",
          "context": "<the chunk text used to support the answer>",
          "question": "<user question>",
          "answer":   "<gold answer with inline citations>"
        }

    The model learns:
        1. To stay in the persona of a strict legal assistant.
        2. To always emit citations in the exact "(Điểm a, Khoản X, Điều Y …)" format.
        3. To refuse politely when the context is insufficient.

Output:
    A PEFT adapter directory at `lora_training.output_dir` (configurable in
    `configs/generation.yaml`). To use it at inference, point `llm_loader`'s
    HF backend at the base model and load the adapter via
    `PeftModel.from_pretrained(base, adapter_dir)`.

Reference: Dettmers, Pagnoni, Holtzman, Zettlemoyer — "QLoRA: Efficient
Finetuning of Quantized LLMs" (NeurIPS 2023).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from src.phase4_generation.prompt_templates import SYSTEM_PROMPT_VI
from src.utils.io import load_yaml, read_jsonl


# ---------------------------------------------------------------------------
# Config bundle
# ---------------------------------------------------------------------------
@dataclass
class LoRATrainConfig:
    base_model: str
    output_dir: str
    train_file: str
    eval_file: Optional[str]
    # Quantisation
    load_in_4bit: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    # LoRA
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    # Optim
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    lr_scheduler_type: str
    optim: str
    max_seq_length: int
    bf16: bool
    fp16: bool
    # Logging
    logging_steps: int
    save_steps: int
    eval_steps: int
    save_total_limit: int
    seed: int


def _load_cfg(yaml_path: Union[str, Path]) -> LoRATrainConfig:
    raw = load_yaml(yaml_path)["lora_training"]
    return LoRATrainConfig(
        base_model=raw["base_model"],
        output_dir=raw["output_dir"],
        train_file=raw["train_file"],
        eval_file=raw.get("eval_file"),
        load_in_4bit=raw.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=raw.get("bnb_4bit_compute_dtype", "bfloat16"),
        bnb_4bit_quant_type=raw.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=raw.get("bnb_4bit_use_double_quant", True),
        lora_r=raw.get("lora_r", 16),
        lora_alpha=raw.get("lora_alpha", 32),
        lora_dropout=raw.get("lora_dropout", 0.05),
        target_modules=raw.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        num_train_epochs=raw.get("num_train_epochs", 3),
        per_device_train_batch_size=raw.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=raw.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=raw.get("gradient_accumulation_steps", 16),
        learning_rate=raw.get("learning_rate", 2e-4),
        warmup_ratio=raw.get("warmup_ratio", 0.03),
        weight_decay=raw.get("weight_decay", 0.0),
        max_grad_norm=raw.get("max_grad_norm", 0.3),
        lr_scheduler_type=raw.get("lr_scheduler_type", "cosine"),
        optim=raw.get("optim", "paged_adamw_8bit"),
        max_seq_length=raw.get("max_seq_length", 1536),
        bf16=raw.get("bf16", True),
        fp16=raw.get("fp16", False),
        logging_steps=raw.get("logging_steps", 10),
        save_steps=raw.get("save_steps", 100),
        eval_steps=raw.get("eval_steps", 100),
        save_total_limit=raw.get("save_total_limit", 3),
        seed=raw.get("seed", 42),
    )


# ---------------------------------------------------------------------------
# Dataset shaping — produces the chat-formatted strings the SFTTrainer expects.
# ---------------------------------------------------------------------------
def _format_example(rec: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """Turn one JSONL record into a fully-rendered training string.

    Each record uses the SAME prompt structure that inference uses, so the
    fine-tune literally trains the model on the format it will be served in.
    """
    system = rec.get("system") or SYSTEM_PROMPT_VI
    context = rec.get("context", "").strip()
    question = rec["question"].strip()
    answer = rec["answer"].strip()

    user_content = (
        f"<context>\n{context}\n</context>\n\n"
        f"Câu hỏi: {question}\n\n"
        "Hãy trả lời câu hỏi trên CHỈ dựa trên <context>, kèm trích dẫn theo đúng định dạng quy định."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
        {"role": "assistant", "content": answer},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


# ---------------------------------------------------------------------------
# The training entry point
# ---------------------------------------------------------------------------
def train(generation_yaml: Union[str, Path] = "configs/generation.yaml") -> str:
    """Fine-tune the configured base model with QLoRA. Returns the adapter dir."""
    # Heavy deps inside the function — keeps `import` of this module cheap.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    try:
        from trl import SFTTrainer
    except ImportError as e:
        raise RuntimeError(
            "trl is required for QLoRA training. Install with `pip install trl`."
        ) from e

    cfg = _load_cfg(generation_yaml)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    _set_seed(cfg.seed)

    # ---- 4-bit quantisation config ----
    compute_dtype = {
        "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32,
    }[cfg.bnb_4bit_compute_dtype]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )

    # ---- Tokeniser + base model ----
    logger.info(f"Loading base model {cfg.base_model} in 4-bit …")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False   # gradient checkpointing-compatible
    model = prepare_model_for_kbit_training(model)

    # ---- LoRA adapter ----
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.target_modules,
    )
    model = get_peft_model(model, peft_config)
    _log_trainable(model)

    # ---- Datasets ----
    train_records = list(read_jsonl(cfg.train_file))
    if not train_records:
        raise ValueError(f"No records in {cfg.train_file}")
    train_ds = Dataset.from_list([_format_example(r, tokenizer) for r in train_records])

    eval_ds = None
    if cfg.eval_file and Path(cfg.eval_file).exists():
        eval_records = list(read_jsonl(cfg.eval_file))
        eval_ds = Dataset.from_list([_format_example(r, tokenizer) for r in eval_records])
    logger.info(
        f"Train: {len(train_ds)} examples"
        + (f"   ·   Eval: {len(eval_ds)} examples" if eval_ds else "")
    )

    # ---- TrainingArguments ----
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=cfg.eval_steps if eval_ds else None,
        report_to=["none"],
        seed=cfg.seed,
    )

    # ---- SFTTrainer ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        packing=False,            # don't pack — preserves per-example boundaries
    )

    logger.info("Beginning QLoRA training …")
    trainer.train()

    # ---- Save adapter ----
    final_dir = Path(cfg.output_dir) / "adapter_final"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    # Snapshot the resolved config alongside the weights for reproducibility.
    with open(final_dir / "lora_train_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)
    logger.info(f"LoRA adapter saved to {final_dir}")
    return str(final_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _log_trainable(model) -> None:
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100 * trainable / max(1, total)
    logger.info(f"Trainable params: {trainable:,} / {total:,}  ({pct:.3f} %)")


def _set_seed(seed: int) -> None:
    import random as _r
    _r.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="QLoRA fine-tune of the Vietnamese legal LLM.")
    p.add_argument("--generation-yaml", default="configs/generation.yaml")
    args = p.parse_args()
    train(args.generation_yaml)


if __name__ == "__main__":
    main()
