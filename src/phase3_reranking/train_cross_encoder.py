"""Fine-tune a Vietnamese cross-encoder with MultipleNegativesRankingLoss + Hard Negatives.

This is the academic centrepiece of Phase 3 — what makes the system "smart" instead of
"just-another-bi-encoder pipeline".

Why MultipleNegativesRankingLoss (MNR), adapted for a cross-encoder?
    MNR (Henderson et al. 2017; popularised by sentence-transformers) trains a model
    to push the correct (query, positive) pair's score ABOVE all candidate negatives
    via a softmax over candidates. For a cross-encoder we cannot share the bi-encoder
    in-batch trick directly — each (q, d) pair needs its own forward pass — but the
    LOSS shape transfers cleanly:

        For each query q with positive p and hard negatives n_1,…,n_K:
            scores = [ CE(q, p),  CE(q, n_1), …,  CE(q, n_K) ]      # K+1 scalars
            loss   = CrossEntropy( scale · scores, target = 0 )      # index 0 is the positive

    Each training step does (K+1) forward passes through the cross-encoder for one
    query, vs. 1 forward pass per bi-encoder example — but we more than pay for that
    in quality: cross-attention between q and d sees the entire pair jointly.

Where do the negatives come from?
    From `hard_negative_mining.mine_hard_negatives()` — BM25 retrieves the top-N
    documents lexically similar to the query, we drop the gold positive(s) and
    near-duplicates, and the remainder are LEXICALLY-plausible-but-WRONG documents.
    These are the negatives the model actually needs to learn to reject, not random
    sentences from elsewhere in the corpus that are trivially distinguishable.

Reference:
    * Karpukhin et al., "Dense Passage Retrieval" (EMNLP 2020), §4.2 — hard negatives.
    * Henderson et al., "Efficient Natural Language Response Suggestion" (2017) — MNR.
    * Nogueira & Cho, "Passage Re-ranking with BERT" (2019) — the cross-encoder pattern.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from loguru import logger

from src.phase2_hybrid_search.bm25_retriever import BM25Retriever
from src.phase3_reranking.hard_negative_mining import (
    MinedExample,
    TrainingPair,
    load_training_pairs,
    mine_hard_negatives,
)
from src.utils.io import load_yaml
from src.utils.vn_tokenizer import VnTokenizer


# ---------------------------------------------------------------------------
# Config bundle — flat so it's easy to log / save alongside checkpoints.
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    base_model: str
    output_dir: str
    train_file: str
    dev_file: Optional[str]
    # Hard negatives
    num_per_positive: int
    bm25_pool_size: int
    dedup_jaccard_threshold: float
    fill_with_random: bool
    random_seed: int
    # Loss
    scale: float
    # Optim
    num_epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    bf16: bool
    fp16: bool
    max_length: int
    # Logging
    logging_steps: int
    save_steps: int
    eval_steps: int
    seed: int
    # Surface form (must match inference)
    segment_input: bool
    include_dieu_title: bool
    include_citation_prefix: bool


def _load_train_config(yaml_path: Union[str, Path]) -> TrainConfig:
    raw = load_yaml(yaml_path)
    t = raw["cross_encoder_training"]
    ce = raw.get("cross_encoder", {})
    hn = t["hard_negatives"]
    lo = t["loss"]
    return TrainConfig(
        base_model=t["base_model"],
        output_dir=t["output_dir"],
        train_file=t["train_file"],
        dev_file=t.get("dev_file"),
        num_per_positive=hn["num_per_positive"],
        bm25_pool_size=hn["bm25_pool_size"],
        dedup_jaccard_threshold=hn["dedup_jaccard_threshold"],
        fill_with_random=hn["fill_with_random"],
        random_seed=hn.get("random_seed", 42),
        scale=lo["scale"],
        num_epochs=t["num_epochs"],
        per_device_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        learning_rate=t["learning_rate"],
        warmup_ratio=t.get("warmup_ratio", 0.1),
        weight_decay=t.get("weight_decay", 0.0),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        bf16=t.get("bf16", False),
        fp16=t.get("fp16", False),
        max_length=t.get("max_length", 256),
        logging_steps=t.get("logging_steps", 25),
        save_steps=t.get("save_steps", 200),
        eval_steps=t.get("eval_steps", 200),
        seed=t.get("seed", 42),
        segment_input=ce.get("segment_input", True),
        include_dieu_title=ce.get("include_dieu_title", True),
        include_citation_prefix=ce.get("include_citation_prefix", False),
    )


# ---------------------------------------------------------------------------
# Doc text construction — MUST stay byte-identical to inference.
# (Imported lazily by both the trainer and the inference reranker.)
# ---------------------------------------------------------------------------
def build_doc_text(chunk: Dict[str, Any], cfg: TrainConfig) -> str:
    parts: List[str] = []
    if cfg.include_citation_prefix and chunk.get("full_citation"):
        parts.append(chunk["full_citation"])
    if cfg.include_dieu_title and chunk.get("dieu_title"):
        parts.append(chunk["dieu_title"])
    if chunk.get("chunk_type") == "table" and chunk.get("linear_form"):
        parts.append(chunk["linear_form"])
    else:
        parts.append(chunk.get("text", ""))
    return ". ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Dataset — yields (query, positive_text, [negative_texts]) ready for tokenisation.
# ---------------------------------------------------------------------------
class CEListwiseDataset:
    """One sample = one query with its positive + K hard negatives.

    We deliberately keep this minimal (no torch.utils.data inheritance) so the
    file remains import-safe in environments where torch is not installed
    (tests, CI smoke checks). The training loop wraps it as needed.
    """

    def __init__(self, examples: List[MinedExample], segment_fn):
        self.examples = examples
        self.segment_fn = segment_fn

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        q = self.segment_fn(ex.query)
        pos = self.segment_fn(ex.positive_text)
        negs = [self.segment_fn(t) for t in ex.hard_negative_texts]
        return {"query": q, "positive": pos, "negatives": negs}


# ---------------------------------------------------------------------------
# The training loop
# ---------------------------------------------------------------------------
def train(
    retrieval_yaml: Union[str, Path] = "configs/retrieval.yaml",
    bm25_index_path: Union[str, Path] = "vector_store/bm25.pkl",
    chunks_jsonl_path: Union[str, Path] = "data/processed/chunks.jsonl",
) -> str:
    """Fine-tune the cross-encoder. Returns the output directory."""
    # Heavy deps imported here so the module remains importable for tests / mining-only flows.
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    cfg = _load_train_config(retrieval_yaml)
    _set_seed(cfg.seed)

    # ---- Load training pairs + chunk corpus + BM25 index ----
    logger.info(f"Loading training pairs from {cfg.train_file}")
    pairs = load_training_pairs(cfg.train_file)
    logger.info(f"  → {len(pairs)} pairs")

    chunks_by_id = _load_chunks(chunks_jsonl_path)
    logger.info(f"Loaded {len(chunks_by_id)} chunks from {chunks_jsonl_path}")

    logger.info(f"Loading BM25 index from {bm25_index_path}")
    bm25 = BM25Retriever.load(bm25_index_path)

    # ---- Mine hard negatives ----
    logger.info("Mining hard negatives…")
    train_examples = mine_hard_negatives(
        pairs=pairs,
        bm25=bm25,
        chunks_by_id=chunks_by_id,
        num_per_positive=cfg.num_per_positive,
        bm25_pool_size=cfg.bm25_pool_size,
        dedup_jaccard_threshold=cfg.dedup_jaccard_threshold,
        fill_with_random=cfg.fill_with_random,
        random_seed=cfg.random_seed,
    )
    if not train_examples:
        raise RuntimeError("Hard-negative mining produced no examples — check your data.")

    # Optional dev pairs (mined the same way, used for a periodic listwise loss check).
    dev_examples: List[MinedExample] = []
    if cfg.dev_file and Path(cfg.dev_file).exists():
        logger.info(f"Loading dev pairs from {cfg.dev_file}")
        dev_pairs = load_training_pairs(cfg.dev_file)
        dev_examples = mine_hard_negatives(
            pairs=dev_pairs,
            bm25=bm25,
            chunks_by_id=chunks_by_id,
            num_per_positive=cfg.num_per_positive,
            bm25_pool_size=cfg.bm25_pool_size,
            dedup_jaccard_threshold=cfg.dedup_jaccard_threshold,
            fill_with_random=cfg.fill_with_random,
            random_seed=cfg.random_seed + 1,
        )
        logger.info(f"  → {len(dev_examples)} dev examples")

    # Persist the mined dataset for reproducibility / audit.
    mined_dump = Path(cfg.output_dir) / "mined_train.jsonl"
    mined_dump.parent.mkdir(parents=True, exist_ok=True)
    _dump_mined(train_examples, mined_dump)
    logger.info(f"Dumped mined training set → {mined_dump}")

    # ---- Tokeniser + model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading base model '{cfg.base_model}' on {device}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.base_model, num_labels=1)
    model.to(device)

    # Mixed precision — only one of bf16/fp16 should be true.
    autocast_dtype = None
    if cfg.bf16 and torch.cuda.is_available():
        autocast_dtype = torch.bfloat16
    elif cfg.fp16 and torch.cuda.is_available():
        autocast_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler() if autocast_dtype == torch.float16 else None

    # ---- Word-segmentation: keep train/inference aligned ----
    segmenter = VnTokenizer(backend="pyvi", lowercase=False, remove_punctuation=False)
    segment_fn = segmenter.segment if cfg.segment_input else (lambda x: x)

    train_ds = CEListwiseDataset(train_examples, segment_fn)
    dev_ds = CEListwiseDataset(dev_examples, segment_fn) if dev_examples else None

    # ---- Optimiser + schedule ----
    steps_per_epoch = math.ceil(len(train_ds) / cfg.per_device_batch_size)
    total_optim_steps = max(1, steps_per_epoch * cfg.num_epochs // cfg.gradient_accumulation_steps)
    warmup_steps = int(cfg.warmup_ratio * total_optim_steps)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_optim_steps)

    logger.info(
        f"Training: {len(train_ds)} examples · {cfg.num_epochs} epochs · "
        f"{steps_per_epoch} steps/epoch · {total_optim_steps} optimiser steps · "
        f"warmup={warmup_steps}"
    )

    # ---- Training loop ----
    global_step = 0
    best_dev_loss = float("inf")
    model.train()
    indices = list(range(len(train_ds)))
    rng = random.Random(cfg.seed)

    for epoch in range(cfg.num_epochs):
        rng.shuffle(indices)
        running_loss, running_n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, idx in enumerate(indices):
            sample = train_ds[idx]
            loss = _mnr_loss_for_sample(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                scale=cfg.scale,
                max_length=cfg.max_length,
                device=device,
                autocast_dtype=autocast_dtype,
            )

            # Per-sample loss; gradient accumulation divides by accum_steps.
            loss_to_back = loss / cfg.gradient_accumulation_steps
            if scaler is not None:
                scaler.scale(loss_to_back).backward()
            else:
                loss_to_back.backward()

            running_loss += float(loss.detach())
            running_n += 1

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.logging_steps == 0:
                    avg = running_loss / max(1, running_n)
                    lr_now = scheduler.get_last_lr()[0]
                    logger.info(f"epoch {epoch} · step {global_step} · loss {avg:.4f} · lr {lr_now:.2e}")
                    running_loss, running_n = 0.0, 0

                if dev_ds is not None and global_step % cfg.eval_steps == 0:
                    dev_loss = _evaluate(model, tokenizer, dev_ds, cfg, device, autocast_dtype)
                    logger.info(f"  ↳ dev_loss = {dev_loss:.4f}")
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        _save_checkpoint(model, tokenizer, Path(cfg.output_dir) / "best", cfg)
                        logger.info(f"  ↳ new best — saved to {cfg.output_dir}/best")

                if global_step % cfg.save_steps == 0:
                    ckpt = Path(cfg.output_dir) / f"checkpoint-{global_step}"
                    _save_checkpoint(model, tokenizer, ckpt, cfg)
                    logger.info(f"  ↳ checkpoint → {ckpt}")

        logger.info(f"=== finished epoch {epoch} ===")

    # ---- Final save ----
    final_dir = Path(cfg.output_dir) / "final"
    _save_checkpoint(model, tokenizer, final_dir, cfg)
    logger.info(f"Final model saved to {final_dir}")
    return str(final_dir)


# ---------------------------------------------------------------------------
# The MNR-for-cross-encoder loss for one (query, positive, [negatives]) sample.
# ---------------------------------------------------------------------------
def _mnr_loss_for_sample(
    model,
    tokenizer,
    sample: Dict[str, Any],
    scale: float,
    max_length: int,
    device: str,
    autocast_dtype,
):
    """Compute -log softmax(scale * scores)[0] for one query.

    scores = [ CE(q, pos), CE(q, neg_1), …, CE(q, neg_K) ]
    target = 0 (the positive sits at index 0).
    """
    import torch
    import torch.nn.functional as F

    q = sample["query"]
    candidates = [sample["positive"], *sample["negatives"]]
    pairs = [[q, c] for c in candidates]

    enc = tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    if autocast_dtype is not None:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            logits = model(**enc).logits.squeeze(-1)  # (K+1,)
            scores = logits * scale
            target = torch.zeros(1, dtype=torch.long, device=device)
            loss = F.cross_entropy(scores.unsqueeze(0), target)
    else:
        logits = model(**enc).logits.squeeze(-1)
        scores = logits * scale
        target = torch.zeros(1, dtype=torch.long, device=device)
        loss = F.cross_entropy(scores.unsqueeze(0), target)
    return loss


def _evaluate(model, tokenizer, dev_ds: CEListwiseDataset, cfg: TrainConfig, device, autocast_dtype) -> float:
    import torch
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(len(dev_ds)):
            sample = dev_ds[i]
            loss = _mnr_loss_for_sample(
                model=model, tokenizer=tokenizer, sample=sample,
                scale=cfg.scale, max_length=cfg.max_length,
                device=device, autocast_dtype=autocast_dtype,
            )
            total += float(loss)
            n += 1
    model.train()
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _load_chunks(chunks_jsonl_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    from src.utils.io import read_jsonl
    chunks_by_id: Dict[str, Dict[str, Any]] = {}
    for rec in read_jsonl(chunks_jsonl_path):
        cid = rec.get("chunk_id") or rec.get("id")
        if not cid:
            continue
        chunks_by_id[cid] = rec
    return chunks_by_id


def _dump_mined(examples: List[MinedExample], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({
                "query": ex.query,
                "positive_text": ex.positive_text,
                "positive_chunk_id": ex.positive_chunk_id,
                "hard_negative_chunk_ids": ex.hard_negative_chunk_ids,
                "negative_sources": ex.negative_sources,
            }, ensure_ascii=False) + "\n")


def _save_checkpoint(model, tokenizer, ckpt_dir: Path, cfg: TrainConfig) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    # Drop the resolved config next to the weights so the run is reproducible.
    with open(ckpt_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Fine-tune the Vietnamese cross-encoder.")
    p.add_argument("--retrieval-yaml", default="configs/retrieval.yaml")
    p.add_argument("--bm25-index", default="vector_store/bm25.pkl")
    p.add_argument("--chunks-jsonl", default="data/processed/chunks.jsonl")
    args = p.parse_args()
    train(args.retrieval_yaml, args.bm25_index, args.chunks_jsonl)


if __name__ == "__main__":
    main()
