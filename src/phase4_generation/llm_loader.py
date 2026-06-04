"""LLM loader — unified interface over HF transformers / Ollama / OpenAI-compatible APIs.

Why a unified interface?
    During development you'll want to swap backends frequently:
        * On a beefy GPU box → load Qwen2-7B-Instruct in 4-bit via transformers.
        * On a laptop / Mac → hit a local `ollama serve` instance.
        * For unit tests or sanity checks → an injected stub predictor.
        * In production → an OpenAI-compatible HTTP endpoint (vLLM, TGI, OpenRouter…).
    Each of these has wildly different SDKs. We hide that under a single
    `.generate(messages) -> str` method so the RAG pipeline doesn't care.

Lazy imports:
    `transformers`, `bitsandbytes`, and `requests` are all imported INSIDE
    the relevant backend's `_load` / `_generate` calls so that:
      * importing `llm_loader` is free and side-effect-free,
      * tests that only use the `predictor` stub never need heavy deps installed,
      * a missing backend dep only surfaces when you actually try to use it.
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")
try:
    import torch
    try:
        torch.classes.__path__ = []
    except Exception:
        pass
except ImportError:
    pass

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from src.utils.io import load_yaml


# ---------------------------------------------------------------------------
# Result type — keeps the return shape stable across backends.
# ---------------------------------------------------------------------------
@dataclass
class LLMResponse:
    text: str
    usage: Dict[str, Any] = field(default_factory=dict)   # {prompt_tokens, completion_tokens, …}
    raw: Any = None                                       # backend-native response, for debugging


# ---------------------------------------------------------------------------
# Unified client. The RAG chain only ever sees this class.
# ---------------------------------------------------------------------------
class LLMClient:
    """Thin facade over a chosen LLM backend.

    The constructor takes a fully-resolved config dict (the relevant subtree
    from `configs/generation.yaml → llm`). It defers actual model loading
    until the first `.generate()` call so importing this module is cheap.
    """

    def __init__(
        self,
        backend: str,
        config: Dict[str, Any],
        # Test hook — bypass real backends entirely.
        predictor: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ):
        if backend not in {"hf", "ollama", "openai", "stub"}:
            raise ValueError(f"Unknown LLM backend: {backend!r}")
        self.backend = backend
        self.config = config
        self._predictor = predictor

        # Backend-specific lazy state.
        self._hf_model = None
        self._hf_tokenizer = None

    # ----------------------------- public -----------------------------

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Run a single chat completion. `messages` is OpenAI-shaped."""
        if self._predictor is not None:
            return LLMResponse(text=self._predictor(messages), raw=None)
        if self.backend == "hf":
            return self._generate_hf(messages, max_new_tokens, temperature)
        if self.backend == "ollama":
            return self._generate_ollama(messages, max_new_tokens, temperature)
        if self.backend == "openai":
            return self._generate_openai(messages, max_new_tokens, temperature)
        raise RuntimeError(f"Backend {self.backend} has no generate() path.")

    # ----------------------------- HuggingFace -----------------------------

    def _ensure_hf_loaded(self) -> None:
        if self._hf_model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.config
        model_name = cfg["model_name"]
        quantization = cfg.get("quantization", "none")

        bnb_config = None
        if quantization in {"4bit", "8bit"}:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise RuntimeError(
                    "bitsandbytes / transformers BitsAndBytesConfig not available. "
                    "Install with `pip install bitsandbytes accelerate`."
                ) from e

            compute_dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[cfg.get("bnb_4bit_compute_dtype", "bfloat16")]

            if quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
                    bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
                )
            else:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        logger.info(f"Loading HF model '{model_name}' (quant={quantization}) …")
        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=cfg.get("trust_remote_code", True),
        )
        if self._hf_tokenizer.pad_token is None:
            self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token

        self._hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=cfg.get("device_map", "auto"),
            trust_remote_code=cfg.get("trust_remote_code", True),
        )
        self._hf_model.eval()

    def _generate_hf(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int],
        temperature: Optional[float],
    ) -> LLMResponse:
        import torch

        self._ensure_hf_loaded()
        cfg = self.config

        # Render messages with the model's own chat template (handles Qwen / Llama / etc.)
        prompt = self._hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._hf_tokenizer(prompt, return_tensors="pt").to(self._hf_model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or cfg.get("max_new_tokens", 512),
            do_sample=cfg.get("do_sample", False),
            temperature=temperature if temperature is not None else cfg.get("temperature", 0.1),
            top_p=cfg.get("top_p", 0.9),
            repetition_penalty=cfg.get("repetition_penalty", 1.0),
            pad_token_id=self._hf_tokenizer.pad_token_id,
        )
        # Greedy: temperature/top_p are ignored when do_sample=False; we drop
        # the keys to avoid the transformers warning.
        if not gen_kwargs["do_sample"]:
            gen_kwargs.pop("temperature")
            gen_kwargs.pop("top_p")

        with torch.no_grad():
            output_ids = self._hf_model.generate(**inputs, **gen_kwargs)

        # Slice off the input portion so only the new tokens are decoded.
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self._hf_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        usage = {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(new_tokens.shape[0]),
        }
        return LLMResponse(text=text, usage=usage)

    # ----------------------------- Ollama -----------------------------

    def _generate_ollama(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int],
        temperature: Optional[float],
    ) -> LLMResponse:
        import requests

        cfg = self.config
        url = cfg["base_url"].rstrip("/") + "/api/chat"
        options = {
                "temperature": temperature if temperature is not None else cfg.get("temperature", 0.1),
                "num_predict": max_new_tokens or cfg.get("max_new_tokens", 512),
                "num_ctx": cfg.get("num_ctx", 2048),
                "repeat_penalty": cfg.get("repetition_penalty", 1.1),
            }
        # Allow forcing CPU-only inference via num_gpu: 0 in config
        if "num_gpu" in cfg:
            options["num_gpu"] = cfg["num_gpu"]
        payload = {
            "model": cfg["model"],
            "messages": messages,
            "stream": False,
            "keep_alive": cfg.get("keep_alive", "10m"),
            "options": options,
        }
        timeout = cfg.get("timeout", 600)  # 10 min for slow GPUs (first-load)
        logger.info(f"Ollama request → {cfg['model']} (timeout={timeout}s)")
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # Ollama's response shape: {message: {role, content}, ...}
        text = data.get("message", {}).get("content", "").strip()
        return LLMResponse(text=text, raw=data,
                           usage={"prompt_tokens": data.get("prompt_eval_count"),
                                  "completion_tokens": data.get("eval_count")})

    # ----------------------------- OpenAI-compatible -----------------------------

    def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int],
        temperature: Optional[float],
    ) -> LLMResponse:
        import requests

        cfg = self.config
        api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"OpenAI-compatible backend requires env var {api_key_env}."
            )
        url = cfg["base_url"].rstrip("/") + "/chat/completions"
        payload = {
            "model": cfg["model"],
            "messages": messages,
            "max_tokens": max_new_tokens or cfg.get("max_new_tokens", 512),
            "temperature": temperature if temperature is not None else cfg.get("temperature", 0.1),
        }
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        return LLMResponse(text=text, raw=data, usage=data.get("usage", {}))


# ---------------------------------------------------------------------------
# Factory — the one function callers should normally use.
# ---------------------------------------------------------------------------
def load_llm(
    generation_yaml: Union[str, "os.PathLike[str]"] = "configs/generation.yaml",
    backend_override: Optional[str] = None,
    predictor: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> LLMClient:
    """Instantiate an LLMClient from `configs/generation.yaml`.

    Args:
        generation_yaml: path to the YAML config.
        backend_override: force a specific backend (useful for switching
            between "hf" on the server and "ollama" on a laptop without
            editing config files).
        predictor: a stub callable for tests. If provided, the real
            backend is never touched — handy for end-to-end RAG tests.
    """
    cfg = load_yaml(generation_yaml)["llm"]
    backend = backend_override or cfg["backend"]
    if predictor is not None:
        return LLMClient(backend="stub", config={}, predictor=predictor)
    sub = cfg.get(backend, {})
    if not sub:
        raise ValueError(f"No config block found for backend {backend!r} under llm.{backend}")
    return LLMClient(backend=backend, config=sub)
