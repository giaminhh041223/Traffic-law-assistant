"""Vietnamese tokenization utilities.

Two backends:
  * pyvi (default) — zero-config, pure Python.
  * vncorenlp — more accurate, requires Java + VnCoreNLP jar (set VNCORENLP_JAR env var).

Used downstream by:
  * Phase 1 — light cleaning of extracted text (optional).
  * Phase 2 — BM25 tokenization (CRITICAL: Vietnamese words are space-separated syllables
              but legally meaningful tokens are multi-syllable — e.g. "giấy_phép_lái_xe").
"""
from __future__ import annotations

import os
import re
import string
from functools import lru_cache
from typing import List

# Vietnamese-aware lowercase punctuation strip; keep digits and Vietnamese chars.
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation + "“”‘’–—…«»"})


class VnTokenizer:
    """Pluggable Vietnamese tokenizer."""

    def __init__(self, backend: str = "pyvi", lowercase: bool = True, remove_punctuation: bool = True):
        self.backend = backend
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self._segmenter = None  # lazy

    # ---------------- public API ----------------

    def segment(self, text: str) -> str:
        """Return text with multi-syllable words joined by underscore.
        Example: 'giấy phép lái xe' -> 'giấy_phép_lái_xe'."""
        if not text:
            return ""
        text = self._clean(text)
        if self.backend == "pyvi":
            from pyvi import ViTokenizer
            return ViTokenizer.tokenize(text)
        if self.backend == "vncorenlp":
            seg = self._get_vncorenlp()
            sentences = seg.tokenize(text)  # List[List[str]]
            return " ".join(" ".join(s) for s in sentences)
        raise ValueError(f"Unknown tokenizer backend: {self.backend}")

    def tokenize(self, text: str) -> List[str]:
        """Return a flat list of tokens (multi-syllable words joined by underscore)."""
        segmented = self.segment(text)
        tokens = segmented.split()
        if self.remove_punctuation:
            tokens = [t for t in tokens if not all(c in string.punctuation for c in t)]
        return tokens

    # ---------------- internals ----------------

    def _clean(self, text: str) -> str:
        text = text.replace(" ", " ")  # NBSP
        text = re.sub(r"\s+", " ", text).strip()
        if self.lowercase:
            text = text.lower()
        if self.remove_punctuation:
            text = text.translate(_PUNCT_TABLE)
            text = re.sub(r"\s+", " ", text).strip()
        return text

    @lru_cache(maxsize=1)
    def _get_vncorenlp(self):
        from vncorenlp import VnCoreNLP
        jar = os.environ.get("VNCORENLP_JAR")
        if not jar or not os.path.exists(jar):
            raise RuntimeError(
                "Set VNCORENLP_JAR env var to the absolute path of VnCoreNLP-1.1.1.jar"
            )
        return VnCoreNLP(jar, annotators="wseg", max_heap_size="-Xmx2g")
