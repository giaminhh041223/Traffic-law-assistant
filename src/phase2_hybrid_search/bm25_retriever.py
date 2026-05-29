"""Sparse retriever — BM25 over Vietnamese-segmented tokens.

Why segment before BM25?
    BM25 treats every token as an atom. In Vietnamese, a single legal
    concept like "giấy phép lái xe" is FOUR space-separated syllables —
    if we whitespace-tokenize, BM25 indexes each syllable independently and
    a query for "GPLX" / "giấy phép lái xe" matches noisily.

    pyvi joins multi-syllable words with '_', so "giấy phép lái xe" becomes
    the single token "giấy_phép_lái_xe". BM25 then matches whole legal terms.

Filtering:
    rank_bm25 scores ALL documents in O(N·|query|). We sort by score,
    then apply user filters post-hoc. With a few thousand chunks this is
    instant; if you scale to millions of chunks, swap in PyTerrier / Pyserini.

Persistence:
    We pickle the whole retriever (BM25Okapi object + tokenised corpus +
    chunk_ids + original metadata). The pickle is the canonical sparse
    index — there is no separate "BM25 → JSON" representation worth keeping.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from rank_bm25 import BM25L, BM25Okapi, BM25Plus

from src.utils.vn_tokenizer import VnTokenizer

# Filter DSL — a single filter value can be a scalar OR a list ("any of").
FilterValue = Union[str, int, float, bool, List[Union[str, int, float, bool]]]
FilterDict = Dict[str, FilterValue]


@dataclass
class BM25RetrieverState:
    """Plain-data state — what we actually pickle (avoids pickling helper objects)."""
    bm25: Any                                # rank_bm25.BM25Okapi / BM25Plus / BM25L
    chunk_ids: List[str]
    tokenized_corpus: List[List[str]]
    metadata: List[Dict[str, Any]]
    variant: str = "okapi"
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25
    tokenizer_backend: str = "pyvi"
    lowercase: bool = True
    remove_punctuation: bool = True


class BM25Retriever:
    """BM25 over pyvi-segmented Vietnamese text, with metadata-filter post-pass."""

    def __init__(
        self,
        variant: str = "okapi",
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        tokenizer_backend: str = "pyvi",
        lowercase: bool = True,
        remove_punctuation: bool = True,
    ):
        if variant not in {"okapi", "plus", "l"}:
            raise ValueError("variant must be one of: okapi, plus, l")
        self.variant = variant
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.tokenizer = VnTokenizer(
            backend=tokenizer_backend,
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
        )
        self._tokenizer_backend = tokenizer_backend
        self._lowercase = lowercase
        self._remove_punctuation = remove_punctuation

        self._bm25: Any = None
        self._chunk_ids: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self._metadata: List[Dict[str, Any]] = []
        self._id_to_pos: Dict[str, int] = {}

    # ----------------------------- indexing -----------------------------

    def index(self, docs: List[Tuple[str, str, Dict[str, Any]]]) -> None:
        """Build the BM25 index.

        Args:
            docs: list of (chunk_id, text, metadata) tuples.
                  `metadata` is kept in its ORIGINAL shape (lists allowed)
                  so we can filter naturally at search time.
        """
        if not docs:
            raise ValueError("No documents provided to index.")
        self._chunk_ids = [d[0] for d in docs]
        self._tokenized_corpus = [self.tokenizer.tokenize(d[1]) for d in docs]
        self._metadata = [d[2] for d in docs]
        self._id_to_pos = {cid: i for i, cid in enumerate(self._chunk_ids)}

        bm25_cls = {"okapi": BM25Okapi, "plus": BM25Plus, "l": BM25L}[self.variant]
        if self.variant == "okapi":
            self._bm25 = bm25_cls(self._tokenized_corpus, k1=self.k1, b=self.b, epsilon=self.epsilon)
        else:
            self._bm25 = bm25_cls(self._tokenized_corpus, k1=self.k1, b=self.b)

    # ----------------------------- search -----------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[FilterDict] = None,
    ) -> List[Tuple[str, float]]:
        """Return [(chunk_id, score)] ranked by BM25, filtered by `filters`."""
        if self._bm25 is None:
            raise RuntimeError("BM25 index is empty. Call `index()` or `load()` first.")
        if not query.strip():
            return []

        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        # rank_bm25 returns a numpy array of scores aligned with corpus order.
        raw_scores = self._bm25.get_scores(query_tokens)

        # Sort all chunks by score desc, then filter, then truncate.
        # NOTE: BM25Okapi can produce NEGATIVE scores on tiny corpora where
        # common terms have df ≈ N (negative IDF). Relative ranking is still
        # meaningful, so we do NOT short-circuit on score ≤ 0 — top_k is the
        # only truncation signal. Downstream RRF uses ranks, not raw scores.
        ranked_idx = sorted(range(len(raw_scores)), key=lambda i: -raw_scores[i])

        out: List[Tuple[str, float]] = []
        for i in ranked_idx:
            if filters and not _passes_filters(self._metadata[i], filters):
                continue
            out.append((self._chunk_ids[i], float(raw_scores[i])))
            if len(out) >= top_k:
                break
        return out

    # ----------------------------- persistence -----------------------------

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = BM25RetrieverState(
            bm25=self._bm25,
            chunk_ids=self._chunk_ids,
            tokenized_corpus=self._tokenized_corpus,
            metadata=self._metadata,
            variant=self.variant,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon,
            tokenizer_backend=self._tokenizer_backend,
            lowercase=self._lowercase,
            remove_punctuation=self._remove_punctuation,
        )
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BM25Retriever":
        with open(path, "rb") as f:
            state: BM25RetrieverState = pickle.load(f)
        inst = cls(
            variant=state.variant,
            k1=state.k1,
            b=state.b,
            epsilon=state.epsilon,
            tokenizer_backend=state.tokenizer_backend,
            lowercase=state.lowercase,
            remove_punctuation=state.remove_punctuation,
        )
        inst._bm25 = state.bm25
        inst._chunk_ids = state.chunk_ids
        inst._tokenized_corpus = state.tokenized_corpus
        inst._metadata = state.metadata
        inst._id_to_pos = {cid: i for i, cid in enumerate(inst._chunk_ids)}
        return inst

    # ----------------------------- accessors -----------------------------

    def __len__(self) -> int:
        return len(self._chunk_ids)

    @property
    def chunk_ids(self) -> List[str]:
        return list(self._chunk_ids)


# ---------------------------------------------------------------------------
# Filter matcher — module-level so build_indexes.py can reuse it.
# ---------------------------------------------------------------------------
def _passes_filters(meta: Dict[str, Any], filters: FilterDict) -> bool:
    """Generic AND-of-clauses filter matcher.

    For each (key, expected) pair:
      * If chunk meta has a list at `key`, we treat `expected` as "any-of":
          - expected scalar  -> chunk passes if expected ∈ meta[key]
          - expected list    -> chunk passes if set(expected) ∩ set(meta[key]) ≠ ∅
      * If chunk meta has a scalar at `key`:
          - expected scalar  -> chunk passes iff equal
          - expected list    -> chunk passes if meta[key] ∈ expected
    """
    for key, expected in filters.items():
        actual = meta.get(key)
        if actual is None:
            return False
        if isinstance(actual, list):
            if isinstance(expected, list):
                if not set(actual) & set(expected):
                    return False
            else:
                if expected not in actual:
                    return False
        else:
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
    return True
