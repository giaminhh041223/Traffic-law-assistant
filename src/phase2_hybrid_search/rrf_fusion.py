"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

Why RRF, not weighted-sum on raw scores?
    BM25 scores and cosine similarities live on entirely different scales —
    BM25 is unbounded above and depends on corpus statistics, while cosine
    sits in [-1, 1]. Mixing them with a weighted sum requires score
    normalisation (z-score, min-max, ...) that is brittle and dataset-
    specific. RRF only uses RANK position, so the scales of the underlying
    methods don't matter.

Formula:
    RRF(d) = Σ_i  w_i / (k + rank_i(d))

    where rank_i(d) is the 1-indexed rank of d in the i-th input list,
    or ∞ if d does not appear (the term then vanishes — naturally handles
    "retrieved by one method only" without any special-case code).

Defaults:
    k = 60   — Cormack et al.'s original choice. The constant dampens the
               contribution of top-1 dominating the fusion; smaller k makes
               the top of each list count more.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

# Ranked-list element: (chunk_id, original_score)
RankedItem = Tuple[str, float]
RankedList = List[RankedItem]


def reciprocal_rank_fusion(
    results_lists: Iterable[RankedList],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Tuple[str, float]]:
    """Fuse N ranked lists using Reciprocal Rank Fusion.

    Args:
        results_lists: iterable of ranked result lists, each [(chunk_id, score), ...]
                       in rank order (best first). The original `score` is IGNORED
                       — only RANK position is used.
        k: dampening constant (default 60).
        weights: optional per-list weight (e.g. [bm25=1.0, dense=1.0]). Must match
                 the number of result lists, or None (uniform).

    Returns:
        Fused list of (chunk_id, rrf_score) sorted by rrf_score desc.

    Edge case handled implicitly:
        If a chunk appears in only one input list, it still gets a (partial)
        score from that list — no special-casing needed.
    """
    lists = list(results_lists)
    if not lists:
        return []
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) != len(lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match number of result lists ({len(lists)})"
        )

    scores: Dict[str, float] = defaultdict(float)
    for w, results in zip(weights, lists):
        for rank, (doc_id, _orig_score) in enumerate(results, start=1):
            scores[doc_id] += w / (k + rank)

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def explain(
    chunk_id: str,
    results_lists: Iterable[RankedList],
    k: int = 60,
    weights: Optional[List[float]] = None,
    list_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Debug helper: return per-list rank + contribution for one chunk_id."""
    lists = list(results_lists)
    if weights is None:
        weights = [1.0] * len(lists)
    if list_names is None:
        list_names = [f"list_{i}" for i in range(len(lists))]

    out: Dict[str, float] = {}
    total = 0.0
    for name, w, results in zip(list_names, weights, lists):
        rank = next((i + 1 for i, (cid, _) in enumerate(results) if cid == chunk_id), None)
        if rank is None:
            out[f"{name}_rank"] = float("inf")
            out[f"{name}_contrib"] = 0.0
        else:
            contrib = w / (k + rank)
            out[f"{name}_rank"] = float(rank)
            out[f"{name}_contrib"] = contrib
            total += contrib
    out["rrf_total"] = total
    return out
