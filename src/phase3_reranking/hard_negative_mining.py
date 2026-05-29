"""Hard-Negative Mining — the academic centerpiece of Phase 3.

Why hard negatives, not random negatives?
    A cross-encoder trained with RANDOM negatives learns to discriminate
    only between "obviously relevant" and "obviously irrelevant" text.
    That's easy — random Vietnamese legal text shares almost no vocabulary
    with a given query, so the model converges quickly to shallow keyword
    matching and stops improving.

    HARD negatives are documents that are LEXICALLY similar to the query
    (high BM25 score) but NOT actually the gold answer. Examples:

        Query:    "Vượt đèn đỏ đối với xe ô tô bị phạt thế nào?"
        Gold:     "Khoản 5 Điều 5 ND100 — phạt 4-6 triệu cho xe ô tô…"
        Hard neg: "Khoản 6 Điều 6 ND100 — phạt 800k-1tr cho xe MÁY"
                  (same violation, WRONG vehicle — high BM25 overlap)

    Training on these forces the model to learn deeper distinctions
    (vehicle type, violation severity, fine bracket) that BM25 alone
    cannot capture. This is the standard recipe from
        Karpukhin et al., "Dense Passage Retrieval" (EMNLP 2020), §4.2.

Pipeline:
    1. Load training pairs:   {query, positive_chunk_id, [extra_positive_ids]}
    2. For each pair, BM25-search the query over the indexed corpus.
    3. Skip the positive(s) — what's left are top-ranked irrelevant docs.
    4. Optionally de-duplicate near-identical chunks via token Jaccard
       (a chunk that's textually almost the same as the positive is not
       a hard negative, it's a labelling problem).
    5. Take top-K as hard negatives. If fewer than K available, fill with
       random chunks.

The output of this module feeds directly into `train_cross_encoder.py`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from loguru import logger

from src.phase2_hybrid_search.bm25_retriever import BM25Retriever


@dataclass
class TrainingPair:
    """One labelled (query, positive) item, optionally with extra positives to exclude."""
    query: str
    positive_chunk_id: str
    extra_positive_ids: List[str] = field(default_factory=list)
    # Optional manual hard negatives (used as-is, no mining).
    manual_negative_ids: List[str] = field(default_factory=list)


@dataclass
class MinedExample:
    """The output of mining — ready for MultipleNegativesRankingLoss training."""
    query: str
    positive_text: str
    positive_chunk_id: str
    hard_negative_texts: List[str]
    hard_negative_chunk_ids: List[str]
    # Per-negative provenance for debugging / academic write-up.
    negative_sources: List[str] = field(default_factory=list)  # "bm25" | "manual" | "random"


# ---------------------------------------------------------------------------
# Token Jaccard — used to skip near-duplicate "hard negatives".
# ---------------------------------------------------------------------------
def _token_jaccard(a_tokens: Sequence[str], b_tokens: Sequence[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    sa, sb = set(a_tokens), set(b_tokens)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# The mining function
# ---------------------------------------------------------------------------
def mine_hard_negatives(
    pairs: List[TrainingPair],
    bm25: BM25Retriever,
    chunks_by_id: Dict[str, Dict[str, Any]],
    num_per_positive: int = 5,
    bm25_pool_size: int = 50,
    dedup_jaccard_threshold: float = 0.85,
    fill_with_random: bool = True,
    random_seed: int = 42,
) -> List[MinedExample]:
    """Mine hard negatives for every training pair.

    Args:
        pairs: training (query, positive_chunk_id) pairs.
        bm25:  the BM25 retriever from Phase 2 (already indexed over the corpus).
        chunks_by_id: lookup table id → full chunk dict.
        num_per_positive: how many hard negatives per positive (the K in MNR).
        bm25_pool_size: how deep to look in BM25 results before giving up.
        dedup_jaccard_threshold: discard negatives whose tokens overlap with
            the positive by more than this (likely near-duplicates / chunker artefacts).
        fill_with_random: if BM25 yields too few hard negatives, top up with
            random chunks from the corpus to keep the per-example shape regular.
        random_seed: for reproducible random fill-ins.

    Returns:
        list of MinedExample, one per input pair (positives that we cannot find
        any negatives for are dropped with a warning).
    """
    rng = random.Random(random_seed)
    all_chunk_ids = list(chunks_by_id.keys())
    if not all_chunk_ids:
        raise ValueError("chunks_by_id is empty.")

    out: List[MinedExample] = []
    dropped = 0

    for i, pair in enumerate(pairs):
        if pair.positive_chunk_id not in chunks_by_id:
            logger.warning(f"[pair {i}] positive_chunk_id '{pair.positive_chunk_id}' not in corpus — skipping.")
            dropped += 1
            continue

        positive_chunk = chunks_by_id[pair.positive_chunk_id]
        positive_text = _doc_text(positive_chunk)
        # Token set for Jaccard de-dup uses the BM25 tokenizer for consistency.
        positive_tokens = bm25.tokenizer.tokenize(positive_text)

        # IDs that must NEVER appear as a "hard negative":
        # the positive itself, any extra labelled positives, and any manual negatives
        # we'll add separately.
        forbidden: Set[str] = {pair.positive_chunk_id} | set(pair.extra_positive_ids)

        # ---- Step 1: manual negatives (user-curated, used as-is) ----
        hard_neg_ids: List[str] = []
        hard_neg_texts: List[str] = []
        sources: List[str] = []
        for nid in pair.manual_negative_ids:
            if nid not in chunks_by_id:
                logger.warning(f"[pair {i}] manual_negative_id '{nid}' not in corpus — skipping it.")
                continue
            hard_neg_ids.append(nid)
            hard_neg_texts.append(_doc_text(chunks_by_id[nid]))
            sources.append("manual")
            forbidden.add(nid)
            if len(hard_neg_ids) >= num_per_positive:
                break

        # ---- Step 2: BM25-mined hard negatives ----
        if len(hard_neg_ids) < num_per_positive:
            bm25_hits = bm25.search(pair.query, top_k=bm25_pool_size, filters=None)
            for cand_id, _score in bm25_hits:
                if cand_id in forbidden:
                    continue
                if cand_id not in chunks_by_id:
                    continue
                cand_text = _doc_text(chunks_by_id[cand_id])
                # De-dup against the positive.
                cand_tokens = bm25.tokenizer.tokenize(cand_text)
                jac = _token_jaccard(positive_tokens, cand_tokens)
                if jac >= dedup_jaccard_threshold:
                    # Likely a near-duplicate (e.g. same Khoản split twice in the chunker).
                    continue
                hard_neg_ids.append(cand_id)
                hard_neg_texts.append(cand_text)
                sources.append("bm25")
                forbidden.add(cand_id)
                if len(hard_neg_ids) >= num_per_positive:
                    break

        # ---- Step 3: fill with random negatives if requested ----
        if fill_with_random and len(hard_neg_ids) < num_per_positive:
            attempts = 0
            max_attempts = num_per_positive * 50
            while len(hard_neg_ids) < num_per_positive and attempts < max_attempts:
                attempts += 1
                cand_id = rng.choice(all_chunk_ids)
                if cand_id in forbidden:
                    continue
                hard_neg_ids.append(cand_id)
                hard_neg_texts.append(_doc_text(chunks_by_id[cand_id]))
                sources.append("random")
                forbidden.add(cand_id)

        if not hard_neg_ids:
            logger.warning(f"[pair {i}] could not mine any negatives for query: {pair.query!r} — skipping.")
            dropped += 1
            continue

        out.append(MinedExample(
            query=pair.query,
            positive_text=positive_text,
            positive_chunk_id=pair.positive_chunk_id,
            hard_negative_texts=hard_neg_texts,
            hard_negative_chunk_ids=hard_neg_ids,
            negative_sources=sources,
        ))

    logger.info(
        f"Mined {len(out)} examples from {len(pairs)} pairs "
        f"(dropped {dropped}). Avg negatives/example: "
        f"{sum(len(e.hard_negative_texts) for e in out) / max(1, len(out)):.2f}"
    )
    return out


def _doc_text(chunk: Dict[str, Any]) -> str:
    """Same construction used by inference reranker — keeps train/inference aligned."""
    parts: List[str] = []
    if chunk.get("dieu_title"):
        parts.append(chunk["dieu_title"])
    if chunk.get("chunk_type") == "table" and chunk.get("linear_form"):
        parts.append(chunk["linear_form"])
    else:
        parts.append(chunk.get("text", ""))
    return ". ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Convenience: load pairs from a JSONL file (schema documented separately).
# ---------------------------------------------------------------------------
def load_training_pairs(jsonl_path: str) -> List[TrainingPair]:
    from src.utils.io import read_jsonl
    pairs: List[TrainingPair] = []
    for rec in read_jsonl(jsonl_path):
        pairs.append(TrainingPair(
            query=rec["query"],
            positive_chunk_id=rec["positive_chunk_id"],
            extra_positive_ids=rec.get("extra_positive_ids", []),
            manual_negative_ids=rec.get("manual_negative_ids", []),
        ))
    return pairs
