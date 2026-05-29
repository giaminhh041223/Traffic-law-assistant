"""HybridSearcher — orchestrates BM25 + dense retrieval + RRF fusion.

Public surface area (one method):
    HybridSearcher.search(query, top_k=10, filters={}) -> List[Dict]

Each returned dict is a hydrated chunk (everything from Phase 1's
chunks.jsonl) plus four diagnostic fields:
    * rrf_score    — final fused score (sortable, comparable)
    * bm25_rank    — 1-indexed rank in BM25's list, or None
    * dense_rank   — 1-indexed rank in dense's list, or None
    * retrievers   — which retrievers found this chunk ("bm25" / "dense" / "both")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from src.utils.io import load_yaml, read_jsonl

from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .rrf_fusion import reciprocal_rank_fusion

FilterDict = Dict[str, Any]


class HybridSearcher:
    """Unified hybrid retrieval interface."""

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        chunks_by_id: Dict[str, Dict[str, Any]],
        rrf_k: int = 60,
        bm25_candidates: int = 50,
        dense_candidates: int = 50,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.chunks_by_id = chunks_by_id
        self.rrf_k = rrf_k
        self.bm25_candidates = bm25_candidates
        self.dense_candidates = dense_candidates
        self.weights = weights or {"bm25": 1.0, "dense": 1.0}

    # ----------------------------- factory -----------------------------

    @classmethod
    def from_configs(
        cls,
        settings_yaml: Union[str, Path] = "configs/settings.yaml",
        retrieval_yaml: Union[str, Path] = "configs/retrieval.yaml",
        chunks_jsonl: Optional[Union[str, Path]] = None,
        bm25_index_path: Optional[Union[str, Path]] = None,
    ) -> "HybridSearcher":
        settings = load_yaml(settings_yaml)
        retrieval = load_yaml(retrieval_yaml)

        chunks_jsonl = chunks_jsonl or settings["paths"]["chunks_file"]
        bm25_index_path = bm25_index_path or settings["paths"]["bm25_index"]

        # 1) Chunk lookup (used to hydrate final results)
        chunks_by_id: Dict[str, Dict[str, Any]] = {}
        for c in read_jsonl(chunks_jsonl):
            chunks_by_id[c["chunk_id"]] = c
        # Also include table chunks if present.
        tables_path = Path(settings["paths"].get("tables_file", ""))
        if tables_path.exists():
            for c in read_jsonl(tables_path):
                chunks_by_id[c["chunk_id"]] = c
        logger.info(f"HybridSearcher loaded {len(chunks_by_id):,} chunks from disk.")

        # 2) Retrievers
        bm25 = BM25Retriever.load(bm25_index_path)
        dense = DenseRetriever.from_config(retrieval)
        logger.info(f"BM25 corpus size: {len(bm25):,};  Dense collection size: {dense.count():,}")

        return cls(
            bm25=bm25,
            dense=dense,
            chunks_by_id=chunks_by_id,
            rrf_k=retrieval["rrf"]["k"],
            bm25_candidates=retrieval["top_k"]["bm25_candidates"],
            dense_candidates=retrieval["top_k"]["dense_candidates"],
            weights=retrieval["rrf"].get("weights", {"bm25": 1.0, "dense": 1.0}),
        )

    # ----------------------------- search -----------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[FilterDict] = None,
    ) -> List[Dict[str, Any]]:
        """Run both retrievers, fuse with RRF, hydrate results.

        Returns up to `top_k` chunks, sorted by RRF score desc.
        """
        bm25_hits = []
        try:
            bm25_hits = self.bm25.search(query, top_k=self.bm25_candidates, filters=filters)
        except Exception as e:
            logger.error(f"[hybrid_search] BM25 retrieval failed: {e}")

        dense_hits = []
        try:
            dense_hits = self.dense.search(query, top_k=self.dense_candidates, filters=filters)
        except Exception as e:
            logger.warning(f"[hybrid_search] Dense retrieval failed, falling back gracefully to BM25-only: {e}")

        fused = reciprocal_rank_fusion(
            [bm25_hits, dense_hits],
            k=self.rrf_k,
            weights=[self.weights.get("bm25", 1.0), self.weights.get("dense", 1.0)],
        )

        # Build rank lookups (1-indexed) for diagnostics.
        bm25_rank = {cid: i + 1 for i, (cid, _) in enumerate(bm25_hits)}
        dense_rank = {cid: i + 1 for i, (cid, _) in enumerate(dense_hits)}

        results: List[Dict[str, Any]] = []
        for cid, rrf_score in fused[:top_k]:
            chunk = self.chunks_by_id.get(cid)
            if chunk is None:
                logger.warning(f"chunk_id {cid} present in retrieval but missing in chunks lookup; skipping.")
                continue
            in_bm25 = cid in bm25_rank
            in_dense = cid in dense_rank
            results.append({
                **chunk,
                "rrf_score": rrf_score,
                "bm25_rank": bm25_rank.get(cid),
                "dense_rank": dense_rank.get(cid),
                "retrievers": "both" if (in_bm25 and in_dense) else ("bm25" if in_bm25 else "dense"),
            })
        return results

    # ----------------------------- convenience -----------------------------

    def search_bm25_only(self, query: str, top_k: int = 10, filters: Optional[FilterDict] = None):
        return self.bm25.search(query, top_k=top_k, filters=filters)

    def search_dense_only(self, query: str, top_k: int = 10, filters: Optional[FilterDict] = None):
        return self.dense.search(query, top_k=top_k, filters=filters)
