"""Smoke tests for retrieval_metrics — the IR side of Phase 5.

We deliberately don't load real retrievers / rerankers here. Stub retrievers
that return canned ID orderings let us assert exact metric values.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.phase5_evaluation.retrieval_metrics import (
    evaluate_retrieval,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    report_to_markdown,
)


# ===========================================================================
# Pure metric tests
# ===========================================================================
def test_recall_at_k_perfect_top1():
    assert recall_at_k(["a", "b", "c"], {"a"}, k=1) == 1.0
    assert recall_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0


def test_recall_at_k_multiple_gold():
    # 2 of 3 gold within top-3 → recall@3 = 2/3
    got = recall_at_k(["a", "b", "x"], {"a", "b", "c"}, k=3)
    assert abs(got - 2 / 3) < 1e-9


def test_recall_at_k_miss():
    assert recall_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0


def test_recall_at_k_empty_relevant_is_zero():
    assert recall_at_k(["a"], set(), k=1) == 0.0


def test_precision_at_k_basic():
    # 1 of 3 in top-3 is relevant → 1/3
    got = precision_at_k(["a", "x", "y"], {"a"}, k=3)
    assert abs(got - 1 / 3) < 1e-9


def test_precision_at_k_handles_k_larger_than_retrieved():
    # Only 2 retrieved, both relevant; k=5 → 2/2
    assert precision_at_k(["a", "b"], {"a", "b"}, k=5) == 1.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3


def test_reciprocal_rank_no_hit():
    assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0


def test_mean_reciprocal_rank_average():
    rankings = [
        (["a", "b"], {"a"}),   # RR = 1
        (["x", "a"], {"a"}),   # RR = 1/2
        (["x", "y"], {"a"}),   # RR = 0
    ]
    got = mean_reciprocal_rank(rankings)
    assert abs(got - (1 + 0.5 + 0) / 3) < 1e-9


# ===========================================================================
# evaluate_retrieval — Phase 2 vs Phase 3 comparison
# ===========================================================================
class _StubRetriever:
    """Always returns the same canned ranking, ignoring the query."""
    def __init__(self, returns: List[Dict[str, Any]]):
        self._returns = returns

    def search(self, query, top_k=10, filters=None):
        return list(self._returns)[:top_k]


class _StubReranker:
    """Reorders by an injected score table keyed on chunk_id."""
    def __init__(self, scores_by_id: Dict[str, float]):
        self.scores = scores_by_id

    def rerank(self, query, candidates, top_k=3):
        # Sort candidates by descending score.
        ranked = sorted(candidates, key=lambda c: -self.scores.get(c["chunk_id"], 0.0))
        return ranked[:top_k]


def _chunk(cid: str) -> Dict[str, Any]:
    return {"chunk_id": cid, "rrf_score": 1.0}


def test_evaluate_retrieval_reports_lift_after_rerank():
    # Two queries, both targeting the same gold chunk — keeps the arithmetic
    # easy to assert and still exercises the per-query averaging path.
    queries = [
        {"question": "q1", "relevant_chunk_ids": ["GOLD"]},
        {"question": "q2", "relevant_chunk_ids": ["GOLD"]},
    ]
    # Hybrid retriever puts GOLD at rank 5 — outside top-3, inside top-5.
    candidates = [_chunk("X"), _chunk("Y"), _chunk("Z"), _chunk("W"), _chunk("GOLD")]
    retriever = _StubRetriever(candidates)
    # Reranker floats GOLD to rank 1.
    reranker = _StubReranker({"GOLD": 10.0})

    report = evaluate_retrieval(
        queries=queries,
        retriever=retriever,
        reranker=reranker,
        k_values=(1, 3, 5),
        hybrid_top_k_for_baseline=5,
        hybrid_top_k_for_rerank=5,
        rerank_final_k=3,
    )

    # Hybrid: GOLD at position 5 → recall@5 = 1, recall@3 = 0, MRR = 1/5.
    assert report.hybrid.recall_at_k[5] == 1.0
    assert report.hybrid.recall_at_k[3] == 0.0
    assert abs(report.hybrid.mrr - 0.2) < 1e-9

    # Rerank: GOLD at rank 1 → recall@1 = 1, MRR = 1.
    assert report.rerank.recall_at_k[1] == 1.0
    assert report.rerank.mrr == 1.0

    # Lift summary makes the slide writable.
    lift = report.lift(3)
    assert lift["recall@3_hybrid"] == 0.0
    assert lift["recall@3_rerank"] == 1.0
    assert lift["mrr_abs_lift"] > 0


def test_evaluate_retrieval_without_reranker():
    queries = [{"question": "q", "relevant_chunk_ids": ["GOLD"]}]
    candidates = [_chunk("GOLD"), _chunk("X"), _chunk("Y")]
    report = evaluate_retrieval(
        queries=queries,
        retriever=_StubRetriever(candidates),
        reranker=None,
        k_values=(1, 3),
    )
    assert report.hybrid.recall_at_k[1] == 1.0
    assert report.rerank.n_queries == 0   # skipped


def test_evaluate_retrieval_skips_queries_without_gold():
    queries = [
        {"question": "valid", "relevant_chunk_ids": ["GOLD"]},
        {"question": "invalid", "relevant_chunk_ids": []},   # should be skipped
    ]
    candidates = [_chunk("GOLD")]
    report = evaluate_retrieval(
        queries=queries,
        retriever=_StubRetriever(candidates),
        reranker=None,
        k_values=(1,),
    )
    assert report.hybrid.n_queries == 1


def test_markdown_table_is_renderable():
    queries = [{"question": "q", "relevant_chunk_ids": ["GOLD"]}]
    candidates = [_chunk("GOLD")]
    report = evaluate_retrieval(
        queries=queries,
        retriever=_StubRetriever(candidates),
        reranker=None,
        k_values=(1, 3),
    )
    md = report_to_markdown(report, k_values=(1, 3))
    assert "Recall@1" in md and "MRR" in md and "Hybrid (Phase 2)" in md


def test_evaluate_retrieval_csv_round_trip(tmp_path):
    queries = [{"question": "q", "relevant_chunk_ids": ["GOLD"]}]
    report = evaluate_retrieval(
        queries=queries,
        retriever=_StubRetriever([_chunk("GOLD"), _chunk("X")]),
        reranker=None,
        k_values=(1, 3),
    )
    path = tmp_path / "out.csv"
    report.write_csv(path)
    text = path.read_text(encoding="utf-8")
    assert "hybrid_recall@1" in text
    assert "GOLD" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
