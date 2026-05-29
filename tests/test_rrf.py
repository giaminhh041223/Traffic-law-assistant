"""Tests for Reciprocal Rank Fusion (Phase 2)."""
from __future__ import annotations

import pytest

from src.phase2_hybrid_search.rrf_fusion import explain, reciprocal_rank_fusion


def test_rrf_basic_two_lists_overlap():
    bm25  = [("A", 9.0), ("B", 7.0), ("C", 5.0)]
    dense = [("B", 0.91), ("A", 0.89), ("D", 0.80)]
    fused = reciprocal_rank_fusion([bm25, dense], k=60)
    # A and B appear in both; C only in BM25; D only in dense.
    # Top of fused list must be A or B (the "both" ones).
    top_ids = [cid for cid, _ in fused[:2]]
    assert set(top_ids) == {"A", "B"}
    # C and D should both score lower than A and B.
    ids = [cid for cid, _ in fused]
    assert ids.index("C") > 1
    assert ids.index("D") > 1


def test_rrf_chunk_only_in_one_list_still_scored():
    """An item appearing only in one list still gets a (partial) RRF score."""
    bm25  = [("X", 5.0)]
    dense = [("Y", 0.9)]
    fused = dict(reciprocal_rank_fusion([bm25, dense], k=60))
    assert "X" in fused
    assert "Y" in fused
    # Both at rank 1 of their list → identical scores (uniform weights).
    assert fused["X"] == pytest.approx(fused["Y"])


def test_rrf_weights_affect_ranking():
    bm25  = [("A", 1.0), ("B", 1.0)]
    dense = [("B", 1.0), ("A", 1.0)]
    # Uniform weights: A and B tied
    fused_uniform = dict(reciprocal_rank_fusion([bm25, dense], k=60))
    assert fused_uniform["A"] == pytest.approx(fused_uniform["B"])
    # Bias toward dense → B should win (B is rank-1 in dense)
    fused_biased = reciprocal_rank_fusion([bm25, dense], k=60, weights=[0.5, 2.0])
    assert fused_biased[0][0] == "B"


def test_rrf_smaller_k_makes_top_more_dominant():
    bm25  = [("A", 1.0), ("B", 1.0)]
    dense = [("C", 1.0), ("D", 1.0)]
    score_a_k1   = dict(reciprocal_rank_fusion([bm25, dense], k=1))["A"]
    score_a_k1k  = dict(reciprocal_rank_fusion([bm25, dense], k=1000))["A"]
    # With k=1, A (rank 1 in bm25) contributes 1/(1+1)=0.5
    # With k=1000, A contributes 1/(1000+1)≈0.001
    assert score_a_k1 > score_a_k1k


def test_rrf_empty_lists_safe():
    fused = reciprocal_rank_fusion([[], []], k=60)
    assert fused == []


def test_rrf_mismatched_weights_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[("A", 1.0)], [("B", 1.0)]], k=60, weights=[1.0])


def test_explain_helper_returns_per_list_breakdown():
    bm25  = [("A", 1.0), ("B", 1.0)]
    dense = [("B", 1.0), ("A", 1.0)]
    e = explain("A", [bm25, dense], k=60, list_names=["bm25", "dense"])
    assert e["bm25_rank"] == 1.0
    assert e["dense_rank"] == 2.0
    assert e["rrf_total"] == pytest.approx(1 / 61 + 1 / 62)


def test_explain_missing_chunk():
    e = explain("Z", [[("A", 1.0)], [("B", 1.0)]], k=60, list_names=["bm25", "dense"])
    assert e["bm25_rank"] == float("inf")
    assert e["dense_rank"] == float("inf")
    assert e["rrf_total"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
