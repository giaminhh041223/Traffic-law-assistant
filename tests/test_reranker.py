"""Smoke tests for Phase 3 — Cross-Encoder Re-ranker + Hard-Negative Mining.

The real PhoRanker model is NOT loaded here (large download, slow). Instead:
    * `CrossEncoderReranker` accepts an injected `predictor` callable that
      stands in for the real CrossEncoder.predict(). We use this to drive
      deterministic re-ranking scenarios.
    * Hard-negative mining is tested against a tiny in-memory BM25 index
      so we can assert exact behaviour (positive exclusion, Jaccard dedup,
      random fill, manual negatives).
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.phase2_hybrid_search.bm25_retriever import BM25Retriever
from src.phase3_reranking.cross_encoder_reranker import CrossEncoderReranker
from src.phase3_reranking.hard_negative_mining import (
    TrainingPair,
    mine_hard_negatives,
)


# ===========================================================================
# Shared fixture: a tiny corpus that mimics the real chunks.jsonl shape.
# ===========================================================================
@pytest.fixture(scope="module")
def corpus() -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": "ND100_dieu5_khoan5_a",
            "dieu_title": "Điều 5. Xử phạt người điều khiển xe ô tô",
            "text": "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển "
                    "xe ô tô vượt đèn đỏ; không chấp hành hiệu lệnh đèn tín hiệu giao thông.",
            "chunk_type": "khoan",
            "vehicle_type": ["o_to"],
            "violation_type": ["vuot_den_do"],
        },
        {
            "chunk_id": "ND100_dieu6_khoan4_e",
            "dieu_title": "Điều 6. Xử phạt người điều khiển xe mô tô, xe gắn máy",
            "text": "Phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với người điều khiển "
                    "xe mô tô vượt đèn đỏ; không chấp hành hiệu lệnh đèn tín hiệu giao thông.",
            "chunk_type": "khoan",
            "vehicle_type": ["xe_may"],
            "violation_type": ["vuot_den_do"],
        },
        {
            "chunk_id": "ND100_dieu5_khoan10_a",
            "dieu_title": "Điều 5. Xử phạt người điều khiển xe ô tô",
            "text": "Phạt tiền từ 30.000.000 đồng đến 40.000.000 đồng đối với người điều khiển "
                    "xe ô tô vi phạm nồng độ cồn vượt quá 80 miligam/100 mililít máu.",
            "chunk_type": "khoan",
            "vehicle_type": ["o_to"],
            "violation_type": ["nong_do_con"],
        },
        {
            "chunk_id": "ND100_dieu6_khoan2_i",
            "dieu_title": "Điều 6. Xử phạt người điều khiển xe mô tô, xe gắn máy",
            "text": "Phạt tiền từ 400.000 đồng đến 600.000 đồng đối với người điều khiển "
                    "xe mô tô không đội mũ bảo hiểm khi tham gia giao thông.",
            "chunk_type": "khoan",
            "vehicle_type": ["xe_may"],
            "violation_type": ["khong_mu_bao_hiem"],
        },
        {
            "chunk_id": "ND100_dieu21_khoan7_a",
            "dieu_title": "Điều 21. Xử phạt các hành vi vi phạm về giấy phép lái xe",
            "text": "Phạt tiền từ 800.000 đồng đến 1.200.000 đồng đối với người điều khiển "
                    "xe mô tô không có giấy phép lái xe theo quy định.",
            "chunk_type": "khoan",
            "vehicle_type": ["xe_may"],
            "violation_type": ["khong_gplx"],
        },
    ]


@pytest.fixture(scope="module")
def chunks_by_id(corpus) -> Dict[str, Dict[str, Any]]:
    return {c["chunk_id"]: c for c in corpus}


@pytest.fixture(scope="module")
def bm25(corpus) -> BM25Retriever:
    r = BM25Retriever(k1=1.5, b=0.75)
    docs = [(c["chunk_id"], c["dieu_title"] + ". " + c["text"], c) for c in corpus]
    r.index(docs)
    return r


# ===========================================================================
# 1. CrossEncoderReranker — uses INJECTED predictor (no real model load).
# ===========================================================================
def _candidates(corpus) -> List[Dict[str, Any]]:
    """Hybrid-search-shaped candidate dicts (the input shape Phase 2 produces)."""
    return [dict(c, rrf_score=1.0 / (i + 1), bm25_rank=i + 1, dense_rank=i + 1)
            for i, c in enumerate(corpus)]


def test_reranker_orders_by_predictor_score(corpus):
    """The reranker must sort candidates by descending predictor output."""
    # Predictor returns scores in reverse order of input → last candidate becomes top.
    def fake_predictor(pairs):
        return [float(i) for i in range(len(pairs))]   # 0,1,2,3,4

    reranker = CrossEncoderReranker(predictor=fake_predictor, segment_input=False)
    out = reranker.rerank("Vượt đèn đỏ xe ô tô phạt bao nhiêu?", _candidates(corpus), top_k=3)
    assert len(out) == 3
    # Highest predictor score was index 4 → that candidate should be on top.
    assert out[0]["chunk_id"] == corpus[4]["chunk_id"]
    assert out[0]["cross_encoder_score"] == 4.0
    # pre_rerank_rank is preserved (1-indexed input position).
    assert out[0]["pre_rerank_rank"] == 5


def test_reranker_adds_score_field_and_keeps_metadata(corpus):
    reranker = CrossEncoderReranker(
        predictor=lambda pairs: [1.0] * len(pairs),
        segment_input=False,
    )
    out = reranker.rerank("xe ô tô", _candidates(corpus), top_k=10)
    assert len(out) == len(corpus)
    for c in out:
        # New fields written:
        assert "cross_encoder_score" in c
        assert "pre_rerank_rank" in c
        # Original metadata preserved:
        assert "dieu_title" in c
        assert "vehicle_type" in c


def test_reranker_empty_candidates_returns_empty():
    reranker = CrossEncoderReranker(predictor=lambda pairs: [], segment_input=False)
    assert reranker.rerank("any query", [], top_k=3) == []


def test_reranker_top_k_truncates(corpus):
    reranker = CrossEncoderReranker(
        predictor=lambda pairs: [float(i) for i in range(len(pairs))],
        segment_input=False,
    )
    out = reranker.rerank("q", _candidates(corpus), top_k=2)
    assert len(out) == 2


def test_reranker_score_count_mismatch_raises(corpus):
    reranker = CrossEncoderReranker(
        predictor=lambda pairs: [0.0],          # too few scores
        segment_input=False,
    )
    with pytest.raises(RuntimeError, match="returned 1 scores"):
        reranker.rerank("q", _candidates(corpus), top_k=3)


def test_reranker_build_doc_text_prefers_dieu_title(corpus):
    reranker = CrossEncoderReranker(predictor=lambda pairs: [0.0])
    text = reranker.build_doc_text(corpus[0])
    assert "Điều 5" in text
    assert "vượt đèn đỏ" in text.lower()


def test_reranker_build_doc_text_uses_linear_form_for_tables():
    reranker = CrossEncoderReranker(predictor=lambda pairs: [0.0])
    chunk = {
        "dieu_title": "Điều 5",
        "chunk_type": "table",
        "linear_form": "Hàng 1, cột Phạt tiền: 4-6 triệu đồng.",
        "text": "| Hành vi | Mức phạt |\n|---|---|\n",     # noisy markdown
    }
    text = reranker.build_doc_text(chunk)
    assert "Hàng 1" in text
    # Markdown noise should NOT leak in.
    assert "|" not in text


# ===========================================================================
# 2. Hard-Negative Mining
# ===========================================================================
def test_mining_excludes_positive(bm25, chunks_by_id):
    """The gold positive must never appear among the mined hard negatives."""
    pairs = [TrainingPair(
        query="Vượt đèn đỏ xe ô tô bị phạt bao nhiêu?",
        positive_chunk_id="ND100_dieu5_khoan5_a",
    )]
    mined = mine_hard_negatives(
        pairs=pairs, bm25=bm25, chunks_by_id=chunks_by_id,
        num_per_positive=3, bm25_pool_size=10,
        fill_with_random=False, random_seed=0,
    )
    assert len(mined) == 1
    ex = mined[0]
    assert ex.positive_chunk_id == "ND100_dieu5_khoan5_a"
    assert "ND100_dieu5_khoan5_a" not in ex.hard_negative_chunk_ids


def test_mining_finds_lexically_similar_distractor(bm25, chunks_by_id):
    """For 'xe ô tô vượt đèn đỏ', the xe-mô-tô / vượt-đèn-đỏ chunk is the
    expected hard negative (same violation, wrong vehicle)."""
    pairs = [TrainingPair(
        query="Vượt đèn đỏ xe ô tô bị phạt bao nhiêu?",
        positive_chunk_id="ND100_dieu5_khoan5_a",
    )]
    mined = mine_hard_negatives(
        pairs=pairs, bm25=bm25, chunks_by_id=chunks_by_id,
        num_per_positive=2, bm25_pool_size=10,
        fill_with_random=False, random_seed=0,
    )
    assert "ND100_dieu6_khoan4_e" in mined[0].hard_negative_chunk_ids


def test_mining_respects_extra_positive_ids(bm25, chunks_by_id):
    """extra_positive_ids must be excluded from negatives just like the primary positive."""
    pairs = [TrainingPair(
        query="Vượt đèn đỏ xe ô tô bị phạt bao nhiêu?",
        positive_chunk_id="ND100_dieu5_khoan5_a",
        extra_positive_ids=["ND100_dieu6_khoan4_e"],   # treat xe-máy version as also valid
    )]
    mined = mine_hard_negatives(
        pairs=pairs, bm25=bm25, chunks_by_id=chunks_by_id,
        num_per_positive=2, bm25_pool_size=10,
        fill_with_random=False, random_seed=0,
    )
    negs = mined[0].hard_negative_chunk_ids
    assert "ND100_dieu5_khoan5_a" not in negs
    assert "ND100_dieu6_khoan4_e" not in negs


def test_mining_uses_manual_negatives_first(bm25, chunks_by_id):
    pairs = [TrainingPair(
        query="Vượt đèn đỏ xe ô tô",
        positive_chunk_id="ND100_dieu5_khoan5_a",
        manual_negative_ids=["ND100_dieu21_khoan7_a"],   # not lexically similar
    )]
    mined = mine_hard_negatives(
        pairs=pairs, bm25=bm25, chunks_by_id=chunks_by_id,
        num_per_positive=3, bm25_pool_size=10,
        fill_with_random=False, random_seed=0,
    )
    ex = mined[0]
    # Manual negative must come first in the list (mined in step 1).
    assert ex.hard_negative_chunk_ids[0] == "ND100_dieu21_khoan7_a"
    assert ex.negative_sources[0] == "manual"


def test_mining_random_fill_when_bm25_too_small(bm25, chunks_by_id):
    """If we ask for more negatives than BM25 can supply, random fill kicks in."""
    pairs = [TrainingPair(
        query="Vượt đèn đỏ xe ô tô",
        positive_chunk_id="ND100_dieu5_khoan5_a",
    )]
    # 10 negatives requested, corpus only has 4 others available.
    mined = mine_hard_negatives(
        pairs=pairs, bm25=bm25, chunks_by_id=chunks_by_id,
        num_per_positive=10, bm25_pool_size=50,
        fill_with_random=True, random_seed=42,
    )
    ex = mined[0]
    # Cannot exceed unique corpus size minus the positive.
    assert len(ex.hard_negative_chunk_ids) == len(chunks_by_id) - 1
    assert "random" in ex.negative_sources or "bm25" in ex.negative_sources


def test_mining_skips_pair_with_unknown_positive(bm25, chunks_by_id):
    pairs = [
        TrainingPair(query="real", positive_chunk_id="ND100_dieu5_khoan5_a"),
        TrainingPair(query="ghost", positive_chunk_id="DOES_NOT_EXIST"),
    ]
    mined = mine_hard_negatives(
        pairs=pairs, bm25=bm25, chunks_by_id=chunks_by_id,
        num_per_positive=2, bm25_pool_size=10,
        fill_with_random=False, random_seed=0,
    )
    assert len(mined) == 1
    assert mined[0].query == "real"


def test_mining_dedup_jaccard_drops_near_duplicate(bm25):
    """If a candidate's text is near-identical to the positive (token Jaccard
    above the threshold), it must be skipped as a chunker artefact, not surfaced
    as a hard negative."""
    # Two chunks that are byte-for-byte identical (simulates double-split bug).
    base_text = ("Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển "
                 "xe ô tô vượt đèn đỏ.")
    chunks = [
        {"chunk_id": "POS",        "text": base_text,           "chunk_type": "khoan"},
        {"chunk_id": "DUPLICATE",  "text": base_text,           "chunk_type": "khoan"},
        {"chunk_id": "OTHER",
         "text": "Phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với xe mô tô vượt đèn đỏ.",
         "chunk_type": "khoan"},
    ]
    cid_to_chunk = {c["chunk_id"]: c for c in chunks}
    r = BM25Retriever()
    r.index([(c["chunk_id"], c["text"], c) for c in chunks])

    pairs = [TrainingPair(query="xe ô tô vượt đèn đỏ", positive_chunk_id="POS")]
    mined = mine_hard_negatives(
        pairs=pairs, bm25=r, chunks_by_id=cid_to_chunk,
        num_per_positive=2, bm25_pool_size=10,
        dedup_jaccard_threshold=0.85,
        fill_with_random=False, random_seed=0,
    )
    negs = mined[0].hard_negative_chunk_ids
    # DUPLICATE has Jaccard = 1.0 with POS — must be dropped.
    assert "DUPLICATE" not in negs
    # OTHER has lower overlap — must be kept.
    assert "OTHER" in negs


def test_mining_reports_provenance(bm25, chunks_by_id):
    """negative_sources must be parallel to hard_negative_chunk_ids."""
    pairs = [TrainingPair(
        query="Vượt đèn đỏ xe ô tô",
        positive_chunk_id="ND100_dieu5_khoan5_a",
        manual_negative_ids=["ND100_dieu21_khoan7_a"],
    )]
    mined = mine_hard_negatives(
        pairs=pairs, bm25=bm25, chunks_by_id=chunks_by_id,
        num_per_positive=3, bm25_pool_size=10,
        fill_with_random=False, random_seed=0,
    )
    ex = mined[0]
    assert len(ex.negative_sources) == len(ex.hard_negative_chunk_ids)
    assert set(ex.negative_sources).issubset({"manual", "bm25", "random"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
