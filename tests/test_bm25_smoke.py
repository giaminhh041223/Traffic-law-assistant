"""Smoke test for BM25Retriever — verifies Vietnamese tokenisation, save/load, filters."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.phase2_hybrid_search.bm25_retriever import BM25Retriever


# Mini-corpus pulled directly from Phase 1's chunker test fixture.
_DOCS = [
    (
        "ND100__Dieu_5__Khoan_2",
        "Phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với người điều khiển xe ô tô "
        "thực hiện hành vi vượt đèn đỏ; không chấp hành hiệu lệnh của đèn tín hiệu giao thông.",
        {"doc_short": "ND100", "dieu": "5", "khoan": "2",
         "vehicle_type": ["o_to"], "violation_type": ["vuot_den_do"]},
    ),
    (
        "ND100__Dieu_5__Khoan_3",
        "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe ô tô "
        "vi phạm nồng độ cồn mà trong máu hoặc hơi thở có nồng độ cồn.",
        {"doc_short": "ND100", "dieu": "5", "khoan": "3",
         "vehicle_type": ["o_to"], "violation_type": ["nong_do_con"]},
    ),
    (
        "ND100__Dieu_6__Khoan_1",
        "Phạt tiền từ 100.000 đồng đến 200.000 đồng đối với người điều khiển xe mô tô "
        "không đội mũ bảo hiểm.",
        {"doc_short": "ND100", "dieu": "6", "khoan": "1",
         "vehicle_type": ["xe_may"], "violation_type": ["khong_mu_bao_hiem"]},
    ),
    (
        "ND100__Dieu_6__Khoan_2",
        "Phạt tiền từ 600.000 đồng đến 1.000.000 đồng đối với người điều khiển xe mô tô "
        "vi phạm nồng độ cồn.",
        {"doc_short": "ND100", "dieu": "6", "khoan": "2",
         "vehicle_type": ["xe_may"], "violation_type": ["nong_do_con"]},
    ),
]


@pytest.fixture(scope="module")
def bm25() -> BM25Retriever:
    r = BM25Retriever(k1=1.5, b=0.75)
    r.index(_DOCS)
    return r


def test_bm25_indexes_all_docs(bm25: BM25Retriever):
    assert len(bm25) == 4


def test_bm25_query_finds_nong_do_con(bm25: BM25Retriever):
    """A Vietnamese query about alcohol should retrieve the alcohol Khoản."""
    hits = bm25.search("nồng độ cồn xe ô tô", top_k=4)
    assert hits, "BM25 returned no results"
    top_id = hits[0][0]
    # Top hit should be the Điều 5 Khoản 3 (ô tô + nồng độ cồn)
    assert top_id == "ND100__Dieu_5__Khoan_3"


def test_bm25_filters_by_doc_and_vehicle(bm25: BM25Retriever):
    """Filter by vehicle_type → only xe_may chunks should be returned."""
    hits = bm25.search("nồng độ cồn", top_k=10, filters={"vehicle_type": "xe_may"})
    assert hits
    assert all(cid.startswith("ND100__Dieu_6") for cid, _ in hits)


def test_bm25_filters_by_scalar_dieu(bm25: BM25Retriever):
    hits = bm25.search("phạt tiền", top_k=10, filters={"dieu": "6"})
    assert hits
    assert all(cid.startswith("ND100__Dieu_6") for cid, _ in hits)


def test_bm25_filter_with_list_expected_any_of(bm25: BM25Retriever):
    hits = bm25.search("phạt tiền", top_k=10,
                       filters={"violation_type": ["nong_do_con", "khong_mu_bao_hiem"]})
    # Should NOT include the vượt đèn đỏ chunk (Dieu_5__Khoan_2).
    cids = {c for c, _ in hits}
    assert "ND100__Dieu_5__Khoan_2" not in cids


def test_bm25_save_load_round_trip(bm25: BM25Retriever):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bm25.pkl"
        bm25.save(path)
        assert path.exists() and path.stat().st_size > 0
        reloaded = BM25Retriever.load(path)
        # Identical results on same query.
        q = "vượt đèn đỏ"
        a = bm25.search(q, top_k=3)
        b = reloaded.search(q, top_k=3)
        assert a == b


def test_bm25_empty_query_returns_empty(bm25: BM25Retriever):
    assert bm25.search("", top_k=5) == []
    assert bm25.search("   ", top_k=5) == []


def test_bm25_tokenization_joins_multi_syllable_words(bm25: BM25Retriever):
    """Sanity: pyvi-segmented tokens contain underscore-joined VN words."""
    toks = bm25.tokenizer.tokenize("không đội mũ bảo hiểm cho người đi xe mô tô")
    joined = " ".join(toks)
    # 'mũ bảo hiểm' is a multi-syllable lexeme — pyvi should join it.
    assert any("_" in t for t in toks), f"Expected underscore-joined tokens, got: {toks}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
