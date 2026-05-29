"""Smoke tests for citation_validator.

We cover the cases that matter for the academic defence:
    * a perfectly-cited answer passes
    * an answer with a fabricated Điều fails
    * an answer with a swapped Khoản/Điều combination fails
    * the verbatim refusal phrase always passes (mandated by the prompt)
    * an answer with no citations fails when require_at_least_one=True
    * forbidden hedging phrases trip the validator
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.phase4_generation.citation_validator import (
    CitationValidator,
    ExtractedCitation,
    extract_citations,
)


def _chunk(doc_short: str, dieu: str, khoan: str, diem: str = None) -> Dict[str, Any]:
    return {"doc_short": doc_short, "dieu": dieu, "khoan": khoan, "diem": diem}


@pytest.fixture
def chunks() -> List[Dict[str, Any]]:
    return [
        _chunk("ND100", "5", "5", "a"),   # ô tô vượt đèn đỏ
        _chunk("ND100", "6", "4", "e"),   # xe máy vượt đèn đỏ
        _chunk("ND100", "5", "10", "a"),  # ô tô nồng độ cồn nặng
    ]


# ---------------------------------------------------------------------------
# Regex extraction
# ---------------------------------------------------------------------------
def test_extract_full_citation():
    text = "...bị phạt 4–6 triệu (Điểm a, Khoản 5, Điều 5 Nghị định 100/2019/NĐ-CP)."
    cits = extract_citations(text)
    assert len(cits) == 1
    c = cits[0]
    assert c.dieu == "5"
    assert c.khoan == "5"
    assert c.diem == "a"
    assert c.nghi_dinh == ("100", "2019")


def test_extract_handles_square_brackets():
    text = "[Khoản 2, Điều 6 Nghị định 100/2019/NĐ-CP]"
    cits = extract_citations(text)
    assert len(cits) == 1
    assert cits[0].dieu == "6"
    assert cits[0].khoan == "2"


def test_extract_multiple_citations():
    text = ("Lần đầu phạt 4 triệu (Khoản 5 Điều 5 Nghị định 100/2019/NĐ-CP), "
            "lần hai bị tước GPLX (Khoản 11 Điều 5 Nghị định 100/2019/NĐ-CP).")
    cits = extract_citations(text)
    assert len(cits) == 2
    assert {c.khoan for c in cits} == {"5", "11"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_validator_passes_well_cited_answer(chunks):
    v = CitationValidator(require_at_least_one=True)
    answer = ("Người điều khiển xe ô tô vượt đèn đỏ bị phạt từ 4 đến 6 triệu đồng "
              "(Điểm a, Khoản 5, Điều 5 Nghị định 100/2019/NĐ-CP).")
    report = v.validate(answer, chunks)
    assert report.is_valid
    assert len(report.supported_citations) == 1
    assert not report.hallucinated_citations
    assert not report.failures


def test_validator_flags_fabricated_dieu(chunks):
    v = CitationValidator(require_at_least_one=True)
    # Điều 99 doesn't exist in any chunk.
    answer = "Phạt 4 triệu (Khoản 3, Điều 99 Nghị định 100/2019/NĐ-CP)."
    report = v.validate(answer, chunks)
    assert not report.is_valid
    assert len(report.hallucinated_citations) == 1
    assert report.hallucinated_citations[0].dieu == "99"


def test_validator_flags_swapped_khoan_dieu(chunks):
    """Classic LLM mistake: gold is (Khoản 5, Điều 5); model emits (Khoản 5, Điều 6)."""
    v = CitationValidator(require_at_least_one=True)
    answer = "Phạt 4 triệu (Khoản 5, Điều 6 Nghị định 100/2019/NĐ-CP)."
    report = v.validate(answer, chunks)
    # (ND100, dieu=6, khoan=5) is NOT in the chunk set — should be flagged.
    assert not report.is_valid
    assert any(c.dieu == "6" and c.khoan == "5" for c in report.hallucinated_citations)


def test_validator_flags_wrong_diem_under_correct_khoan_dieu(chunks):
    """Same Khoản+Điều but the Điểm letter doesn't exist."""
    v = CitationValidator(require_at_least_one=True)
    answer = "Phạt 4 triệu (Điểm z, Khoản 5, Điều 5 Nghị định 100/2019/NĐ-CP)."
    report = v.validate(answer, chunks)
    assert not report.is_valid


def test_validator_accepts_refusal_phrase_without_citation(chunks):
    v = CitationValidator(require_at_least_one=True)
    refusal = ("Tôi không tìm thấy quy định phù hợp trong văn bản pháp luật "
               "được cung cấp để trả lời câu hỏi này.")
    report = v.validate(refusal, chunks)
    assert report.is_valid
    assert not report.failures


def test_validator_requires_citation_when_configured(chunks):
    v = CitationValidator(require_at_least_one=True)
    answer = "Việc này bị phạt rất nặng theo luật."
    report = v.validate(answer, chunks)
    assert not report.is_valid
    assert any("No legal citation" in f for f in report.failures)


def test_validator_optional_citation_passes_without_one(chunks):
    v = CitationValidator(require_at_least_one=False)
    report = v.validate("Đây là một câu trả lời không có trích dẫn.", chunks)
    assert report.is_valid


def test_validator_flags_forbidden_hedging_phrases(chunks):
    v = CitationValidator(
        require_at_least_one=True,
        forbidden_phrases=["theo tôi", "có thể là"],
    )
    answer = ("Theo tôi, lái xe ô tô vượt đèn đỏ bị phạt 4 triệu "
              "(Điểm a, Khoản 5, Điều 5 Nghị định 100/2019/NĐ-CP).")
    report = v.validate(answer, chunks)
    assert not report.is_valid
    assert "theo tôi" in report.forbidden_phrases_hit


def test_validator_accepts_dieu_only_citation(chunks):
    """When a chunk is granularity=dieu (no Khoản), Điều-only citation is acceptable."""
    v = CitationValidator(require_at_least_one=True)
    dieu_only_chunks = [{"doc_short": "ND100", "dieu": "7", "khoan": None, "diem": None}]
    answer = "Quy định tại Điều 7 Nghị định 100/2019/NĐ-CP."
    report = v.validate(answer, dieu_only_chunks)
    assert report.is_valid


def test_validator_flags_vehicle_type_mismatch():
    v = CitationValidator(require_at_least_one=True)
    
    # Chunk applies to "xe_dap"
    bike_chunks = [{
        "doc_short": "ND100",
        "dieu": "8",
        "khoan": "2",
        "diem": "đ",
        "metadata": {"vehicle_type": ["xe_dap"]}
    }]
    
    # Query is about pedestrian (người đi bộ)
    query = "Vậy đối với người đi bộ vượt đèn đỏ"
    
    # Answer cites bike chunk (Điều 8 Khoản 2 Điểm đ)
    answer = "Phạt từ 100k-200k (Điểm đ, Khoản 2, Điều 8 Nghị định 100/2019/NĐ-CP)."
    
    # Run validation with query
    report = v.validate(answer, bike_chunks, query=query)
    
    # It should fail due to vehicle type mismatch!
    assert not report.is_valid
    assert any("Subject/Vehicle Type Mismatch" in f for f in report.failures)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
