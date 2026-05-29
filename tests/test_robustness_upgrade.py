"""Unit and regression tests for the Vietnamese Traffic Law RAG Robustness Upgrade.

Verifies the expanded rewriter slang dictionary, deconstructor splitting improvements, 
system prompt updates, and vehicle mismatch validation.
"""
from __future__ import annotations

import re
import pytest

from src.phase2_hybrid_search.query_rewriter import TrafficQueryRewriter
from src.phase2_hybrid_search.query_deconstructor import TrafficQueryDeconstructor
from src.phase4_generation.citation_validator import CitationValidator, _detect_vehicle_types
from src.phase4_generation.prompt_templates import build_system_prompt


# ===========================================================================
# 1. Slang Translation (Query Rewriter) Tests
# ===========================================================================
def test_rewriter_translates_slang_terms():
    rewriter = TrafficQueryRewriter()
    
    # Test Query 1 slang
    q1 = "Hôm qua đi con xe cọp bị bồ câu tuýt lại vì thông chốt với nẹt pô không mang tờ sớ"
    rewritten1 = rewriter.rewrite(q1)
    
    assert "xe mô tô độ, xe mô tô thay đổi kết cấu" in rewritten1
    assert "Cảnh sát giao thông" in rewritten1
    assert "không chấp hành hiệu lệnh, chỉ dẫn" in rewritten1
    assert "rú ga liên tục, nẹt pô, rú còi" in rewritten1
    assert "Giấy phép lái xe" in rewritten1
    
    # Test Query 2 slang
    q2 = "Vừa làm vài chén rượu mà cưỡi con Wave tàu đi ngược chiều bị giam xe"
    rewritten2 = rewriter.rewrite(q2)
    
    assert "trong máu hoặc hơi thở có nồng độ cồn" in rewritten2
    assert "xe mô tô" in rewritten2
    assert "tạm giữ phương tiện" in rewritten2


# ===========================================================================
# 2. Query Deconstructor Splitting Tests
# ===========================================================================
def test_deconstructor_splits_on_periods_and_new_keywords():
    deconstructor = TrafficQueryDeconstructor()
    
    # Test Query 3: Multi-intent split
    q3 = "Tôi đi ô tô vượt đèn đỏ. Không thắt dây an toàn. Lại còn đang nghe điện thoại bằng tay. Không có bảo hiểm bắt buộc."
    sub_queries = deconstructor.deconstruct(q3)
    
    # It should split into 4 atomic intents due to period punctuation
    assert len(sub_queries) >= 4
    assert any("vượt đèn đỏ" in sq for sq in sub_queries)
    assert any("dây an toàn" in sq for sq in sub_queries)
    assert any("điện thoại" in sq for sq in sub_queries)
    assert any("bảo hiểm" in sq for sq in sub_queries)
    
    # Test vehicle prefix propagation
    for sq in sub_queries:
        assert "xe ô tô" in sq.lower()


# ===========================================================================
# 3. Prompt & LLM Guardrails Tests
# ===========================================================================
def test_system_prompt_contains_upgrades():
    sys_prompt = build_system_prompt()
    
    # Assert Xe đạp điện guideline update
    assert "xe đạp điện" in sys_prompt.lower()
    assert "áp dụng chung khung hình phạt của xe đạp" in sys_prompt.lower()
    
    # Assert anti-leading query guideline
    assert "CẢNH BÁO CHỐNG BỊ DẪN DẮT" in sys_prompt
    assert "CONSTRAINED PENALTY ENFORCEMENT" in sys_prompt
    assert "400.000 đồng đến 600.000 đồng" in sys_prompt


# ===========================================================================
# 4. Vehicle Type Alignment & Detection Tests
# ===========================================================================
def test_detect_vehicle_types_helper():
    assert "nguoi_di_bo" in _detect_vehicle_types("Vậy đối với người đi bộ vượt đèn đỏ")
    assert "xe_dap" in _detect_vehicle_types("Tôi đi xe đạp điện có bị phạt cồn")
    assert "xe_may" in _detect_vehicle_types("Lái xe mô tô Wave tàu nẹt pô")
    assert "o_to" in _detect_vehicle_types("Tôi đi xe hơi lùi trên cao tốc")


def test_validator_detects_subject_mismatch():
    v = CitationValidator(require_at_least_one=True)
    
    # Chunk only applies to "xe_may" (motorcycle)
    moto_chunks = [{
        "doc_short": "ND100",
        "dieu": "6",
        "khoan": "4",
        "diem": "e",
        "metadata": {"vehicle_type": ["xe_may"]}
    }]
    
    # Query asks about ô tô (car)
    query = "Lái xe ô tô vượt đèn đỏ phạt thế nào?"
    answer = "Bị phạt tiền từ 800k đến 1 triệu (Điểm e, Khoản 4, Điều 6 Nghị định 100/2019/NĐ-CP)."
    
    # This should fail validation due to vehicle type mismatch!
    report = v.validate(answer, moto_chunks, query=query)
    assert not report.is_valid
    assert any("Subject/Vehicle Type Mismatch" in f for f in report.failures)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
