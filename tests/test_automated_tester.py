"""Unit tests for the Automated Evaluation Framework.

Verifies test parsing, metric calculations, and evaluator robust execution.
"""
from __future__ import annotations

import pytest
from src.phase5_evaluation.automated_tester import TrafficLawAutomatedTester


def test_evaluator_metric_helpers():
    # We initialize the tester with a dummy path since we only want to test pure helper functions
    # without spinning up the entire PyTorch/SBERT/Ollama pipeline.
    tester = object.__new__(TrafficLawAutomatedTester)
    
    # 1. Test parse_expected_fines
    gt_text = "Phạt tiền từ 400.000 đồng đến 600.000 đồng và cộng thêm 2.600.000 đồng."
    fines = tester.parse_expected_fines(gt_text)
    assert "400.000" in fines
    assert "600.000" in fines
    assert "2.600.000" in fines
    
    # 2. Test parse_expected_citations
    behavior_text = "Retrieval should fetch Điều 6 Khoản 3 Điểm c Nghị định 100/2019/NĐ-CP."
    truth_text = "Căn cứ QCVN 41 và Luật TTATGT."
    cits = tester.parse_expected_citations(behavior_text, truth_text)
    
    assert "dieu_6" in cits
    assert "khoan_3" in cits
    assert "diem_c" in cits
    assert "QCVN_41" in cits
    assert "LUAT_TTATGT" in cits

    # 3. Test evaluate_retrieval_precision
    retrieved_chunks = [
        {"chunk_id": "ND100_123__Dieu_6__Khoan_3__Diem_c", "dieu": "6", "khoan": "3", "diem": "c"},
        {"chunk_id": "QCVN41_2024__Dieu_8", "text": "QCVN 41 rules"}
    ]
    expected_cits = {"dieu_6", "khoan_3", "QCVN_41", "LUAT_TTATGT"}
    
    # dieu_6, khoan_3, QCVN_41 should match. LUAT_TTATGT will not match.
    # Score should be 3/4 = 0.75
    score = tester.evaluate_retrieval_precision(retrieved_chunks, expected_cits)
    assert score == 0.75

    # 4. Test evaluate_calculation_accuracy
    actual_ans = "Mức phạt lớn nhất bạn phải chịu là tổng cộng 2.600.000đ và thêm 400.000 đồng."
    # Matches since both clean '2600000' and '400000' are in actual clean answer
    assert tester.evaluate_calculation_accuracy(actual_ans, ["2.600.000", "400.000"]) == 1.0
    # Mismatch since '600000' is not in actual clean answer
    assert tester.evaluate_calculation_accuracy(actual_ans, ["2.600.000", "600.000"]) == 0.0
