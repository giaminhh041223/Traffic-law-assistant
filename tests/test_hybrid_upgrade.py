"""Unit tests for the Hybrid RAG System Upgrade components.

This verifies the query rewriter, the query deconstructor, the XML thought extractor,
and the AI router on CPU without calling external LLM APIs.
"""
from __future__ import annotations

import re
import pytest
from dataclasses import dataclass

from src.phase2_hybrid_search.query_rewriter import TrafficQueryRewriter
from src.phase2_hybrid_search.query_deconstructor import TrafficQueryDeconstructor
from src.phase4_generation.ai_router import TrafficAIRouter
from src.phase4_generation.rag_chain import RAGResult


# ===========================================================================
# 1. Module 1: Query Rewriter Tests
# ===========================================================================
def test_query_rewriter_mapping():
    rewriter = TrafficQueryRewriter()
    
    # Test case 1: slang terms
    q1 = "Tôi đi xe máy vượt đèn đỏ bị phạt thế nào?"
    r1 = rewriter.rewrite(q1)
    assert "xe mô tô" in r1
    assert "không chấp hành hiệu lệnh của đèn tín hiệu giao thông" in r1
    
    # Test case 2: other slang terms
    q2 = "đi ngược chiều không mang bằng lái bị phạt nhiêu"
    r2 = rewriter.rewrite(q2)
    assert "đi ngược chiều của đường một chiều" in r2
    assert "không mang theo Giấy phép lái xe" in r2

    # Test case 3: unaffected queries
    q3 = "Điều khiển xe ô tô đi vào đường cấm"
    r3 = rewriter.rewrite(q3)
    assert r3 == q3


# ===========================================================================
# 2. Module 4: Query Deconstructor Tests
# ===========================================================================
def test_query_deconstructor_splitting():
    deconstructor = TrafficQueryDeconstructor()
    
    # Compound query 1
    q1 = "vượt đèn đỏ và không có bằng lái xe máy"
    sub1 = deconstructor.deconstruct(q1)
    assert len(sub1) == 2
    assert any("vượt đèn đỏ" in s for s in sub1)
    assert any("không có bằng lái" in s for s in sub1)
    
    # Compound query 2 (with prefix injection)
    q2 = "Lái xe máy vượt đèn đỏ đồng thời uống rượu say gây tai nạn"
    sub2 = deconstructor.deconstruct(q2)
    assert len(sub2) == 3
    assert any("vượt đèn đỏ" in s for s in sub2)
    # Prefix "xe mô tô " should be injected because "xe máy" is in the main query
    # but not in the subsequent atomic parts
    assert any("xe mô tô uống rượu say" in s for s in sub2)
    assert any("xe mô tô gây tai nạn" in s for s in sub2)


# ===========================================================================
# 3. Module 5: Hybrid AI Router Tests
# ===========================================================================
def test_ai_router_classification():
    router = TrafficAIRouter()
    
    # Simple query -> local
    q_simple = "Vượt đèn đỏ phạt bao nhiêu xe máy?"
    assert router.evaluate_complexity(q_simple) == "local"
    
    # Complex query due to severe liability keywords -> cloud
    q_dispute = "Gây tai nạn giao thông dẫn đến tử vong phạt tù thế nào?"
    assert router.evaluate_complexity(q_dispute) == "cloud"
    
    # Complex query due to compound intent count >= 3 -> cloud
    q_compound = "Vượt đèn đỏ và không mang bằng lái cộng thêm đấm cán bộ"
    assert router.evaluate_complexity(q_compound) == "cloud"


# ===========================================================================
# 4. Module 2: XML CoT Thought Parser Tests
# ===========================================================================
def test_xml_cot_parsing():
    raw_response = (
        "<thought>\n"
        "- Xác định phương tiện: xe máy\n"
        "- Hành vi: vượt đèn đỏ\n"
        "</thought>\n"
        "Trả lời: Bị phạt từ 800.000 đến 1.000.000 đồng."
    )
    
    # Verify string regex extraction
    thought_match = re.search(r"<thought>(.*?)</thought>", raw_response, re.DOTALL | re.IGNORECASE)
    assert thought_match is not None
    thought_text = thought_match.group(1).strip()
    assert "Xác định phương tiện" in thought_text
    
    clean_answer = re.sub(r"<thought>.*?</thought>", "", raw_response, flags=re.DOTALL | re.IGNORECASE).strip()
    if clean_answer.startswith("Trả lời:"):
        clean_answer = clean_answer[len("Trả lời:"):].strip()
        
    assert clean_answer == "Bị phạt từ 800.000 đến 1.000.000 đồng."


def test_xml_cot_parsing_with_chatty_prefix():
    raw_response = (
        "Xin chào! Dưới đây là phân tích và câu trả lời của tôi:\n"
        "<thought>\n"
        "- Xác định phương tiện: xe máy\n"
        "- Hành vi: vượt đèn đỏ\n"
        "</thought>\n"
        "Trả lời: Bị phạt từ 800.000 đến 1.000.000 đồng."
    )
    
    # Verify string regex extraction
    thought_match = re.search(r"<thought>(.*?)</thought>", raw_response, re.DOTALL | re.IGNORECASE)
    assert thought_match is not None
    thought_text = thought_match.group(1).strip()
    assert "Xác định phương tiện" in thought_text
    
    # Test our new robust splitting logic
    parts = re.split(r"</thought>", raw_response, flags=re.IGNORECASE)
    clean_answer = parts[-1].strip()
    if clean_answer.startswith("Trả lời:"):
        clean_answer = clean_answer[len("Trả lời:"):].strip()
        
    assert clean_answer == "Bị phạt từ 800.000 đến 1.000.000 đồng."
