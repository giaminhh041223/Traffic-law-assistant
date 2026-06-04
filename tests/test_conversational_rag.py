"""Unit tests for the Context-Aware Conversational RAG & Session Persistence features.

Verifies the session manager, query reformulator, and full RAG integration on CPU without calling real LLM APIs.
"""
from __future__ import annotations

import os
from pathlib import Path
import pytest

from src.utils import session_manager
from src.phase2_hybrid_search.query_reformulator import TrafficQueryReformulator
from src.phase4_generation.llm_loader import LLMClient
from src.phase4_generation.rag_chain import RAGResult, TrafficLawRAG
from src.phase4_generation.citation_validator import ValidationReport, ExtractedCitation, CitationValidator
from src.phase3_reranking.cross_encoder_reranker import CrossEncoderReranker


# ===========================================================================
# 1. Session Manager Serialization/Deserialization Tests
# ===========================================================================
def test_session_manager_serialization_lifecycle(tmp_path, monkeypatch):
    # Redirect HISTORY_DIR to temp folder
    monkeypatch.setattr(session_manager, "HISTORY_DIR", tmp_path)
    
    # Create session
    sid = session_manager.create_session("Tôi đi xe máy đúng làn...")
    
    # Mock RAGResult with custom nested dataclasses
    cit = ExtractedCitation(dieu="5", khoan="5", diem="a", nghi_dinh=("100", "2019"))
    validation = ValidationReport(
        is_valid=True,
        citations_found=[cit],
        supported_citations=[cit],
        hallucinated_citations=[],
        failures=[],
        forbidden_phrases_hit=[]
    )
    res = RAGResult(
        query="Tôi lái ô tô vượt đèn đỏ phạt bao nhiêu?",
        answer="Bị phạt từ 4-6 triệu.",
        thought="Lập luận: xe ô tô vượt đèn đỏ...",
        sources=[{"chunk_id": "test_chunk", "text": "Đoạn văn bản mẫu..."}],
        validation=validation,
        timings={"retrieve": 0.1, "generate": 0.5},
        usage={"prompt_tokens": 100, "completion_tokens": 50},
        refused_due_to_low_confidence=False
    )
    
    messages = [
        {"role": "user", "content": "Tôi lái ô tô vượt đèn đỏ phạt bao nhiêu?", "result": None},
        {"role": "assistant", "content": "Bị phạt từ 4-6 triệu.", "result": res}
    ]
    
    # Save session
    session_data = {
        "conversation_id": sid,
        "title": "Tôi đi xe máy đúng làn...",
        "timestamp": "2026-05-29T11:20:00",
        "messages": messages
    }
    session_manager.save_session(sid, session_data)
    
    # Load session and assert Reconstruction
    loaded = session_manager.load_session(sid)
    assert loaded["conversation_id"] == sid
    assert loaded["title"] == "Tôi đi xe máy đúng làn..."
    assert len(loaded["messages"]) == 2
    
    # Reconstructed RAGResult assertion
    loaded_res = loaded["messages"][1]["result"]
    assert isinstance(loaded_res, RAGResult)
    assert loaded_res.query == "Tôi lái ô tô vượt đèn đỏ phạt bao nhiêu?"
    assert loaded_res.thought == "Lập luận: xe ô tô vượt đèn đỏ..."
    assert loaded_res.timings["retrieve"] == 0.1
    
    # Reconstructed nested ValidationReport assertion
    assert isinstance(loaded_res.validation, ValidationReport)
    assert loaded_res.validation.is_valid is True
    
    # Reconstructed deeply nested ExtractedCitation assertion
    loaded_cit = loaded_res.validation.citations_found[0]
    assert isinstance(loaded_cit, ExtractedCitation)
    assert loaded_cit.dieu == "5"
    assert loaded_cit.nghi_dinh == ("100", "2019")  # Reconstructed tuple from list
    
    # Clean up deletion
    session_manager.delete_session(sid)
    assert not (tmp_path / f"{sid}.json").exists()


# ===========================================================================
# 2. Query Reformulator Tests
# ===========================================================================
def test_query_reformulator_resolves_pronouns():
    def fake_predictor(messages):
        # Assert formatting of history in the prompt
        prompt = messages[-1]["content"]
        assert "- Người dùng: Tôi đi xe máy đúng làn đường" in prompt
        assert "- Trợ lý: Bạn không bị phạt" in prompt
        assert "Câu hỏi mới của người dùng: Thế còn ô tô?" in prompt
        return "Lái xe ô tô đi đúng làn đường thì có bị phạt không?"

    llm = LLMClient(backend="stub", config={}, predictor=fake_predictor)
    reformulator = TrafficQueryReformulator(llm)
    
    chat_history = [
        {"role": "user", "content": "Tôi đi xe máy đúng làn đường thì có bị phạt không?"},
        {"role": "assistant", "content": "Bạn không bị phạt (0 đồng) theo Điểm i Khoản 1 Điều 6."}
    ]
    
    rewritten = reformulator.reformulate("Thế còn ô tô?", chat_history)
    assert rewritten == "Lái xe ô tô đi đúng làn đường thì có bị phạt không?"



def test_query_reformulator_skips_empty_history():
    llm = LLMClient(backend="stub", config={}, predictor=lambda m: "should not be called")
    reformulator = TrafficQueryReformulator(llm)
    
    assert reformulator.reformulate("Tôi đi xe máy đúng làn", []) == "Tôi đi xe máy đúng làn"


# ===========================================================================
# 3. Conversational RAG Pipeline Integration Tests
# ===========================================================================
class _StubRetriever:
    def search(self, query: str, top_k: int = 10, filters=None):
        return [{"chunk_id": "test", "dieu": "5", "khoan": "5", "doc_short": "ND100", "text": "Phạt 4-6 triệu..."}]


def test_full_conversational_rag_pipeline():
    captured_messages = []

    def fake_llm(messages):
        # We capture the prompt messages passed to the generator
        for m in messages:
            captured_messages.append(m)
        return "Bạn không bị phạt xe máy (0 đồng)."

    llm = LLMClient(backend="stub", config={}, predictor=fake_llm)
    
    # Reranker returns input candidates
    reranker = CrossEncoderReranker(predictor=lambda pairs: [1.0] * len(pairs), segment_input=False)
    
    rag = TrafficLawRAG(
        retriever=_StubRetriever(),
        reranker=reranker,
        llm=llm,
        validator=CitationValidator(require_at_least_one=False),
        hybrid_top_k=10,
        final_top_k=3
    )
    
    # Mock active history
    chat_history = [
        {"role": "user", "content": "Chào bạn!"},
        {"role": "assistant", "content": "Tôi là trợ lý pháp luật giao thông."}
    ]
    
    # Run pipeline
    result = rag.run("Tôi đi xe máy đúng làn đường thì có bị phạt không?", chat_history=chat_history)
    
    # Assertions
    assert result.answer == "Bạn không bị phạt xe máy (0 đồng)."
    assert "reformulate" in result.timings
    
    # Assert history was injected in the generator prompt messages list
    assert any("Chào bạn!" in m["content"] for m in captured_messages if m["role"] == "user")
    assert any("Tôi là trợ lý pháp luật giao thông." in m["content"] for m in captured_messages if m["role"] == "assistant")
