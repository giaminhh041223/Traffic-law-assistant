"""Smoke tests for the Phase 4 prompt and the full RAG chain.

We DO NOT load a real LLM, real BM25, real cross-encoder, or real ChromaDB
here. Each stage is constructed inline with stubs / injectables so the entire
pipeline can be exercised in a couple of milliseconds.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.phase3_reranking.cross_encoder_reranker import CrossEncoderReranker
from src.phase4_generation.citation_validator import CitationValidator
from src.phase4_generation.llm_loader import LLMClient
from src.phase4_generation.prompt_templates import (
    SYSTEM_PROMPT_VI,
    build_messages,
    build_system_prompt,
    build_user_prompt,
    format_chunks_for_prompt,
)
from src.phase4_generation.rag_chain import TrafficLawRAG


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def chunks() -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": "ND100_dieu5_khoan5_a",
            "doc_short": "ND100",
            "dieu": "5", "khoan": "5", "diem": "a",
            "dieu_title": "Điều 5. Xử phạt người điều khiển xe ô tô",
            "text": "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển "
                    "xe ô tô vượt đèn đỏ.",
            "chunk_type": "khoan",
        },
        {
            "chunk_id": "ND100_dieu6_khoan4_e",
            "doc_short": "ND100",
            "dieu": "6", "khoan": "4", "diem": "e",
            "dieu_title": "Điều 6. Xử phạt người điều khiển xe mô tô",
            "text": "Phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với người điều khiển "
                    "xe mô tô vượt đèn đỏ.",
            "chunk_type": "khoan",
        },
    ]


# ===========================================================================
# 1. Prompt construction
# ===========================================================================
def test_system_prompt_is_constant_and_strict():
    sp = build_system_prompt()
    assert sp == SYSTEM_PROMPT_VI
    # The non-negotiables must appear verbatim:
    assert "Trợ lý pháp luật giao thông Việt Nam" in sp
    assert "<context>" in sp
    assert "Tôi không tìm thấy quy định phù hợp" in sp
    assert "Nghị định 100/2019/NĐ-CP" in sp


def test_format_chunks_renders_citation_tag(chunks):
    out = format_chunks_for_prompt(chunks)
    # Each chunk gets a [Đoạn N] [citation tag] header.
    assert "[Đoạn 1]" in out
    assert "[Đoạn 2]" in out
    # The exact citation string the LLM is expected to copy back:
    assert "Điểm a" in out and "Khoản 5" in out and "Điều 5" in out
    assert "Nghị định 100/2019/NĐ-CP" in out


def test_format_chunks_uses_linear_form_for_tables():
    chunk = {
        "doc_short": "ND100", "dieu": "5", "khoan": "5",
        "dieu_title": "Điều 5", "chunk_type": "table",
        "linear_form": "Hàng 1: ô tô vượt đèn đỏ, mức phạt 4-6 triệu.",
        "text": "| Hành vi | Phạt |\n|---|---|\n| Vượt đèn đỏ | 4-6tr |",
    }
    out = format_chunks_for_prompt([chunk])
    assert "Hàng 1" in out
    assert "|" not in out, "Markdown table noise leaked into the prompt"


def test_format_chunks_empty():
    assert "Không có" in format_chunks_for_prompt([])


def test_build_messages_shape(chunks):
    msgs = build_messages("Vượt đèn đỏ ô tô phạt bao nhiêu?", chunks)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "Câu hỏi: Vượt đèn đỏ" in msgs[1]["content"]


def test_build_user_prompt_strips_question_whitespace(chunks):
    up = build_user_prompt("   spaced query   ", chunks)
    assert "Câu hỏi: spaced query" in up


# ===========================================================================
# 2. LLMClient with stub predictor
# ===========================================================================
def test_llm_client_predictor_stub_is_used():
    captured = {}

    def fake(messages):
        captured["messages"] = messages
        return "Phạt 4 triệu (Khoản 5, Điều 5 Nghị định 100/2019/NĐ-CP)."

    client = LLMClient(backend="stub", config={}, predictor=fake)
    resp = client.generate([{"role": "user", "content": "test"}])
    assert "Khoản 5" in resp.text
    assert captured["messages"][0]["content"] == "test"


# ===========================================================================
# 3. Full RAG pipeline with stub retriever + stub reranker + stub LLM
# ===========================================================================
class _StubRetriever:
    """Mimics HybridSearcher.search()."""
    def __init__(self, returns: List[Dict[str, Any]]):
        self._returns = returns

    def search(self, query: str, top_k: int = 10, filters=None):
        return list(self._returns)


def test_full_rag_chain_with_stubs(chunks):
    # Reranker that just returns input order with a synthetic score.
    reranker = CrossEncoderReranker(
        predictor=lambda pairs: [float(len(pairs) - i) for i in range(len(pairs))],
        segment_input=False,
    )

    # LLM stub that "answers" with the canonical first-chunk citation.
    def fake_llm(messages):
        return ("Người điều khiển xe ô tô vượt đèn đỏ bị phạt từ 4.000.000 đồng đến "
                "6.000.000 đồng (Điểm a, Khoản 5, Điều 5 Nghị định 100/2019/NĐ-CP).")
    llm = LLMClient(backend="stub", config={}, predictor=fake_llm)

    rag = TrafficLawRAG(
        retriever=_StubRetriever(chunks),
        reranker=reranker,
        llm=llm,
        validator=CitationValidator(require_at_least_one=True),
        hybrid_top_k=10,
        final_top_k=2,
    )
    result = rag.run("Vượt đèn đỏ xe ô tô phạt bao nhiêu?")
    assert result.answer.startswith("Người điều khiển xe ô tô")
    assert len(result.sources) == 2
    assert result.validation.is_valid
    # All four stages should report a timing.
    assert {"retrieve", "rerank", "generate", "validate"} <= set(result.timings)


def test_rag_chain_refuses_when_retrieval_empty(chunks):
    rag = TrafficLawRAG(
        retriever=_StubRetriever([]),
        reranker=CrossEncoderReranker(predictor=lambda pairs: [], segment_input=False),
        llm=LLMClient(backend="stub", config={}, predictor=lambda m: "should not be called"),
        validator=CitationValidator(require_at_least_one=True),
    )
    result = rag.run("Câu hỏi không có dữ liệu")
    assert result.refused_due_to_low_confidence
    assert "không tìm thấy quy định phù hợp" in result.answer.lower()
    assert result.validation.is_valid   # the refusal itself is valid


def test_rag_chain_refuses_when_below_confidence_gate(chunks):
    rag = TrafficLawRAG(
        retriever=_StubRetriever(chunks),
        # Reranker returns very low scores → below gate.
        reranker=CrossEncoderReranker(
            predictor=lambda pairs: [-5.0 for _ in pairs],
            segment_input=False,
        ),
        llm=LLMClient(backend="stub", config={}, predictor=lambda m: "should not be called"),
        validator=CitationValidator(require_at_least_one=True),
        min_cross_encoder_score=0.0,
    )
    result = rag.run("vague question")
    assert result.refused_due_to_low_confidence
    assert "không tìm thấy quy định phù hợp" in result.answer.lower()


def test_rag_chain_flags_hallucination_in_answer(chunks):
    """If the LLM stub cites a fabricated Điều, the validation report must fail."""
    reranker = CrossEncoderReranker(
        predictor=lambda pairs: [1.0] * len(pairs), segment_input=False,
    )
    bad_llm = LLMClient(
        backend="stub", config={},
        # Điều 99 doesn't exist in the chunks fixture.
        predictor=lambda m: "Phạt 4 triệu (Khoản 3, Điều 99 Nghị định 100/2019/NĐ-CP).",
    )
    rag = TrafficLawRAG(
        retriever=_StubRetriever(chunks),
        reranker=reranker,
        llm=bad_llm,
        validator=CitationValidator(require_at_least_one=True),
    )
    result = rag.run("any q")
    assert not result.validation.is_valid
    assert result.validation.hallucinated_citations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
