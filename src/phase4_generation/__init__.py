"""Phase 4 — LLM Generation with citation enforcement.

Public API surface (re-exports kept lazy via __all__ to avoid eager heavy imports):

    * LLMClient                 — unified .generate() over HF / Ollama / OpenAI-compatible backends
    * load_llm                  — factory that returns an LLMClient from configs/generation.yaml
    * build_system_prompt       — strict Vietnamese system prompt builder
    * build_user_prompt         — context-injection user prompt builder
    * format_chunks_for_prompt  — turns Top-K reranked chunks into citation-tagged context
    * CitationValidator         — regex-based post-gen validator
    * ValidationReport          — its result dataclass
    * TrafficLawRAG             — end-to-end query → answer + validation
    * RAGResult                 — its result dataclass
"""

__all__ = [
    "LLMClient",
    "load_llm",
    "build_system_prompt",
    "build_user_prompt",
    "format_chunks_for_prompt",
    "CitationValidator",
    "ValidationReport",
    "TrafficLawRAG",
    "RAGResult",
]
