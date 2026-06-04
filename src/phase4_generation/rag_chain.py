"""TrafficLawRAG — the end-to-end Hybrid AI pipeline.

Pipeline (pure-Python — no hard LangChain dependency):

    query
      │
      ▼
    TrafficAIRouter.evaluate_complexity(query)      ← Stage 0 (advisory routing)
      │
      ▼
    TrafficQueryRewriter.rewrite(query)             ← Stage 1 (slang → formal)
      │
      ▼
    TrafficQueryDeconstructor → parallel search      ← Stage 2 (compound split + RRF)
      │
      ▼
    CrossEncoderReranker.rerank(query, …, top_k=3)  ← Stage 3 (cross-encoder)
      │
      ▼
    prompt_templates.build_messages(query, top_3)   ← Stage 4 (strict VN system prompt)
      │
      ▼
    LLMClient.generate(messages)                    ← Stage 4 (LLM + CoT parsing)
      │
      ▼
    CitationValidator.validate(answer, top_3)       ← Stage 5 (post-gen safety net)
      │
      ▼
    RAGResult { answer, thought, sources, validation, timings }

Design notes:
    * No LangChain. The "chain" is just function composition — easier to debug,
      easier to test, no hidden retries / template magic.
    * Every stage is injected, not constructed inline. That keeps the pipeline
      unit-testable with stub retrievers + stub LLMs.
    * The pipeline NEVER swallows validation failures — it returns them on the
      RAGResult so the UI can decorate the answer with a warning banner.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from src.phase2_hybrid_search.hybrid_search import HybridSearcher
from src.phase2_hybrid_search.query_rewriter import TrafficQueryRewriter
from src.phase2_hybrid_search.query_deconstructor import TrafficQueryDeconstructor
from src.phase2_hybrid_search.query_reformulator import TrafficQueryReformulator
from src.phase3_reranking.cross_encoder_reranker import CrossEncoderReranker
from src.phase4_generation.ai_router import TrafficAIRouter
from src.phase4_generation.citation_validator import (
    CitationValidator,
    ValidationReport,
)
from src.phase4_generation.llm_loader import LLMClient, LLMResponse, load_llm
from src.phase4_generation.prompt_templates import build_messages
from src.utils.io import load_yaml


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------
@dataclass
class RAGResult:
    query: str
    answer: str
    # The intermediate CoT thought extracted from XML tags.
    thought: str = ""
    # The Top-K reranked chunks that grounded the answer.
    sources: List[Dict[str, Any]] = field(default_factory=list)
    # Validation report from CitationValidator.
    validation: Optional[ValidationReport] = None
    # Stage-by-stage wall-clock (seconds).
    timings: Dict[str, float] = field(default_factory=dict)
    # Token usage when available (HF returns it; Ollama / OpenAI sometimes too).
    usage: Dict[str, Any] = field(default_factory=dict)
    # Raw LLM message list — useful for debugging prompt construction.
    messages: List[Dict[str, str]] = field(default_factory=list)
    # When the retrieval gate fires, we may answer "no info found" WITHOUT calling the LLM.
    refused_due_to_low_confidence: bool = False


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
class TrafficLawRAG:
    def __init__(
        self,
        retriever: HybridSearcher,
        reranker: CrossEncoderReranker,
        llm: LLMClient,
        validator: CitationValidator,
        rewriter: Optional[TrafficQueryRewriter] = None,
        deconstructor: Optional[TrafficQueryDeconstructor] = None,
        router: Optional[TrafficAIRouter] = None,
        hybrid_top_k: int = 10,
        final_top_k: int = 3,
        min_cross_encoder_score: Optional[float] = None,
        refusal_phrase: str = (
            "Tôi không tìm thấy quy định phù hợp trong văn bản pháp luật được cung cấp "
            "để trả lời câu hỏi này."
        ),
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.validator = validator
        # --- Hybrid AI upgrade modules ---
        self.rewriter = rewriter or TrafficQueryRewriter()
        self.deconstructor = deconstructor or TrafficQueryDeconstructor()
        self.router = router or TrafficAIRouter(deconstructor=self.deconstructor)
        self.reformulator = TrafficQueryReformulator(self.llm)
        self.hybrid_top_k = hybrid_top_k
        self.final_top_k = final_top_k
        self.min_cross_encoder_score = min_cross_encoder_score
        self.refusal_phrase = refusal_phrase

    # ----------------------------- factory -----------------------------

    @classmethod
    def from_configs(
        cls,
        settings_yaml: Union[str, Path] = "configs/settings.yaml",
        retrieval_yaml: Union[str, Path] = "configs/retrieval.yaml",
        generation_yaml: Union[str, Path] = "configs/generation.yaml",
        chunks_jsonl: Optional[Union[str, Path]] = None,
        bm25_index_path: Optional[Union[str, Path]] = None,
        llm_backend_override: Optional[str] = None,
        llm_predictor: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ) -> "TrafficLawRAG":
        retriever = HybridSearcher.from_configs(
            settings_yaml=settings_yaml,
            retrieval_yaml=retrieval_yaml,
            chunks_jsonl=chunks_jsonl,
            bm25_index_path=bm25_index_path,
        )
        reranker = CrossEncoderReranker.from_config(retrieval_yaml)
        llm = load_llm(
            generation_yaml=generation_yaml,
            backend_override=llm_backend_override,
            predictor=llm_predictor,
        )
        validator = CitationValidator.from_config(generation_yaml)

        gen_cfg = load_yaml(generation_yaml).get("rag", {})
        # Build the hybrid AI upgrade modules
        rewriter = TrafficQueryRewriter()
        deconstructor = TrafficQueryDeconstructor()
        router = TrafficAIRouter(deconstructor=deconstructor)

        return cls(
            retriever=retriever,
            reranker=reranker,
            llm=llm,
            validator=validator,
            rewriter=rewriter,
            deconstructor=deconstructor,
            router=router,
            hybrid_top_k=gen_cfg.get("hybrid_top_k", 10),
            final_top_k=gen_cfg.get("final_top_k", 3),
            min_cross_encoder_score=gen_cfg.get("min_cross_encoder_score"),
        )

    # ----------------------------- public -----------------------------

    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RAGResult:
        """One-shot query → answer with full provenance.

        Upgraded pipeline flow:
            0. AI Router evaluates complexity (local vs cloud logging).
            1. QueryRewriter normalises slang → formal legal terms.
            2. QueryDeconstructor splits compound queries → parallel retrieval.
            3. Cross-encoder rerank on merged candidates.
            4. Prompt + LLM generation with CoT thought parsing.
            5. Citation validation.
        """
        result = RAGResult(query=query, answer="")

        # ---- Stage 1.5: conversational query reformulation ----
        t_ref = time.perf_counter()
        if chat_history:
            reformulated_query = self.reformulator.reformulate(query, chat_history)
        else:
            reformulated_query = query
        result.timings["reformulate"] = time.perf_counter() - t_ref

        # ---- Stage 0: complexity routing (advisory) ----
        route_target = self.router.evaluate_complexity(reformulated_query)
        logger.info(f"[rag] AI Router advisory target: {route_target}")
        # NOTE: currently always uses local LLM; cloud routing will be
        # enabled once OPENAI_API_KEY support is configured.

        # ---- Stage 1 & 2: Rewriting (slang → formal) THEN Deconstruction ----
        # CRITICAL: Rewriter runs FIRST on the full query so that slang terms
        # (xe cọp, thông chốt, nẹt pô) are resolved to formal legal terms
        # BEFORE the Deconstructor makes vehicle-type branching decisions.
        t_rw = time.perf_counter()
        pre_rewritten = self.rewriter.rewrite(reformulated_query)
        sub_queries = self.deconstructor.deconstruct(pre_rewritten)
        rewritten_query = " và ".join(sub_queries) if len(sub_queries) > 1 else sub_queries[0]
        result.timings["rewrite"] = time.perf_counter() - t_rw

        # ---- Stage 3: Localized Reranking & Retrieval ----
        t0 = time.perf_counter()

        if len(sub_queries) > 1:
            # SOTA Parallel localized retrieval & reranking for compound queries
            logger.info(
                f"[rag] compound query detected with {len(sub_queries)} intents. "
                f"Running localized atomic retrievals and cross-encoder rerankings..."
            )
            intent_tops = []
            for sq in sub_queries:
                try:
                    sq_candidates = self.retriever.search(sq, top_k=self.hybrid_top_k, filters=filters)
                    if sq_candidates:
                        sq_top = self.reranker.rerank(sq, sq_candidates, top_k=2)
                        # Fall back to RRF order if CE confidence is extremely low (max score < 0.1)
                        # This prevents the Cross-Encoder from introducing noise when it fails to match colloquial terms
                        if sq_top and sq_top[0].get("cross_encoder_score", -1e9) < 0.1:
                            logger.info(
                                f"[rag] CE score too low ({sq_top[0].get('cross_encoder_score'):.4f} < 0.1) for '{sq}'. "
                                f"Falling back to RRF ordering."
                            )
                            fallback_top = []
                            for c in sq_candidates[:2]:
                                new_c = dict(c)
                                new_c["cross_encoder_score"] = float(new_c.get("rrf_score", 1.0))
                                fallback_top.append(new_c)
                            sq_top = fallback_top
                        intent_tops.append(sq_top)
                    else:
                        intent_tops.append([])
                except Exception as exc:
                    logger.error(f"[rag] atomic query '{sq}' retrieval/rerank failed: {exc}")
                    intent_tops.append([])
            
            t_elapsed = time.perf_counter() - t0
            result.timings["retrieve"] = t_elapsed * 0.2
            result.timings["rerank"] = t_elapsed * 0.8

            # Context-budget cap: small LLMs (1.5B–7B) degrade with too many
            # context chunks. We keep at most final_top_k + 2 chunks but
            # guarantee every sub-intent has at least its top-1 representative.
            merged_top_chunks = []
            seen_ids = set()
            
            # Phase 1: Enforce the guarantee - add all top-1 representatives first
            for tops in intent_tops:
                if len(tops) > 0:
                    c = tops[0]
                    # Keep the top representative of each intent to guarantee complete coverage
                    if c.get("cross_encoder_score", -1e9) > -999.0:
                        cid = c.get("chunk_id")
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            merged_top_chunks.append(c)
                        
            # Phase 2: Collect other candidates (including tops[0] if it was skipped) and sort them by CE score desc
            others = []
            for tops in intent_tops:
                for c in tops:
                    cid = c.get("chunk_id")
                    if cid not in seen_ids:
                        others.append(c)
            
            others.sort(key=lambda c: c.get("cross_encoder_score", -1e9), reverse=True)
            
            # Fill remaining capacity up to max_chunks (5)
            max_chunks = self.final_top_k + 2          # default: 5
            for c in others:
                if len(merged_top_chunks) >= max_chunks:
                    break
                cid = c.get("chunk_id")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    merged_top_chunks.append(c)
                    
            logger.info(
                f"[rag] merged parallel retrieval results. "
                f"Total unique chunks: {len(merged_top_chunks)} (guaranteed top-1s: {len([t for t in intent_tops if t])})"
            )
            top_chunks = merged_top_chunks
        else:
            # Simple single-intent query → standard retrieval and rerank.
            candidates = self.retriever.search(
                rewritten_query, top_k=self.hybrid_top_k, filters=filters
            )
            result.timings["retrieve"] = time.perf_counter() - t0
            
            if not candidates:
                result.answer = self.refusal_phrase
                result.refused_due_to_low_confidence = True
                result.validation = self.validator.validate(result.answer, [], query=reformulated_query)
                return result
                
            t1 = time.perf_counter()
            top_chunks = self.reranker.rerank(rewritten_query, candidates, top_k=self.final_top_k)
            result.timings["rerank"] = time.perf_counter() - t1
            
        result.sources = top_chunks

        # Confidence gate: if even the best reranked candidate is below threshold,
        # refuse rather than gamble on a hallucination. (None disables the gate.)
        if (
            self.min_cross_encoder_score is not None
            and top_chunks
            and top_chunks[0].get("cross_encoder_score", -1e9) < self.min_cross_encoder_score
        ):
            logger.info(
                f"[rag] refusing: top cross-encoder score "
                f"{top_chunks[0]['cross_encoder_score']:.3f} < threshold "
                f"{self.min_cross_encoder_score}"
            )
            result.answer = self.refusal_phrase
            result.refused_due_to_low_confidence = True
            result.validation = self.validator.validate(result.answer, top_chunks, query=reformulated_query)
            return result

        # ---- Stage 4: prompt → LLM ----
        # Use the fully rewritten query (standardized terms like 'xe máy' -> 'xe mô tô')
        # so the 1.5B model doesn't fail at basic lexical matching against the context.
        messages = build_messages(rewritten_query, top_chunks, chat_history=chat_history, intents=sub_queries)
        result.messages = messages
        t2 = time.perf_counter()
        
        # Build context string for the router
        context_str = "\n".join([f"[{i+1}]. {chunk['text']}" for i, chunk in enumerate(top_chunks)])
        
        llm_resp: LLMResponse = self.router.route_generation(
            query=reformulated_query,
            context=context_str,
            route_status=route_target,
            local_generator_fn=self.llm.generate,
            messages=messages
        )
        
        result.timings["generate"] = time.perf_counter() - t2

        # Parse XML thought tags if present
        raw_text = llm_resp.text or ""
        thought_match = re.search(r"<thought>(.*?)</thought>", raw_text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            result.thought = thought_match.group(1).strip()
            # Clean the answer by taking everything after </thought>
            # This robustly handles any chatty/conversational prefixes prepended before the thought tag
            parts = re.split(r"</thought>", raw_text, flags=re.IGNORECASE)
            clean_answer = parts[-1].strip()
            result.answer = clean_answer
        else:
            result.answer = raw_text

        # Strip standard "Trả lời:" prefix if prepended by few-shot copying
        if result.answer.startswith("Trả lời:"):
            result.answer = result.answer[len("Trả lời:"):].strip()

        result.usage = llm_resp.usage

        # ---- Stage 5: citation validation ----
        t3 = time.perf_counter()
        result.validation = self.validator.validate(result.answer, top_chunks, query=reformulated_query)
        result.timings["validate"] = time.perf_counter() - t3

        if not result.validation.is_valid:
            logger.warning(
                f"[rag] answer failed validation. failures={result.validation.failures}. "
                f"Falling back to refusal phrase."
            )
            result.answer = self.refusal_phrase
            result.refused_due_to_low_confidence = True
            result.validation = self.validator.validate(result.answer, top_chunks, query=reformulated_query)

        return result

