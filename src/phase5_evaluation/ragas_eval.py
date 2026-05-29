"""RAGAS evaluation — LLM-as-a-judge metrics for the generator.

Why RAGAS in addition to Recall@K / MRR?
    IR metrics only see RETRIEVAL — they cannot tell whether the LLM actually
    used the retrieved chunks or hallucinated around them. RAGAS closes that
    gap with a small family of LLM-as-a-judge metrics:

      * Faithfulness          — "Does every claim in the answer follow from
                                 the supplied contexts?"  Hallucination detector.
      * Answer Relevancy      — "Does the answer actually address the question?"
                                 Off-topic detector.
      * Context Precision     — "Of the retrieved contexts, how many were
                                 actually relevant?"  Reranker quality signal.
      * Context Recall        — "Of the ground-truth facts, how many appear in
                                 some retrieved context?"  Retriever quality signal.

    The four together form a 2×2 grid: { retrieval, generation } × { recall, precision }.
    That's a single slide the panel will understand instantly.

    Reference: Es, James, Espinosa-Anke, Schockaert — "RAGAS: Automated
    Evaluation of Retrieval Augmented Generation" (EACL 2024).

Judge model choice:
    RAGAS calls the judge LLM dozens of times per question (one call per
    claim for faithfulness, one per sentence for relevancy, …). Use a CHEAP
    model — `gpt-4o-mini` is the recommended default. Local Ollama / on-device
    HF are supported via langchain wrappers (see `_build_judge`).

Two operating modes:
    * `run_pipeline_live: true`  — for each golden question, run the full
      Phase 2 → 3 → 4 pipeline, capture (answer, contexts), then judge.
      Most faithful to deployment, but slow and costs LLM tokens.

    * `run_pipeline_live: false` — read precomputed `answer` and `contexts`
      from the JSONL. Useful when iterating on RAGAS settings without
      re-running the heavy pipeline.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from src.utils.io import load_yaml, read_jsonl


# ---------------------------------------------------------------------------
# A flat per-question result row — what we CSV-dump for the slides.
# ---------------------------------------------------------------------------
@dataclass
class RagasRow:
    question: str
    answer: str
    ground_truth: str
    contexts: List[str]
    # The four metrics; None if disabled in config.
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None


# ---------------------------------------------------------------------------
# Judge LLM + embeddings construction.
# ---------------------------------------------------------------------------
def _build_judge(judge_cfg: Dict[str, Any]):
    """Construct the (llm, embeddings) pair RAGAS expects.

    Returns (llm, embeddings) — both wrapped in the LangChain interfaces
    that the current version of RAGAS accepts.
    """
    backend = judge_cfg["backend"]

    if backend == "openai":
        sub = judge_cfg["openai"]
        api_key = os.environ.get(sub["api_key_env"])
        if not api_key:
            raise RuntimeError(
                f"RAGAS openai judge requires env var {sub['api_key_env']}."
            )
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        llm = ChatOpenAI(
            model=sub["model"],
            api_key=api_key,
            base_url=sub.get("base_url"),
            temperature=0.0,
        )
        embeddings = OpenAIEmbeddings(
            model=sub["embeddings_model"],
            api_key=api_key,
            base_url=sub.get("base_url"),
        )
        return llm, embeddings

    if backend == "ollama":
        sub = judge_cfg["ollama"]
        from langchain_community.chat_models import ChatOllama
        from langchain_huggingface import HuggingFaceEmbeddings
        llm = ChatOllama(
            model=sub["model"],
            base_url=sub.get("base_url", "http://localhost:11434"),
            temperature=0.0,
        )
        embeddings = HuggingFaceEmbeddings(model_name=sub["embeddings_model_hf"])
        return llm, embeddings

    if backend == "hf":
        sub = judge_cfg["hf"]
        from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        tok = AutoTokenizer.from_pretrained(sub["model_name"], trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            sub["model_name"], device_map="auto", trust_remote_code=True,
        )
        gen = pipeline("text-generation", model=mdl, tokenizer=tok,
                       max_new_tokens=512, do_sample=False)
        llm = HuggingFacePipeline(pipeline=gen)
        embeddings = HuggingFaceEmbeddings(model_name=sub["embeddings_model_hf"])
        return llm, embeddings

    raise ValueError(f"Unknown RAGAS judge backend: {backend!r}")


# ---------------------------------------------------------------------------
# The pipeline-live path: produce (answer, contexts) for each question.
# ---------------------------------------------------------------------------
def _materialise_with_pipeline(
    queries: List[Dict[str, Any]],
    rag,                  # TrafficLawRAG instance
) -> List[RagasRow]:
    out: List[RagasRow] = []
    for q in queries:
        result = rag.run(q["question"])
        contexts = []
        for c in result.sources:
            if c.get("chunk_type") == "table" and c.get("linear_form"):
                contexts.append(c["linear_form"])
            else:
                contexts.append(c.get("text", ""))
        out.append(RagasRow(
            question=q["question"],
            answer=result.answer,
            ground_truth=q.get("ground_truth", ""),
            contexts=contexts,
        ))
    return out


def _materialise_from_jsonl(queries: List[Dict[str, Any]]) -> List[RagasRow]:
    out: List[RagasRow] = []
    for q in queries:
        if "answer" not in q or "contexts" not in q:
            raise ValueError(
                "run_pipeline_live=false requires 'answer' and 'contexts' on every record."
            )
        out.append(RagasRow(
            question=q["question"],
            answer=q["answer"],
            ground_truth=q.get("ground_truth", ""),
            contexts=list(q["contexts"]),
        ))
    return out


# ---------------------------------------------------------------------------
# The judging step.
# ---------------------------------------------------------------------------
def _judge_with_ragas(rows: List[RagasRow], judge_cfg: Dict[str, Any],
                      metrics_cfg: Dict[str, bool]) -> List[RagasRow]:
    """Score each row with RAGAS. Mutates `rows` and returns it for convenience."""
    # Heavy import kept inside the function so this module is cheap to import.
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    metrics = []
    if metrics_cfg.get("faithfulness", True):
        metrics.append(faithfulness)
    if metrics_cfg.get("answer_relevancy", True):
        metrics.append(answer_relevancy)
    if metrics_cfg.get("context_precision", True):
        metrics.append(context_precision)
    if metrics_cfg.get("context_recall", True):
        metrics.append(context_recall)
    if not metrics:
        raise ValueError("All RAGAS metrics disabled — nothing to compute.")

    judge_llm, judge_emb = _build_judge(judge_cfg)

    ds = Dataset.from_list([
        {
            "question": r.question,
            "answer": r.answer,
            "contexts": r.contexts,
            "ground_truth": r.ground_truth,
        }
        for r in rows
    ])

    logger.info(f"Running RAGAS over {len(rows)} rows with {len(metrics)} metric(s) …")
    result = evaluate(ds, metrics=metrics, llm=judge_llm, embeddings=judge_emb)
    # `result` exposes both an aggregated view and per-row scores via .to_pandas()
    df = result.to_pandas()

    # Patch scores back onto our rows by position.
    for i, r in enumerate(rows):
        if "faithfulness" in df.columns:
            r.faithfulness = _safe_float(df.iloc[i]["faithfulness"])
        if "answer_relevancy" in df.columns:
            r.answer_relevancy = _safe_float(df.iloc[i]["answer_relevancy"])
        if "context_precision" in df.columns:
            r.context_precision = _safe_float(df.iloc[i]["context_precision"])
        if "context_recall" in df.columns:
            r.context_recall = _safe_float(df.iloc[i]["context_recall"])

    logger.info("RAGAS aggregate: " + " · ".join(
        f"{m}={result[m]:.3f}" for m in result.keys() if m in
        {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
    ))
    return rows


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CSV dump
# ---------------------------------------------------------------------------
def _write_csv(rows: List[RagasRow], path: Union[str, Path]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["question", "ground_truth", "answer", "n_contexts",
                  "faithfulness", "answer_relevancy",
                  "context_precision", "context_recall"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "question": r.question,
                "ground_truth": r.ground_truth,
                "answer": r.answer,
                "n_contexts": len(r.contexts),
                "faithfulness": r.faithfulness,
                "answer_relevancy": r.answer_relevancy,
                "context_precision": r.context_precision,
                "context_recall": r.context_recall,
            })
    return str(path)


# ---------------------------------------------------------------------------
# Public entry — drives the full evaluation from configs.
# ---------------------------------------------------------------------------
def run_ragas_evaluation(
    eval_yaml: Union[str, Path] = "configs/evaluation.yaml",
    settings_yaml: Union[str, Path] = "configs/settings.yaml",
    retrieval_yaml: Union[str, Path] = "configs/retrieval.yaml",
    generation_yaml: Union[str, Path] = "configs/generation.yaml",
) -> str:
    """Run the full RAGAS evaluation and write a per-question CSV. Returns the CSV path."""
    cfg = load_yaml(eval_yaml)
    queries = list(read_jsonl(cfg["golden_set"]["path"]))
    if not queries:
        raise ValueError(f"Golden set at {cfg['golden_set']['path']} is empty.")
    logger.info(f"Golden set: {len(queries)} questions")

    ragas_cfg = cfg["ragas"]
    if ragas_cfg.get("run_pipeline_live", True):
        # Live pipeline path
        from src.phase4_generation.rag_chain import TrafficLawRAG
        rag = TrafficLawRAG.from_configs(
            settings_yaml=settings_yaml,
            retrieval_yaml=retrieval_yaml,
            generation_yaml=generation_yaml,
        )
        rows = _materialise_with_pipeline(queries, rag)
    else:
        rows = _materialise_from_jsonl(queries)

    rows = _judge_with_ragas(rows, ragas_cfg["judge"], ragas_cfg["metrics"])
    out = _write_csv(rows, ragas_cfg["output_csv"])
    logger.info(f"RAGAS per-question scores → {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Run RAGAS evaluation over the golden set.")
    p.add_argument("--eval-yaml", default="configs/evaluation.yaml")
    p.add_argument("--settings-yaml", default="configs/settings.yaml")
    p.add_argument("--retrieval-yaml", default="configs/retrieval.yaml")
    p.add_argument("--generation-yaml", default="configs/generation.yaml")
    args = p.parse_args()
    run_ragas_evaluation(
        args.eval_yaml, args.settings_yaml, args.retrieval_yaml, args.generation_yaml,
    )


if __name__ == "__main__":
    main()
