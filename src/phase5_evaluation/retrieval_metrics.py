"""Retrieval metrics — Recall@K, Precision@K, MRR.

Why these three?
    They form the standard IR evaluation suite used across the academic literature
    (Manning, Raghavan & Schütze, "Introduction to Information Retrieval", §8).
    Each tells a different story:

      * Recall@K     — Did the retriever surface at least one gold answer
                       within its top-K results? Sensitive to coverage.
      * Precision@K  — How dense are the gold answers among the top-K? Sensitive
                       to noise. (For single-answer queries Precision@K is just
                       Recall@K / K, but legal QA often has multiple valid
                       chunks per question, so precision becomes informative.)
      * MRR          — 1 / (rank of the first gold answer); averages well across
                       queries with very different difficulty. The metric most
                       people quote when comparing retrieval systems.

    For Phase 5's headline chart we compute all three for BOTH stages:

      * Stage A — Hybrid (BM25 ⊕ dense ⊕ RRF) only.
      * Stage B — Hybrid → Cross-Encoder rerank.

    The lift from A → B is exactly the "value of Phase 3" claim we'll defend.

Public API:
    * recall_at_k, precision_at_k, mean_reciprocal_rank — pure functions.
    * evaluate_retrieval — runs the full Stage A vs Stage B comparison and
      returns a RetrievalReport that can be CSV-dumped for slides.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from loguru import logger


# ===========================================================================
# Pure metric functions — order-sensitive, no side effects.
# ===========================================================================
def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of RELEVANT items captured within the top-K results.

    Note the denominator is `len(relevant_ids)`, NOT `k` — that's recall, not
    precision. A query with 3 valid gold chunks where the retriever surfaces 2
    of them in its top-5 gets recall@5 = 2/3, not 2/5.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of TOP-K results that are relevant."""
    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / min(k, len(top_k))


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """1 / rank of the first relevant item; 0 if none appear."""
    relevant = set(relevant_ids)
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    rankings: Iterable[Tuple[Sequence[str], Iterable[str]]],
) -> float:
    """MRR over many queries. `rankings` = iterable of (retrieved, relevant) tuples."""
    rrs = [reciprocal_rank(r, g) for r, g in rankings]
    return sum(rrs) / len(rrs) if rrs else 0.0


# ===========================================================================
# Aggregated report
# ===========================================================================
@dataclass
class StageReport:
    """Metrics for ONE stage (Hybrid-only or After-rerank), averaged over the queries."""
    stage: str                                  # "hybrid" | "rerank"
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    n_queries: int = 0


@dataclass
class RetrievalReport:
    """End-to-end Phase 2 vs Phase 3 comparison."""
    hybrid: StageReport
    rerank: StageReport
    # Per-query breakdown for CSV dump / slide tables.
    per_query: List[Dict[str, Any]] = field(default_factory=list)

    def lift(self, k: int) -> Dict[str, float]:
        """Absolute and relative lift from hybrid → rerank at recall@k and MRR."""
        h_r = self.hybrid.recall_at_k.get(k, 0.0)
        b_r = self.rerank.recall_at_k.get(k, 0.0)
        h_m = self.hybrid.mrr
        b_m = self.rerank.mrr
        return {
            f"recall@{k}_hybrid": h_r,
            f"recall@{k}_rerank": b_r,
            f"recall@{k}_abs_lift": b_r - h_r,
            f"recall@{k}_rel_lift_pct": (100.0 * (b_r - h_r) / h_r) if h_r > 0 else float("inf"),
            "mrr_hybrid": h_m,
            "mrr_rerank": b_m,
            "mrr_abs_lift": b_m - h_m,
            "mrr_rel_lift_pct": (100.0 * (b_m - h_m) / h_m) if h_m > 0 else float("inf"),
        }

    # ----------------------------- CSV ---------------------------------
    def write_csv(self, path: Union[str, Path]) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.per_query:
            raise ValueError("No per-query records to write.")
        fieldnames = list(self.per_query[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.per_query)
        return str(path)


# ===========================================================================
# The driver — runs both stages over a labelled set and reports.
# ===========================================================================
def evaluate_retrieval(
    queries: List[Dict[str, Any]],
    retriever,                       # HybridSearcher (or anything with .search(query, top_k, filters))
    reranker=None,                   # CrossEncoderReranker (or None to skip stage B)
    k_values: Sequence[int] = (1, 3, 5, 10),
    hybrid_top_k_for_baseline: int = 10,
    hybrid_top_k_for_rerank: int = 10,
    rerank_final_k: int = 3,
) -> RetrievalReport:
    """Evaluate Stage A (hybrid only) vs Stage B (hybrid → rerank).

    Args:
        queries: list of dicts with at least:
            * `question`: str
            * `relevant_chunk_ids`: list[str]   (one or more gold chunks)
        retriever: a HybridSearcher (must expose `.search(q, top_k, filters)`).
        reranker:  a CrossEncoderReranker; if None, the rerank arm is skipped.
        k_values:  K cuts to report Recall@K / Precision@K for.

    Returns:
        RetrievalReport — aggregated + per-query.
    """
    hybrid_report = StageReport(stage="hybrid")
    rerank_report = StageReport(stage="rerank")

    sum_recall_h: Dict[int, float] = {k: 0.0 for k in k_values}
    sum_recall_r: Dict[int, float] = {k: 0.0 for k in k_values}
    sum_prec_h: Dict[int, float] = {k: 0.0 for k in k_values}
    sum_prec_r: Dict[int, float] = {k: 0.0 for k in k_values}
    sum_rr_h, sum_rr_r = 0.0, 0.0
    n_h, n_r = 0, 0

    per_query: List[Dict[str, Any]] = []

    for q in queries:
        question = q["question"]
        gold: Set[str] = set(q.get("relevant_chunk_ids", []))
        if not gold:
            logger.warning(f"Skipping query with no gold ids: {question!r}")
            continue

        # ----- Stage A: hybrid only -----
        hybrid_pool = retriever.search(
            question, top_k=hybrid_top_k_for_baseline, filters=q.get("filters")
        )
        hybrid_ids = [c["chunk_id"] for c in hybrid_pool]

        rec_h = {k: recall_at_k(hybrid_ids, gold, k) for k in k_values}
        prec_h = {k: precision_at_k(hybrid_ids, gold, k) for k in k_values}
        rr_h = reciprocal_rank(hybrid_ids, gold)
        for k in k_values:
            sum_recall_h[k] += rec_h[k]
            sum_prec_h[k] += prec_h[k]
        sum_rr_h += rr_h
        n_h += 1

        # ----- Stage B: hybrid → cross-encoder rerank -----
        rec_r: Dict[int, float] = {}
        prec_r: Dict[int, float] = {}
        rr_r: float = 0.0
        rerank_ids: List[str] = []
        if reranker is not None:
            # Re-retrieve a (possibly larger) pool, then rerank.
            pool = retriever.search(
                question, top_k=hybrid_top_k_for_rerank, filters=q.get("filters")
            )
            reranked = reranker.rerank(question, pool, top_k=rerank_final_k)
            rerank_ids = [c["chunk_id"] for c in reranked]

            rec_r = {k: recall_at_k(rerank_ids, gold, k) for k in k_values}
            prec_r = {k: precision_at_k(rerank_ids, gold, k) for k in k_values}
            rr_r = reciprocal_rank(rerank_ids, gold)
            for k in k_values:
                sum_recall_r[k] += rec_r[k]
                sum_prec_r[k] += prec_r[k]
            sum_rr_r += rr_r
            n_r += 1

        # ----- per-query record (one row per query, both stages flattened) -----
        record: Dict[str, Any] = {
            "question": question,
            "gold_ids": "|".join(sorted(gold)),
            "hybrid_topk_ids": "|".join(hybrid_ids[:max(k_values)]),
            "hybrid_rr": rr_h,
        }
        for k in k_values:
            record[f"hybrid_recall@{k}"] = rec_h[k]
            record[f"hybrid_precision@{k}"] = prec_h[k]
        if reranker is not None:
            record["rerank_topk_ids"] = "|".join(rerank_ids)
            record["rerank_rr"] = rr_r
            for k in k_values:
                record[f"rerank_recall@{k}"] = rec_r.get(k, 0.0)
                record[f"rerank_precision@{k}"] = prec_r.get(k, 0.0)
        per_query.append(record)

    # ----- average -----
    if n_h:
        for k in k_values:
            hybrid_report.recall_at_k[k] = sum_recall_h[k] / n_h
            hybrid_report.precision_at_k[k] = sum_prec_h[k] / n_h
        hybrid_report.mrr = sum_rr_h / n_h
        hybrid_report.n_queries = n_h
    if n_r:
        for k in k_values:
            rerank_report.recall_at_k[k] = sum_recall_r[k] / n_r
            rerank_report.precision_at_k[k] = sum_prec_r[k] / n_r
        rerank_report.mrr = sum_rr_r / n_r
        rerank_report.n_queries = n_r

    return RetrievalReport(hybrid=hybrid_report, rerank=rerank_report, per_query=per_query)


# ===========================================================================
# Pretty printer — turns a report into a Markdown table for slides.
# ===========================================================================
def report_to_markdown(report: RetrievalReport, k_values: Sequence[int] = (1, 3, 5, 10)) -> str:
    head = "| Metric | Hybrid (Phase 2) | + Cross-Encoder (Phase 3) | Δ |\n|---|---|---|---|\n"
    rows: List[str] = []
    for k in k_values:
        h = report.hybrid.recall_at_k.get(k, 0.0)
        r = report.rerank.recall_at_k.get(k, 0.0)
        rows.append(f"| Recall@{k} | {h:.3f} | {r:.3f} | {r - h:+.3f} |")
        h = report.hybrid.precision_at_k.get(k, 0.0)
        r = report.rerank.precision_at_k.get(k, 0.0)
        rows.append(f"| Precision@{k} | {h:.3f} | {r:.3f} | {r - h:+.3f} |")
    rows.append(f"| MRR | {report.hybrid.mrr:.3f} | {report.rerank.mrr:.3f} | "
                f"{report.rerank.mrr - report.hybrid.mrr:+.3f} |")
    rows.append(f"| N queries | {report.hybrid.n_queries} | {report.rerank.n_queries} | — |")
    return head + "\n".join(rows)


# ===========================================================================
# CLI entry — typical use: `python -m src.phase5_evaluation.retrieval_metrics`
# ===========================================================================
def main() -> None:
    import argparse
    from src.phase2_hybrid_search.hybrid_search import HybridSearcher
    from src.phase3_reranking.cross_encoder_reranker import CrossEncoderReranker
    from src.utils.io import load_yaml, read_jsonl

    parser = argparse.ArgumentParser(description="Run retrieval metrics (Phase 2 vs Phase 3).")
    parser.add_argument("--eval-yaml", default="configs/evaluation.yaml")
    parser.add_argument("--settings-yaml", default="configs/settings.yaml")
    parser.add_argument("--retrieval-yaml", default="configs/retrieval.yaml")
    parser.add_argument("--skip-rerank", action="store_true",
                        help="Evaluate hybrid only (skip Phase 3 — fast baseline).")
    args = parser.parse_args()

    cfg = load_yaml(args.eval_yaml)
    queries = list(read_jsonl(cfg["golden_set"]["path"]))
    logger.info(f"Loaded {len(queries)} golden queries.")

    retriever = HybridSearcher.from_configs(
        settings_yaml=args.settings_yaml, retrieval_yaml=args.retrieval_yaml
    )
    reranker = None if args.skip_rerank else CrossEncoderReranker.from_config(args.retrieval_yaml)

    rm = cfg["retrieval_metrics"]
    report = evaluate_retrieval(
        queries=queries, retriever=retriever, reranker=reranker,
        k_values=tuple(rm["k_values"]),
        hybrid_top_k_for_baseline=rm["hybrid_top_k_for_baseline"],
        hybrid_top_k_for_rerank=rm["hybrid_top_k_for_rerank"],
        rerank_final_k=rm["rerank_final_k"],
    )
    out = report.write_csv(rm["output_csv"])
    logger.info(f"Per-query CSV → {out}")
    print(report_to_markdown(report, tuple(rm["k_values"])))


if __name__ == "__main__":
    main()
