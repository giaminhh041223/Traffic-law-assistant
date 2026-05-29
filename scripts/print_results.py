"""Print saved evaluation results with robust UTF-8 encoding.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    results_path = Path("data/evaluation/eval_results.json")
    if not results_path.exists():
        print(f"Error: results file not found at {results_path}")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Print beautiful summary report
    print("\n" + "="*80)
    print("                    VIETNAMESE TRAFFIC LAW RAG EVALUATION REPORT                    ")
    print("="*80)
    print(f"Total Test Cases Run:         {summary['total_test_cases']}")
    print(f"Average Retrieval Precision:  {summary['average_retrieval_precision']*100:.1f}%")
    print(f"Average Calculation Accuracy: {summary['average_calculation_accuracy']*100:.1f}%")
    print(f"Average Hallucination Rate:  {summary['average_hallucination_rate']*100:.1f}%")
    print(f"Validation Pass Rate:         {summary['validation_pass_rate']*100:.1f}%")
    print("="*80)
    
    # Output markdown table
    print("\n### Category Breakdown")
    print("| ID | Category | Query | Precision | Calc Acc | Hallucination | Valid |")
    print("|----|----------|-------|-----------|----------|---------------|-------|")
    for res in summary["results"]:
        query_snippet = res["query"][:35] + "..." if len(res["query"]) > 35 else res["query"]
        m = res.get("metrics", {"retrieval_precision": 0.0, "calculation_accuracy": 0.0, "hallucination_rate": 1.0})
        print(
            f"| {res['id']} | {res['category']} | {query_snippet} | "
            f"{m['retrieval_precision']*100:.0f}% | {m['calculation_accuracy']*100:.0f}% | "
            f"{m['hallucination_rate']*100:.0f}% | {res['is_valid_citation']} |"
        )
    print("="*80)

if __name__ == "__main__":
    main()
