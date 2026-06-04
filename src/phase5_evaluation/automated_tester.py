"""TrafficLawAutomatedTester — Automated evaluation module for Vietnamese Traffic Law RAG.

This module loads the test matrix JSON, executes each test case against the TrafficLawRAG
pipeline, and evaluates retrieval precision, calculation accuracy, and hallucination rate.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from loguru import logger

from src.phase4_generation.rag_chain import TrafficLawRAG, RAGResult


class TrafficLawAutomatedTester:
    def __init__(
        self,
        test_matrix_path: str = "data/evaluation/test_matrix.json",
        results_output_path: str = "data/evaluation/eval_results.json",
        settings_yaml: str = "configs/settings.yaml",
        retrieval_yaml: str = "configs/retrieval.yaml",
        generation_yaml: str = "configs/generation.yaml",
        llm_backend_override: Optional[str] = None,
    ):
        self.test_matrix_path = Path(test_matrix_path)
        self.results_output_path = Path(results_output_path)
        self.settings_yaml = settings_yaml
        self.retrieval_yaml = retrieval_yaml
        self.generation_yaml = generation_yaml
        self.llm_backend_override = llm_backend_override
        
        # Load the RAG pipeline
        logger.info("Initializing TrafficLawRAG pipeline for automated evaluation...")
        self.rag = TrafficLawRAG.from_configs(
            settings_yaml=settings_yaml,
            retrieval_yaml=retrieval_yaml,
            generation_yaml=generation_yaml,
            llm_backend_override=llm_backend_override,
        )
        logger.info("RAG pipeline successfully loaded!")

    def load_test_matrix(self) -> List[Dict[str, Any]]:
        """Load the 15-case test matrix JSON file."""
        if not self.test_matrix_path.exists():
            raise FileNotFoundError(f"Test matrix not found at: {self.test_matrix_path}")
        with open(self.test_matrix_path, "r", encoding="utf-8") as f:
            matrix = json.load(f)
        logger.info(f"Successfully loaded test matrix with {len(matrix)} test cases.")
        return matrix

    def parse_expected_fines(self, expected_truth: str) -> List[str]:
        """Extract monetary amounts (e.g. '10.600.000', '2.600.000') from ground truth for mathematical validation."""
        pattern = r"\b\d{1,3}(?:\.\d{3})+\b"
        return re.findall(pattern, expected_truth)

    def parse_expected_citations(self, expected_behavior: str, expected_truth: str) -> Set[str]:
        """Extract article / clause / QCVN / Luật references from ground truth and expected behavior."""
        citations = set()
        
        # Match 'Điều X'
        for m in re.finditer(r"(?i)Điều\s+(\d+)", expected_behavior + " " + expected_truth):
            citations.add(f"dieu_{m.group(1)}")
            
        # Match 'Khoản X'
        for m in re.finditer(r"(?i)Khoản\s+(\d+)", expected_behavior + " " + expected_truth):
            citations.add(f"khoan_{m.group(1)}")
            
        # Match 'Điểm X'
        for m in re.finditer(r"(?i)Điểm\s+([a-zđ])", expected_behavior + " " + expected_truth):
            citations.add(f"diem_{m.group(1).lower()}")
            
        # Match 'QCVN 41'
        if "QCVN 41" in (expected_behavior + " " + expected_truth):
            citations.add("QCVN_41")
            
        # Match 'Luật TTATGT'
        if "TTATGT" in (expected_behavior + " " + expected_truth) or "trật tự" in (expected_behavior + " " + expected_truth).lower():
            citations.add("LUAT_TTATGT")
            
        return citations

    def evaluate_retrieval_precision(self, retrieved_chunks: List[Dict[str, Any]], expected_cits: Set[str]) -> float:
        """Evaluate how precisely the retrieved chunks align with the expected articles and clauses.
        
        Formula: (Number of expected citations covered by retrieved chunks) / (Total expected citations)
        """
        if not expected_cits:
            return 1.0  # Out of domain / simple query
            
        covered = 0
        for expected in expected_cits:
            found = False
            for chunk in retrieved_chunks:
                text = (chunk.get("text") or "").lower()
                chunk_id = (chunk.get("chunk_id") or "").lower()
                
                # Check match based on tag
                if expected == "QCVN_41" and ("qcvn 41" in text or "qcvn" in chunk_id):
                    found = True
                elif expected == "LUAT_TTATGT" and ("ttatgt" in chunk_id or "luật trật tự" in text):
                    found = True
                elif expected.startswith("dieu_"):
                    dieu_num = expected.split("_")[1]
                    # Check if 'dieu5' or 'dieu_5' is in chunk_id, or if 'Điều X' matches chunk metadata
                    if f"dieu_{dieu_num}" in chunk_id or f"dieu{dieu_num}" in chunk_id or chunk.get("dieu") == dieu_num:
                        found = True
                elif expected.startswith("khoan_"):
                    khoan_num = expected.split("_")[1]
                    if f"khoan_{khoan_num}" in chunk_id or f"khoan{khoan_num}" in chunk_id or chunk.get("khoan") == khoan_num:
                        found = True
                elif expected.startswith("diem_"):
                    diem_letter = expected.split("_")[1]
                    if f"diem_{diem_letter}" in chunk_id or f"diem{diem_letter}" in chunk_id or chunk.get("diem") == diem_letter:
                        found = True
                        
                if found:
                    break
            if found:
                covered += 1
                
        return covered / len(expected_cits)

    def _normalize_vn_numbers(self, text: str) -> str:
        """Normalize Vietnamese text-based numbers to full digits.
        
        Handles patterns like:
        - '6 triệu' → '6000000'
        - '400 nghìn' → '400000'  
        - '18,5 triệu' → '18500000'
        - '2.5 triệu' → '2500000'
        """
        import re as _re
        result = text
        
        # Pattern: number + triệu (million)
        result = _re.sub(
            r'(\d+)[.,](\d+)\s*triệu',
            lambda m: str(int(m.group(1)) * 1000000 + int(m.group(2)) * (100000 if len(m.group(2)) == 1 else 10000 if len(m.group(2)) == 2 else 1000)),
            result
        )
        result = _re.sub(
            r'(\d+)\s*triệu',
            lambda m: str(int(m.group(1)) * 1000000),
            result
        )
        
        # Pattern: number + nghìn/ngàn (thousand)
        result = _re.sub(
            r'(\d+)\s*(?:nghìn|ngàn)',
            lambda m: str(int(m.group(1)) * 1000),
            result
        )
        
        return result

    def evaluate_calculation_accuracy(self, actual_answer: str, expected_fines: List[str]) -> float:
        """Verify if fine calculations in the answer match the expected values.
        
        Formula: 1.0 if all expected fine figures are successfully stated in the answer as whole numbers, 0.0 otherwise.
        Now handles Vietnamese text-based numbers (e.g. '6 triệu') via normalization.
        """
        if not expected_fines:
            return 1.0  # No calculations expected
        
        # First normalize Vietnamese text numbers, then strip dots/commas
        actual_normalized = self._normalize_vn_numbers(actual_answer)
        actual_clean = actual_normalized.replace(".", "").replace(",", "")
        
        # Check if all expected numbers appear in actual clean answer as whole numbers
        for expected in expected_fines:
            clean_expected = expected.replace(".", "").replace(",", "")
            # Use negative lookaround to prevent matching part of a larger number (e.g. 600000 matching inside 2600000)
            pattern = r"(?<!\d)" + re.escape(clean_expected) + r"(?!\d)"
            if not re.search(pattern, actual_clean):
                logger.warning(f"Calculation mismatched: expected {expected} VND as a whole number in answer.")
                return 0.0
                
        return 1.0

    def evaluate_hallucination_rate(self, result: RAGResult) -> float:
        """Evaluate if the answer has fabricated citations.
        
        Formula: Number of hallucinated citations / Total citations found.
        """
        val = result.validation
        if not val:
            return 0.0
            
        total_citations = len(val.citations_found)
        if total_citations == 0:
            # If at least one citation was required but none found, it's invalid, but not strictly a citation hallucination rate of 1.0 unless it fabricated something.
            return 0.0
            
        hallucinated = len(val.hallucinated_citations)
        return hallucinated / total_citations

    def run_eval(self) -> Dict[str, Any]:
        """Execute the automated test suite over the test matrix."""
        test_cases = self.load_test_matrix()
        
        total_cases = len(test_cases)
        sum_precision = 0.0
        sum_calc_acc = 0.0
        sum_hallucination = 0.0
        passed_validation_count = 0
        
        detailed_results = []
        
        logger.info(f"Starting execution of {total_cases} test cases...")
        for i, tc in enumerate(test_cases, start=1):
            tc_id = tc["id"]
            category = tc["category"]
            query = tc["user_query"]
            exp_behavior = tc["expected_behavior"]
            exp_truth = tc["expected_ground_truth"]
            
            logger.info(f"[{i}/{total_cases}] Running {tc_id} ({category})...")
            
            t_start = time.perf_counter()
            try:
                # Run pipeline
                result = self.rag.run(query)
                duration = time.perf_counter() - t_start
                
                # Parse expected structures for comparison
                expected_fines = self.parse_expected_fines(exp_truth)
                expected_cits = self.parse_expected_citations(exp_behavior, exp_truth)
                
                # Compute scores
                retrieval_prec = self.evaluate_retrieval_precision(result.sources, expected_cits)
                calc_acc = self.evaluate_calculation_accuracy(result.answer, expected_fines)
                hallucination_rate = self.evaluate_hallucination_rate(result)
                is_valid = result.validation.is_valid if result.validation else False
                
                if is_valid:
                    passed_validation_count += 1
                
                # Accumulate
                sum_precision += retrieval_prec
                sum_calc_acc += calc_acc
                sum_hallucination += hallucination_rate
                
                # Record result
                detailed_results.append({
                    "id": tc_id,
                    "category": category,
                    "query": query,
                    "duration_seconds": duration,
                    "actual_answer": result.answer,
                    "thought": result.thought,
                    "retrieved_chunk_ids": [c.get("chunk_id") for c in result.sources],
                    "citations_found": [str(c) for c in (result.validation.citations_found if result.validation else [])],
                    "hallucinated_citations": [str(c) for c in (result.validation.hallucinated_citations if result.validation else [])],
                    "is_valid_citation": is_valid,
                    "metrics": {
                        "retrieval_precision": retrieval_prec,
                        "calculation_accuracy": calc_acc,
                        "hallucination_rate": hallucination_rate
                    }
                })
                
                logger.info(
                    f"Finished {tc_id} in {duration:.2f}s. "
                    f"Precision: {retrieval_prec:.2f}, Calc Acc: {calc_acc:.2f}, Hallucination: {hallucination_rate:.2f}, Valid: {is_valid}"
                )
                
            except Exception as e:
                logger.exception(f"Failed to run test case {tc_id}: {e}")
                detailed_results.append({
                    "id": tc_id,
                    "category": category,
                    "query": query,
                    "error": str(e),
                    "is_valid_citation": False,
                    "metrics": {
                        "retrieval_precision": 0.0,
                        "calculation_accuracy": 0.0,
                        "hallucination_rate": 1.0
                    }
                })
                
        # Calculate final metrics
        avg_precision = sum_precision / total_cases if total_cases > 0 else 0.0
        avg_calc_acc = sum_calc_acc / total_cases if total_cases > 0 else 0.0
        avg_hallucination = sum_hallucination / total_cases if total_cases > 0 else 0.0
        validation_pass_rate = passed_validation_count / total_cases if total_cases > 0 else 0.0
        
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_test_cases": total_cases,
            "average_retrieval_precision": avg_precision,
            "average_calculation_accuracy": avg_calc_acc,
            "average_hallucination_rate": avg_hallucination,
            "validation_pass_rate": validation_pass_rate,
            "results": detailed_results
        }
        
        # Output to file
        self.results_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved evaluation results to {self.results_output_path}")
        return summary


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
        
    import argparse
    parser = argparse.ArgumentParser(description="Run Vietnamese Traffic Law Automated RAG Evaluation.")
    parser.add_argument("--test-matrix", default="data/evaluation/test_matrix.json")
    parser.add_argument("--output-json", default="data/evaluation/eval_results.json")
    parser.add_argument("--llm-backend", default=None, help="LLM backend override (e.g. 'openai' or 'ollama')")
    args = parser.parse_args()
    
    tester = TrafficLawAutomatedTester(
        test_matrix_path=args.test_matrix,
        results_output_path=args.output_json,
        llm_backend_override=args.llm_backend,
    )
    
    summary = tester.run_eval()
    
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
    
    # Output markdown table for logging
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
