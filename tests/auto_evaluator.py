import json
import logging
import re
from typing import List, Dict, Any
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)

# Thêm src vào PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))
from src.phase4_generation.rag_chain import TrafficLawRAG

def check_hallucination(actual_answer: str, expected_keywords: List[str]) -> bool:
    """Kiểm tra xem mô hình có sinh ra số tiền phạt ảo giác không (VD: 10.000.000) mà không có trong expected."""
    # Tìm tất cả các số tiền trong câu trả lời
    money_pattern = r'\b\d{1,3}(?:\.\d{3})+\b'
    found_money = set(re.findall(money_pattern, actual_answer))
    
    expected_money = set()
    for kw in expected_keywords:
        m = re.findall(money_pattern, kw)
        expected_money.update(m)
        
    for money in found_money:
        if money not in expected_money:
            return True # Có số tiền lạ -> Hallucination
    return False

def evaluate():
    logger.info("Initializing RAG pipeline for evaluation...")
    rag = TrafficLawRAG.from_configs(llm_backend_override='ollama')
    
    matrix_path = Path("tests/mutant_test_matrix.json")
    with open(matrix_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
        
    report = []
    total = len(test_cases)
    passed = 0
    
    for tc in test_cases:
        query = tc['user_query']
        expected = tc['expected_keywords']
        logger.info(f"Evaluating TC: {tc['id']} - {query}")
        
        try:
            res = rag.run(query)
            answer = res.answer
            sources = res.sources
            
            missing_keywords = []
            for kw in expected:
                if kw.lower() not in answer.lower():
                    missing_keywords.append(kw)
            
            is_pass = len(missing_keywords) == 0
            rca = None
            
            if not is_pass:
                # Phân tích Root Cause Analysis (RCA)
                if "tôi không tìm thấy" in expected[0].lower():
                    # Đáng lẽ phải từ chối nhưng mô hình lại trả lời
                    rca = "Hallucination"
                else:
                    # Đáng lẽ phải trả lời mức phạt, nhưng bị thiếu keyword
                    # Check xem chunk có chứa keyword không
                    chunks_text = " ".join([s.get('text', '') for s in sources]).lower()
                    drifted = False
                    for kw in missing_keywords:
                        if kw.lower() not in chunks_text:
                            drifted = True
                            
                    if drifted:
                        rca = "Retrieval_Drift"
                    else:
                        rca = "Context_Loss"
                        
            # Bẫy ảo giác: Mô hình tự bịa số tiền
            if check_hallucination(answer, expected) and "tôi không tìm thấy" not in answer.lower():
                is_pass = False
                rca = "Hallucination"
                
            report_item = {
                "id": tc["id"],
                "test_type": tc["test_type"],
                "query": query,
                "expected": expected,
                "actual": answer,
                "pass": is_pass,
                "missing": missing_keywords,
                "rca": rca
            }
            report.append(report_item)
            if is_pass:
                passed += 1
                logger.info(f"TC {tc['id']} PASS")
            else:
                logger.warning(f"TC {tc['id']} FAIL - RCA: {rca}")
                
        except Exception as e:
            logger.error(f"Error evaluating TC {tc['id']}: {e}")
            report.append({
                "id": tc["id"],
                "query": query,
                "pass": False,
                "rca": "System_Error",
                "error": str(e)
            })
            
        # Iteratively dump report
        report_path = Path("tests/eval_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"total": len(report), "passed": passed, "details": report}, f, ensure_ascii=False, indent=2)

    logger.info(f"Evaluation complete. Passed: {passed}/{total}. Report saved to {report_path}")

if __name__ == "__main__":
    evaluate()
