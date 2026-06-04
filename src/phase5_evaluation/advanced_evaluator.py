import json
import os
import re
from typing import List, Dict, Any
from src.phase4_generation.rag_chain import TrafficLawRAG

class CognitiveJudge:
    @staticmethod
    def check_self_contradiction(answer: str) -> bool:
        """Detects if the answer contradicts itself (e.g., saying no fine, but then listing a fine)."""
        answer_lower = answer.lower()
        no_fine_patterns = ["không bị phạt", "không bị xử phạt", "không vi phạm", "được miễn"]
        fine_patterns = ["phạt tiền từ", "bị phạt tiền", "mức phạt là"]
        
        has_no_fine = any(p in answer_lower for p in no_fine_patterns)
        has_fine = any(p in answer_lower for p in fine_patterns)
        
        # If both are strongly stated, it's a contradiction.
        return has_no_fine and has_fine

    @staticmethod
    def check_citation_drift(answer: str, target_articles: List[str]) -> bool:
        """Checks if the expected target articles are missing from the answer."""
        answer_lower = answer.lower()
        for article in target_articles:
            # Simple heuristic: remove punctuation and check if the core article is cited
            article_core = article.lower().split(",")[0].strip()
            if article_core not in answer_lower:
                return True # Drift detected!
        return False
        
    @staticmethod
    def check_missing_entities(answer: str, expected_entities: List[str]) -> List[str]:
        """Returns a list of expected entities that are missing from the answer."""
        answer_lower = answer.lower()
        missing = []
        for entity in expected_entities:
            if entity.lower() not in answer_lower:
                missing.append(entity)
        return missing

def analyze_failure_modes(failed_cases: List[Dict[str, Any]]):
    """Root Cause Analysis Agent logic."""
    for case in failed_cases:
        test_type = case.get("test_type")
        errors = case.get("errors", [])
        
        root_causes = []
        if test_type == "vehicle_bias":
            if "missing_entities" in errors:
                root_causes.append("Lỗi Module 2 (Query Deconstructor): Không có Vehicle Prefix Propagation hoặc không ép được việc sinh 2 sub-queries cho ô tô và xe máy.")
        elif test_type == "multi_turn":
            if "missing_entities" in errors or "citation_drift" in errors:
                root_causes.append("Lỗi Module 1 (Query Reformulator): Conversational Context Loss, không bảo toàn được hành vi vi phạm khi người dùng đổi chủ ngữ (xe máy).")
        elif test_type == "exception_clause":
            if "self_contradiction" in errors:
                root_causes.append("Lỗi System Prompt & Context Window: Attention-Dilution. LLM không chú ý đúng mức vào mệnh đề ngoại lệ 'Trừ trường hợp...'.")
        elif test_type == "numeric_trap":
            if "self_contradiction" in errors or "citation_drift" in errors:
                root_causes.append("Lỗi Bẫy thông số (Numeric Trap): RAG kéo sai chunk hoặc LLM hiểu sai đơn vị.")
                
        if not root_causes:
            root_causes.append("Lỗi chưa phân loại, cần kiểm tra thủ công.")
            
        case["root_causes"] = root_causes
        
    return failed_cases

def run_evaluation():
    matrix_path = "tests/advanced_test_matrix.json"
    if not os.path.exists(matrix_path):
        print(f"Error: {matrix_path} not found.")
        return

    with open(matrix_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    rag = TrafficLawRAG.from_configs(
        settings_yaml="configs/settings.yaml",
        retrieval_yaml="configs/retrieval.yaml",
        generation_yaml="configs/generation.yaml"
    )
    failed_cases = []
    
    print(f"Running LLM-as-a-judge Cognitive Tester on {len(test_cases)} edge-cases...\n")

    for test in test_cases:
        print(f"Testing ID: {test['id']} | Type: {test['test_type']}")
        user_query = test["user_query"]
        expected_entities = test.get("expected_entities", [])
        target_articles = test.get("target_articles", [])
        
        answer = ""
        try:
            if isinstance(user_query, list):
                # Multi-turn simulation
                chat_history = []
                for idx, q in enumerate(user_query):
                    print(f"  -> Turn {idx+1}: {q}")
                    result = rag.run(q, chat_history=chat_history)
                    chat_history.append({"role": "user", "content": q})
                    chat_history.append({"role": "assistant", "content": result.answer})
                    if idx == len(user_query) - 1:
                        answer = result.answer
            else:
                print(f"  -> Query: {user_query}")
                result = rag.run(user_query, chat_history=[])
                answer = result.answer
                
            errors = []
            
            # Cognitive Judge Evaluation
            if CognitiveJudge.check_self_contradiction(answer):
                errors.append("self_contradiction")
                
            if CognitiveJudge.check_citation_drift(answer, target_articles):
                errors.append("citation_drift")
                
            missing_entities = CognitiveJudge.check_missing_entities(answer, expected_entities)
            if missing_entities:
                errors.append("missing_entities")
                
            if errors:
                failed_cases.append({
                    "id": test["id"],
                    "test_type": test["test_type"],
                    "query": user_query,
                    "answer": answer,
                    "errors": errors,
                    "missing_entities": missing_entities
                })
                print(f"  ❌ FAILED: {errors}")
            else:
                print(f"  ✅ PASSED")
                
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            failed_cases.append({
                "id": test["id"],
                "test_type": test["test_type"],
                "query": user_query,
                "answer": "ERROR",
                "errors": ["system_exception", str(e)]
            })

    if failed_cases:
        print("\nAnalyzing Failure Modes...")
        failed_cases = analyze_failure_modes(failed_cases)
        
        with open("failed_cases.json", "w", encoding="utf-8") as f:
            json.dump(failed_cases, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(failed_cases)} failures to failed_cases.json")
    else:
        print("\n🎉 ALL TESTS PASSED!")

if __name__ == "__main__":
    run_evaluation()
