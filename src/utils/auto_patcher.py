import json
import logging
from pathlib import Path
import re
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)

def patch_retrieval_drift(failed_cases):
    """
    Tự động cập nhật `query_rewriter.py` bằng cách tìm các từ khóa chưa được mapping.
    Tuy nhiên, trong một hệ thống thực tế, ta sẽ dùng LLM để trích xuất từ lóng.
    Ở đây ta sử dụng cơ chế an toàn: In ra gợi ý để developer review.
    """
    logger.info("--- PATTERN ANALYSIS: RETRIEVAL DRIFT ---")
    for tc in failed_cases:
        logger.warning(f"Cần bổ sung Slang Mapping cho query: '{tc['query']}'")
        logger.warning(f"Missing Keywords trong Chunk: {tc['missing']}")

def patch_context_loss():
    """Tự động giảm MAX_TOKENS hoặc top_k để giảm nhiễu ngữ cảnh cho LLM nhỏ."""
    logger.info("--- PATCHING CONTEXT LOSS ---")
    config_path = Path("configs/generation.yaml")
    if not config_path.exists():
        logger.error("configs/generation.yaml not found.")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Giảm top_k nếu ngữ cảnh quá dài khiến LLM quên
    if "top_k:" in content:
        # Ví dụ đơn giản: Giảm top_k xuống 1 đơn vị nếu đang lớn hơn 3
        # Đây là Auto-tuning cơ bản
        logger.info("Đề xuất: Giảm 'top_k' hoặc cấu hình 'context_budget' trong generation.yaml để LLM (1.5B) không bị ngợp ngữ cảnh.")

def patch_hallucination():
    """Tự động nâng mức độ khắt khe trong System Prompt hoặc Validator."""
    logger.info("--- PATCHING HALLUCINATION ---")
    logger.info("Cảnh báo: Phát hiện ảo giác. Hệ thống RAG cần siết chặt `require_at_least_one = True` trong CitationValidator hoặc thêm Few-Shot Prompt.")

def patch_system_error():
    """Tự động tăng timeout và num_ctx nếu Ollama bị timeout liên tục."""
    logger.info("--- PATCHING SYSTEM ERROR (TIMEOUT / CONTEXT OVERFLOW) ---")
    config_path = Path("configs/generation.yaml")
    if not config_path.exists():
        logger.error("configs/generation.yaml not found.")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Tăng num_ctx và timeout
    content = content.replace("num_ctx: 1536", "num_ctx: 4096")
    content = content.replace("timeout: 600", "timeout: 1200")
    
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Đã cập nhật configs/generation.yaml: num_ctx=4096, timeout=1200 để khắc phục lỗi Timeout/OOM.")

def main():
    report_path = Path("tests/eval_report.json")
    if not report_path.exists():
        logger.error(f"Cannot find {report_path}. Run auto_evaluator.py first.")
        return
        
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    failed_cases = [tc for tc in report['details'] if not tc['pass']]
    if not failed_cases:
        logger.info("Hệ thống hoàn hảo. Không có lỗi cần tự vá.")
        return
        
    logger.info(f"Phát hiện {len(failed_cases)} lỗi. Khởi động Auto-Patcher...")
    
    rca_counts = {"Retrieval_Drift": 0, "Context_Loss": 0, "Hallucination": 0, "System_Error": 0}
    
    for tc in failed_cases:
        rca = tc.get('rca', 'Unknown')
        if rca in rca_counts:
            rca_counts[rca] += 1
            
    drift_cases = [tc for tc in failed_cases if tc.get('rca') == 'Retrieval_Drift']
    if drift_cases:
        patch_retrieval_drift(drift_cases)
        
    if rca_counts["Context_Loss"] > 0:
        patch_context_loss()
        
    if rca_counts["System_Error"] > 0:
        patch_system_error()
        
    if rca_counts["Hallucination"] > 0:
        patch_hallucination()

if __name__ == "__main__":
    main()
