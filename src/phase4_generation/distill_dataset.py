"""Dataset Distiller — synthesizes training data for legal QLoRA fine-tuning.

This script ingests raw legal text chunks, calls a frontier OpenAI model (e.g., gpt-4o-mini)
to synthesize realistic Vietnamese traffic law questions, and generates expert-grade
step-by-step reasoning (<thought>) and perfectly cited answers.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from loguru import logger
from openai import OpenAI

from src.phase4_generation.prompt_templates import SYSTEM_PROMPT_VI
from src.utils.io import read_jsonl, load_yaml


class DatasetDistiller:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model_name = model_name
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY environment variable.")
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def distill_chunk(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Synthesize a high-quality Q&A pair from a single chunk."""
        text = chunk.get("text", "").strip()
        title = chunk.get("dieu_title", "").strip()
        citation = chunk.get("full_citation", "").strip()
        dieu = chunk.get("dieu")
        khoan = chunk.get("khoan")
        diem = chunk.get("diem")
        doc_short = chunk.get("doc_short", "ND100")

        # Map ND100_123 to standard Decree for citation examples
        doc_name = "Nghị định 100/2019/NĐ-CP"
        if doc_short == "ND123":
            doc_name = "Nghị định 123/2021/NĐ-CP"

        if not text:
            return None

        # Build prompt for synthesizer
        system_prompt = (
            "Bạn là một chuyên gia soạn thảo câu hỏi kiểm thử và giáo án đào tạo Luật giao thông đường bộ Việt Nam. "
            "Nhiệm vụ của bạn là đọc đoạn văn bản pháp luật được cung cấp, sau đó sinh ra một cặp Câu hỏi và Câu trả lời "
            "tối ưu dành cho việc tinh chỉnh (fine-tuning) mô hình RAG cục bộ."
        )

        user_prompt = (
            f"Văn bản pháp luật:\nTiêu đề: {title}\nNội dung: {text}\nTrích dẫn gốc: {citation}\n\n"
            "YÊU CẦU SOẠN THẢO:\n"
            "1. Sinh một Câu hỏi thực tế (người dân thường hỏi, có thể dùng tiếng lóng hoặc từ đồng nghĩa như xe máy, vượt đèn đỏ) "
            "mà thông tin TRONG ĐOẠN VĂN BẢN TRÊN ĐÁP ỨNG ĐẦY ĐỦ để trả lời.\n"
            "2. Sinh một Câu trả lời chính xác, trung thực, chỉ dựa vào văn bản quy phạm pháp luật trên, cấu trúc đúng định dạng sau:\n"
            "   Trước tiên, giải trình suy nghĩ lập luận từng bước trong thẻ XML <thought>...</thought> (phân tích phương tiện, hành vi vi phạm, mức phạt, trích dẫn).\n"
            "   Sau đó, ghi câu trả lời chính thức ở dưới bắt đầu bằng cụm từ 'Trả lời: ', kèm trích dẫn luật quy chuẩn ngay sau khẳng định pháp lý.\n"
            f"   Định dạng trích dẫn bắt buộc: (Điểm {diem or '<chữ>'}, Khoản {khoan or '<số>'}, Điều {dieu or '<số>'} {doc_name}).\n\n"
            "Hãy trả về kết quả dưới dạng JSON thuần túy theo cấu trúc sau, không thêm lời dẫn giải:\n"
            "{\n"
            "  \"question\": \"<câu hỏi sinh ra>\",\n"
            "  \"answer\": \"<thought>\\n- Lập luận...\\n</thought>\\nTrả lời: <câu trả lời chính thức kèm trích dẫn>\"\n"
            "}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            res_data = json.loads(resp.choices[0].message.content)
            return {
                "system": SYSTEM_PROMPT_VI,
                "context": text,
                "question": res_data.get("question", "").strip(),
                "answer": res_data.get("answer", "").strip()
            }
        except Exception as e:
            logger.error(f"[distill_dataset] failed to distill chunk: {e}")
            return None

    def distill_pipeline(
        self,
        chunks_jsonl: str = "data/chunks.jsonl",
        output_jsonl: str = "data/distilled_dataset.jsonl",
        max_samples: int = 100,
    ) -> List[Dict[str, Any]]:
        """Run the end-to-end knowledge distillation over the chunks corpus."""
        chunks_path = Path(chunks_jsonl)
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {chunks_jsonl}")

        logger.info(f"Reading chunks from {chunks_jsonl} ...")
        all_chunks = list(read_jsonl(chunks_path))
        logger.info(f"Loaded {len(all_chunks)} chunks.")

        # Limit samples for current run
        samples_to_process = all_chunks[:max_samples]
        distilled_data: List[Dict[str, Any]] = []

        logger.info(f"Starting dataset synthesis (limit={max_samples}) ...")
        
        # Open output file in append/write mode
        out_path = Path(output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "w", encoding="utf-8") as f:
            for c in tqdm(samples_to_process, desc="Distill Chunks"):
                result = self.distill_chunk(c)
                if result:
                    distilled_data.append(result)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()

        logger.info(f"Distillation complete. Synthesized {len(distilled_data)} samples. Saved to {output_jsonl}")
        return distilled_data


if __name__ == "__main__":
    # Standalone diagnostic run
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Try loading openAI key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY environment variable not set. Synthesizer running in dry-run/mock mode.")
    else:
        distiller = DatasetDistiller(api_key=api_key)
        # Distill 5 chunks for dry run verification
        distiller.distill_pipeline(max_samples=5)
