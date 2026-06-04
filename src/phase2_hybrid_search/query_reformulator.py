"""TrafficQueryReformulator — converts multi-turn follow-up queries into standalone search queries.

This solves the RAG retrieval gap in conversational contexts by performing anaphora
and ellipsis resolution (e.g. rewriting "Thế còn ô tô thì sao?" -> "Lái xe ô tô đi đúng làn
đường thì phạt bao nhiêu?").
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional
from loguru import logger

from src.phase4_generation.llm_loader import LLMClient


class TrafficQueryReformulator:
    def __init__(self, llm_client: LLMClient):
        """Initialize the query reformulator with an LLMClient.

        Args:
            llm_client: The unified LLMClient instance to perform the reformulation.
        """
        self.llm = llm_client

    def reformulate(self, query: str, chat_history: Optional[List[Dict[str, str]]]) -> str:
        """Resolve pronouns and conversational gaps to produce a standalone search query.

        Args:
            query: The current conversational user query.
            chat_history: A list of dicts representing the active chat history.

        Returns:
            The reformulated standalone search query.
        """
        if not chat_history:
            return query

        # Keep only the last 5 turns to minimize prompt size but ensure context is not lost
        recent_history = chat_history[-5:]
        history_lines = []
        for turn in recent_history:
            role = "Người dùng" if turn["role"] == "user" else "Trợ lý"
            content = turn.get("content") or ""
            # Strip any internal XML thought tags from history so they don't clutter the prompt
            content_clean = re.sub(
                r"<thought>.*?</thought>", "", content, flags=re.DOTALL | re.IGNORECASE
            ).strip()
            # Truncate very long assistant replies to keep local inference speed fast and focus on context
            if len(content_clean) > 200:
                content_clean = content_clean[:200] + "..."
            history_lines.append(f"- {role}: {content_clean}")

        history_block = "\n".join(history_lines)

        system_prompt = (
            "Bạn là một trợ lý ảo chuyên viết lại câu hỏi tiếp nối của người dùng thành một câu hỏi ĐỘC LẬP duy nhất "
            "để dùng làm từ khóa tìm kiếm luật giao thông Việt Nam.\n"
            "Yêu cầu cực kỳ quan trọng:\n"
            "1. BẮT BUỘC KẾ THỪA TOÀN BỘ HÀNH VI VI PHẠM từ lịch sử. Nếu người dùng hỏi 'Thế còn xe máy thì sao?', bạn phải chép lại toàn bộ hành vi vi phạm ở câu trước nhưng đổi phương tiện thành xe máy.\n"
            "2. Nếu câu hỏi mới không có phương tiện cụ thể, hãy tự động kế thừa phương tiện ở lịch sử.\n"
            "3. TUYỆT ĐỐI CHỈ TRẢ VỀ DUY NHẤT 1 CÂU HỎI ĐỘC LẬP. Không giải thích, không tự trả lời câu hỏi, không chào hỏi.\n"
            "4. NGHIÊM CẤM TỰ TRẢ LỜI. BẠN LÀ BỘ VIẾT LẠI CÂU HỎI, KHÔNG PHẢI BỘ TRẢ LỜI. KHÔNG ĐƯỢC BỊA RA SỐ TIỀN PHẠT HAY KẾT LUẬN LUẬT."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            # Few-shot example 1
            {
                "role": "user",
                "content": (
                    "Lịch sử:\n"
                    "- Người dùng: Lái xe ô tô đi ngược chiều bị phạt thế nào?\n"
                    "- Trợ lý: Đi ngược chiều đối với xe ô tô bị phạt từ 4.000.000 đến 6.000.000 đồng.\n"
                    "Câu hỏi mới của người dùng: Vậy xe máy thì sao?"
                )
            },
            {"role": "assistant", "content": "Lái xe máy đi ngược chiều bị phạt thế nào?"},
            # Few-shot example 2
            {
                "role": "user",
                "content": (
                    "Lịch sử:\n"
                    "- Người dùng: Xe máy không gương chiếu hậu bị phạt bao nhiêu?\n"
                    "- Trợ lý: Không gương phạt từ 100.000 đến 200.000 đồng.\n"
                    "Câu hỏi mới của người dùng: Không mang bằng lái xe thì phạt bao nhiêu?"
                )
            },
            {"role": "assistant", "content": "Xe máy không mang bằng lái xe phạt bao nhiêu?"},
            # Real task
            {
                "role": "user",
                "content": f"Lịch sử:\n{history_block}\nCâu hỏi mới của người dùng: {query}"
            }
        ]

        try:
            logger.info(f"[query_reformulator] reformulating query: '{query}'")
            resp = self.llm.generate(messages, max_new_tokens=100, temperature=0.0)
            text = resp.text.strip()
            
            # Clean up wrapping quotes if added by the LLM
            if (text.startswith('"') and text.endswith('"')) or (
                text.startswith("'") and text.endswith("'")
            ):
                text = text[1:-1].strip()

            # Robust post-processing: remove any conversational prefixes
            for prefix in ["Câu hỏi viết lại:", "Viết lại:", "Câu hỏi độc lập:", "Câu hỏi độc lập viết lại:", "Viết lại câu hỏi:"]:
                if text.lower().startswith(prefix.lower()):
                    text = text[len(prefix):].strip()
            
            if text:
                logger.info(f"[query_reformulator] query reformulated: '{query}' -> '{text}'")
                return text
            return query
        except Exception as e:
            logger.error(f"[query_reformulator] failed to reformulate query: {e}")
            return query

