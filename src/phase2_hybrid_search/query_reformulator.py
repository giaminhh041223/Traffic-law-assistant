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

        # Keep only the last 3 turns to minimize prompt size and keep local CPU inference fast
        recent_history = chat_history[-3:]
        history_lines = []
        for turn in recent_history:
            role = "Người dùng" if turn["role"] == "user" else "Trợ lý"
            content = turn.get("content") or ""
            # Strip any internal XML thought tags from history so they don't clutter the prompt
            content_clean = re.sub(
                r"<thought>.*?</thought>", "", content, flags=re.DOTALL | re.IGNORECASE
            ).strip()
            history_lines.append(f"{role}: {content_clean}")

        history_block = "\n".join(history_lines)

        system_prompt = (
            "Bạn là trợ lý ngôn ngữ chuyên viết lại câu hỏi tiếp theo thành một câu hỏi ĐỘC LẬP đầy đủ ý nghĩa, "
            "phù hợp để làm từ khóa tìm kiếm trong cơ sở dữ liệu luật giao thông đường bộ Việt Nam.\n"
            "Hãy đảm bảo câu hỏi viết lại:\n"
            "1. Xác định rõ loại phương tiện (xe ô tô, xe mô tô, xe gắn máy, xe đạp...) và hành vi vi phạm pháp luật đang được bàn luận.\n"
            "2. Tuyệt đối KHÔNG tự trả lời câu hỏi, KHÔNG thêm lời chào hay giải thích, chỉ đưa ra duy nhất câu hỏi độc lập được viết lại.\n"
            "3. KẾ THỪA PHƯƠNG TIỆN (VEHICLE TYPE INHERITANCE): Nếu câu hỏi tiếp theo của người dùng không nhắc đến loại phương tiện nào (ô tô, mô tô, xe máy, xe đạp...) nhưng trong lịch sử hội thoại trước đó đang nói về một loại phương tiện cụ thể (ví dụ: xe cọp/xe mô tô), bạn BẮT BUỘC phải đưa loại phương tiện đó vào câu hỏi viết lại. Tuyệt đối không để câu hỏi viết lại ở dạng chung chung thiếu phương tiện."
        )

        user_prompt = (
            f"Lịch sử hội thoại trước đó:\n"
            f"{history_block}\n\n"
            f"Câu hỏi tiếp theo của người dùng: {query}\n"
            f"Câu hỏi độc lập viết lại:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
            
            if text:
                logger.info(f"[query_reformulator] query reformulated: '{query}' -> '{text}'")
                return text
            return query
        except Exception as e:
            logger.error(f"[query_reformulator] failed to reformulate query: {e}")
            return query
