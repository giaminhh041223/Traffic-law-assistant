"""TrafficAIRouter — smart router to dispatch queries to local CPU vs commercial cloud LLM.

This module evaluates prompt complexity (length, compound intent, presence of legal dispute
or extreme liability keywords like 'tai nạn', 'tử vong', 'hình sự') and routes them.
Includes a session-aware cloud rate limiter to protect commercial cloud API quotas.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional
from loguru import logger

from src.phase2_hybrid_search.query_deconstructor import TrafficQueryDeconstructor
from src.utils.session_manager import get_or_create_streamlit_session_id


class TrafficAIRouter:
    def __init__(
        self,
        deconstructor: Optional[TrafficQueryDeconstructor] = None,
        complexity_threshold_words: int = 30,
        max_cloud_calls_per_minute: int = 5,
    ):
        self.deconstructor = deconstructor or TrafficQueryDeconstructor()
        self.complexity_threshold_words = complexity_threshold_words
        self.max_cloud_calls_per_minute = max_cloud_calls_per_minute
        
        # Session-based request tracking for rate limiting
        # Maps session_id -> list of float timestamps
        self.cloud_request_timestamps: Dict[str, List[float]] = {}
        
        # High-complexity legal dispute, severe liability, or criminal keywords
        self.complex_keywords = [
            "tai nạn",
            "tử vong",
            "chết người",
            "hình sự",
            "tòa án",
            "khởi tố",
            "truy cứu",
            "đền bù",
            "bồi thương",
            "thương tật",
            "nồng độ cồn kịch khung",
            "gây tai nạn",
            "bỏ trốn",
            "chống người thi hành công vụ",
            "hồ sơ vụ án",
            
            # --- Legal Dispute & Reasoning Keywords ---
            "khiếu nại",
            "biên bản",
            "mờ",
            "không rõ",
            "phân biệt",
            "mâu thuẫn",
            "cấp cứu",
            "bệnh viện",
            "tình thế cấp thiết",
            "bất khả kháng",
            "đúng không",
            "có đúng không",
        ]

    def evaluate_complexity(self, query: str) -> str:
        """Analyze query complexity and determine the routing target.

        Args:
            query: The raw user query.

        Returns:
            "local" if simple/standard, "cloud" if highly complex or severe.
        """
        if not query:
            return "local"

        clean_query = query.strip()
        words = clean_query.split()
        word_count = len(words)

        # Heuristic 1: Extremely long prompts (multi-sentence explanations)
        if word_count > self.complexity_threshold_words:
            logger.info(f"[ai_router] query categorized as 'cloud' (word count {word_count} > threshold {self.complexity_threshold_words})")
            return "cloud"

        # Heuristic 2: Multiple legal intents (compound query)
        sub_queries = self.deconstructor.deconstruct(clean_query)
        if len(sub_queries) >= 3:
            logger.info(f"[ai_router] query categorized as 'cloud' (compound query split into {len(sub_queries)} intents)")
            return "cloud"

        # Heuristic 3: Severe liability / criminal keywords
        query_lower = clean_query.lower()
        matched_keywords = [k for k in self.complex_keywords if k in query_lower]
        if matched_keywords:
            logger.info(f"[ai_router] query categorized as 'cloud' (matched dispute/criminal keywords: {matched_keywords})")
            return "cloud"

        logger.debug(f"[ai_router] query categorized as 'local' (simple, low-complexity standard query)")
        return "local"

    def route_generation(
        self,
        query: str,
        local_generator_fn: Any,
        cloud_generator_fn: Any,
        *args,
        **kwargs,
    ) -> Any:
        """Route generation task to the appropriate engine with Rate Limiting protection.

        Args:
            query: The user query.
            local_generator_fn: Callable for local CPU-based LLM generation.
            cloud_generator_fn: Callable for commercial cloud-based LLM generation.

        Returns:
            The generated response object from the selected engine.
        """
        target = self.evaluate_complexity(query)
        if target == "cloud":
            # Verify if API key exists for cloud run
            if not os.environ.get("OPENAI_API_KEY"):
                logger.warning("[ai_router] cloud routing triggered but OPENAI_API_KEY not found. Falling back to local generation.")
                return local_generator_fn(*args, **kwargs)
            
            # --- Rate Limiter Protection ---
            session_id = get_or_create_streamlit_session_id()
            now = time.time()
            
            # Initialize or clean timestamps
            if session_id not in self.cloud_request_timestamps:
                self.cloud_request_timestamps[session_id] = []
            
            # Keep only timestamps in the last 60 seconds
            active_timestamps = [t for t in self.cloud_request_timestamps[session_id] if now - t < 60.0]
            self.cloud_request_timestamps[session_id] = active_timestamps
            
            if len(active_timestamps) >= self.max_cloud_calls_per_minute:
                logger.warning(
                    f"[ai_router] Rate limit exceeded for session '{session_id}'! "
                    f"Spam detected: {len(active_timestamps)} cloud calls in the last 60 seconds "
                    f"(limit: {self.max_cloud_calls_per_minute}/min). "
                    f"FALLING BACK TO FREE LOCAL OLLAMA ENGINE to protect API quotas."
                )
                return local_generator_fn(*args, **kwargs)
            
            # Register allowed call timestamp
            self.cloud_request_timestamps[session_id].append(now)
            logger.info(
                f"[ai_router] routing execution path to COMMERCIAL CLOUD LLM. "
                f"Session '{session_id}' call volume: {len(self.cloud_request_timestamps[session_id])}/min"
            )
            return cloud_generator_fn(*args, **kwargs)
        
        logger.info("[ai_router] routing execution path to LOCAL CPU LLM.")
        return local_generator_fn(*args, **kwargs)
