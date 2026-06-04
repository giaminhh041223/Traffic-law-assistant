"""TrafficAIRouter - smart router to dispatch queries to local CPU vs commercial cloud LLM."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional
from loguru import logger

from src.phase2_hybrid_search.query_deconstructor import TrafficQueryDeconstructor
from src.utils.session_manager import get_or_create_streamlit_session_id
from src.phase4_generation.llm_loader import LLMResponse

try:
    import streamlit as st
    from google import genai
    from google.genai import types
    from openai import OpenAI
    
    HAS_CLOUD = True
    try:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        import toml
        # Fallback for CLI testing without Streamlit
        secrets_path = os.path.join(os.path.dirname(__file__), "..", "..", ".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            secrets = toml.load(secrets_path)
            GEMINI_API_KEY = secrets.get("GEMINI_API_KEY")
            OPENAI_API_KEY = secrets.get("OPENAI_API_KEY")
        else:
            GEMINI_API_KEY = None
            OPENAI_API_KEY = None
            
    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        gemini_client = None
        
    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        openai_client = None
        
except ImportError:
    HAS_CLOUD = False


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
        self.cloud_request_timestamps: Dict[str, List[float]] = {}
        
        self.force_cloud_keywords = ["cứu thương", "ưu tiên", "trực thăng", "xe lăn", "bỏ chạy", "chống đối"]
        self.complex_keywords = [
            "tai nạn",
            "tử vong",
            "chết người",
            "hình sự",
            "tòa án",
            "khởi tố",
            "truy cứu",
            "đền bù",
            "bồi thường",
            "thương tật",
            "nồng độ cồn kịch khung",
            "gây tai nạn",
            "bỏ trốn",
            "chống người thi hành công vụ",
            "hồ sơ vụ án",
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

    def evaluate_complexity(self, query: str, metadata_entities: Optional[List[str]] = None) -> str:
        """Xác định xem truy vấn có chứa yếu tố bẫy phức hợp hay không"""
        if not query:
            return "local"
            
        clean_query = query.strip()
        words = clean_query.split()
        word_count = len(words)

        # 1. Heuristic 1: Extremely long prompts
        if word_count > self.complexity_threshold_words:
            return "cloud"
        
        if metadata_entities is None:
            metadata_entities = self.deconstructor.deconstruct(clean_query)
            
        # 2. Heuristic 2: Multiple legal intents (compound query)
        if len(metadata_entities) >= 3:
            return "cloud"
            
        # 3. Heuristic 3: Force cloud keywords
        query_lower = clean_query.lower()
        if any(kw in query_lower for kw in self.force_cloud_keywords):
            return "cloud"
            
        # 4. Heuristic 4: Severe liability / criminal keywords
        if any(kw in query_lower for kw in self.complex_keywords):
            return "cloud"
            
        return "local"

    def route_generation(self, query: str, context: str, route_status: str, local_generator_fn: Any, *args, **kwargs) -> Any:
        """Định tuyến luồng sinh chữ dựa trên độ phức tạp"""
        messages = kwargs.get("messages", [])
        
        # Prepare content structures for cloud LLMs from messages to preserve strict system prompt
        system_prompt = ""
        gemini_contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                role = "model" if msg["role"] == "assistant" else msg["role"]
                gemini_contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=msg["content"])]
                    )
                )
        
        # Default fallback string prompt if messages list is empty
        prompt = f"Ngữ cảnh luật:\n{context}\n\nCâu hỏi: {query}"
        if not gemini_contents:
            gemini_contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        
        if route_status == "cloud" and HAS_CLOUD:
            # Route to cloud using Gemini as primary, OpenAI as fallback
            if gemini_client:
                logger.info("[ai_router] Routing execution path to COMMERCIAL CLOUD LLM (Gemini 1.5 Pro).")
                try:
                    response = gemini_client.models.generate_content(
                        model='gemini-1.5-pro',
                        contents=gemini_contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt if system_prompt else None,
                        )
                    )
                    return LLMResponse(text=response.text, usage={})
                except Exception as e:
                    logger.error(f"[ai_router] Gemini API Failed: {e}. Trying OpenAI...")
            if openai_client:
                logger.info("[ai_router] Routing execution path to COMMERCIAL CLOUD LLM (GPT-4o-mini).")
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages if messages else [{"role": "user", "content": prompt}]
                    )
                    return LLMResponse(text=response.choices[0].message.content, usage={})
                except Exception as e:
                    logger.error(f"[ai_router] OpenAI API Failed: {e}. Falling back to LOCAL_SLM.")
            
            logger.warning("[ai_router] No cloud client available or API failed. Falling back to LOCAL_SLM.")
            return local_generator_fn(*args, **kwargs)
            
        elif route_status == "FORCE_CLOUD_GEMINI" and HAS_CLOUD and gemini_client:
            logger.info("[ai_router] Routing execution path to COMMERCIAL CLOUD LLM (Gemini 1.5 Pro).")
            try:
                response = gemini_client.models.generate_content(
                    model='gemini-1.5-pro',
                    contents=gemini_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt if system_prompt else None,
                    )
                )
                return LLMResponse(text=response.text, usage={})
            except Exception as e:
                logger.error(f"[ai_router] Gemini API Failed: {e}. Falling back to LOCAL_SLM.")
                return local_generator_fn(*args, **kwargs)
            
        elif route_status == "FORCE_CLOUD_CHATGPT" and HAS_CLOUD and openai_client:
            logger.info("[ai_router] Routing execution path to COMMERCIAL CLOUD LLM (GPT-4o-mini).")
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages if messages else [{"role": "user", "content": prompt}]
                )
                return LLMResponse(text=response.choices[0].message.content, usage={})
            except Exception as e:
                logger.error(f"[ai_router] OpenAI API Failed: {e}. Falling back to LOCAL_SLM.")
                return local_generator_fn(*args, **kwargs)
            
        else:
            # Luồng xử lý bình thường bằng mô hình Local để tiết kiệm chi phí
            logger.info("[ai_router] Routing execution path to LOCAL CPU LLM.")
            return local_generator_fn(*args, **kwargs)
