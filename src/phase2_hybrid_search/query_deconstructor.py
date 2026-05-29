"""TrafficQueryDeconstructor — splits compound multi-intent queries into atomic sub-queries.

This module splits queries with multiple violations (e.g. crossing a red light AND driving
without a license) into individual sub-queries, executes them in parallel, and merges
the retrieved legal chunks, preventing retrieval gaps.
"""
from __future__ import annotations

import re
import concurrent.futures
from typing import Any, Dict, List, Optional, Set
from loguru import logger

from src.phase2_hybrid_search.hybrid_search import HybridSearcher


class TrafficQueryDeconstructor:
    def __init__(self, connectors: Optional[List[str]] = None):
        # Traditional Vietnamese linguistic conjunctions for compound legal queries
        self.connectors = connectors or [
            r"\band\b",
            r"\bđồng thời\b",
            r"\bnhưng\b",
            r"\bvà cả\b",
            r"\bvà\b",
            r"\bkèm theo cả\b",
            r"\bkèm theo\b",
            r"\bcộng thêm\b",
            r"\bvừa\b",
            r"\blại còn\b",
            r"\bvới hành vi\b",
            r"\bnhưng không\b",
        ]
        # Common violation keywords that signal separate intents in run-on sentences.
        # We split BEFORE these using a positive lookahead (?=...) so they are preserved in their parts.
        self.violation_keywords = [
            r"lạng lách",
            r"đánh võng",
            r"vượt đèn đỏ",
            r"không đội mũ bảo hiểm",
            r"không đội nón bảo hiểm",
            r"không mang theo",
            r"không có bằng lái",
            r"uống rượu",
            r"say rượu",
            r"nồng độ cồn",
            r"chất kích thích",
            r"chất ma túy",
            r"gây tai nạn",
            r"gây tai nạn giao thông",
            r"chở người bệnh đi cấp cứu",
            r"chở đi bệnh viện",
            r"đi bệnh viện",
            r"chở đi cấp cứu",
            r"đi cấp cứu",
            r"đấm cán bộ",
            r"đâm cán bộ",
            r"đấm công an",
            r"chống người thi hành",
            
            # --- New Violation Keywords for Robustness ---
            r"thắt dây",
            r"dây an toàn",
            r"nghe điện thoại",
            r"sử dụng điện thoại",
            r"bảo hiểm bắt buộc",
            r"không có bảo hiểm",
            r"không mang theo bảo hiểm",
            r"không trình ra",
            r"chở quá số người",
            r"chưa đủ tuổi",
            r"quá tốc độ",
            r"chạy quá tốc độ",
            r"chủ xe",
            r"tài xế",
            r"tạm giữ phương tiện",
            r"tạm giữ",
            r"giam xe",
            r"bị giam",
            
            # --- Automated Tester Additions ---
            r"không gắn biển",
            r"không có biển số",
            r"đóng bỉm",
            r"giao xe",
            r"cho mượn xe",
            r"đâm công an",
        ]
        lookahead_pattern = r"(?=\b(?:" + "|".join(self.violation_keywords) + r")\b)"
        connectors_pattern = r"[.,;]|\b" + r"\b|\b".join(self.connectors) + r"\b"
        
        # Split on punctuation, explicit connectors, or right before a run-on violation keyword
        self.split_regex = re.compile(
            f"{lookahead_pattern}|{connectors_pattern}", re.IGNORECASE
        )

    def deconstruct(self, query: str) -> List[str]:
        """Split a compound Vietnamese query into atomic legal questions.

        Args:
            query: The raw compound user query.

        Returns:
            A list of atomic sub-queries.
        """
        if not query:
            return []

        clean_query = query.strip()
        
        # Split on conjunctions and punctuation
        parts = self.split_regex.split(clean_query)
        sub_queries: List[str] = []
        
        for part in parts:
            trimmed = part.strip()
            # Heuristics: keep only if it has a violation keyword, or is a sufficiently long descriptive clause (>= 15 chars)
            has_violation = any(re.search(kw, trimmed, re.IGNORECASE) for kw in self.violation_keywords)
            if has_violation or len(trimmed) >= 15:
                # If a sub-part is missing key subject/context like "xe máy", 
                # prepend the vehicle prefix if found in the main query
                vehicle_prefix = ""
                lower_query = clean_query.lower()
                lower_trimmed = trimmed.lower()
                
                # Check for "xe máy", "mô tô", "xe gắn máy"
                has_moto_main = any(x in lower_query for x in ["xe máy", "mô tô", "xe gắn máy"])
                has_moto_sub = any(x in lower_trimmed for x in ["xe máy", "mô tô", "xe gắn máy"])
                
                # Check for "ô tô" by ignoring "mô tô"
                query_no_moto = lower_query.replace("mô tô", "")
                trimmed_no_moto = lower_trimmed.replace("mô tô", "")
                
                has_auto_main = "ô tô" in query_no_moto
                has_auto_sub = "ô tô" in trimmed_no_moto
                
                if has_moto_main and not has_moto_sub:
                    vehicle_prefix = "xe mô tô "
                elif has_auto_main and not has_auto_sub:
                    vehicle_prefix = "xe ô tô "
                
                sub_queries.append(f"{vehicle_prefix}{trimmed}")

        # Fallback: if no splitting occurred, return the original query
        if not sub_queries:
            sub_queries = [clean_query]

        logger.info(f"[query_deconstructor] split query into {len(sub_queries)} sub-queries: {sub_queries}")
        return sub_queries

    def parallel_hybrid_search(
        self,
        searcher: HybridSearcher,
        query: str,
        top_k: int = 15,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run hybrid search for all sub-queries sequentially and merge results.

        Args:
            searcher: The initialized HybridSearcher instance.
            query: The compound user query.
            top_k: The hybrid retrieval count per sub-query.
            filters: Optional metadata filters.

        Returns:
            A merged and deduplicated list of candidate chunks.
        """
        sub_queries = self.deconstruct(query)
        
        # Execute sub-queries sequentially to remain 100% thread-safe (prevent HNSW index lock/race conditions)
        merged_candidates: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        for q in sub_queries:
            try:
                candidates = searcher.search(q, top_k, filters)
                for c in candidates:
                    cid = c.get("chunk_id")
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        merged_candidates.append(c)
            except Exception as exc:
                logger.error(f"[query_deconstructor] sub-query '{q}' generated an exception: {exc}")

        # Sort the merged candidates by RRF score to ensure best candidates appear first
        merged_candidates.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        
        logger.info(
            f"[query_deconstructor] merged parallel retrieval results. "
            f"Total unique chunks found: {len(merged_candidates)}"
        )
        return merged_candidates
