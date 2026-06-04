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
        # Removed weak connectors like 'mà', 'thì', 'với', 'khi', 'xong' because they over-split single queries
        self.connectors = connectors or [
            r"\band\b",
            r"\bđồng thời\b",
            r"\bnhưng\b",
            r"\bvà cả\b",
            r"\bvà\b",
            r"\bkèm theo cả\b",
            r"\bkèm theo\b",
            r"\bcộng thêm\b",
            r"\blại còn\b",
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
            
            # --- New Highway & Cargo Keywords ---
            r"đi vào đường cao tốc",
            r"đường cao tốc",
            r"(?<!đường\s)cao tốc",
            r"chở hàng hóa",
            r"chở hàng",
            r"cồng kềnh",
            r"quá tải",
            r"chở quá tải",
            
            # --- Automated Tester Additions ---
            r"không gắn biển",
            r"không có biển số",
            r"đóng bỉm",
            r"giao xe",
            r"cho mượn xe",
            r"đâm công an",
        ]
        lookahead_pattern = r"(?=\b(?:" + "|".join(self.violation_keywords) + r")\b)"
        connectors_pattern = r"(?<!\d)[.;](?!\d)|\b" + r"\b|\b".join(self.connectors) + r"\b"
        
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
            # Keep only if it has a violation keyword, or is a sufficiently long descriptive clause (>= 15 chars)
            has_violation = any(re.search(kw, trimmed, re.IGNORECASE) for kw in self.violation_keywords)
            if trimmed and (has_violation or len(trimmed) >= 15):
                # If a sub-part is missing key subject/context like "xe máy", 
                # prepend the vehicle prefix if found in the main query
                vehicle_prefix = ""
                lower_query = clean_query.lower()
                lower_trimmed = trimmed.lower()
                
                # Check for "xe máy", "mô tô", "xe gắn máy", and implicit motorcycle items with word boundaries
                moto_keywords = ["xe máy", "mô tô", "xe gắn máy", "mũ bảo hiểm", "nón bảo hiểm"]
                moto_pattern = r"\b(?:" + "|".join(re.escape(x) for x in moto_keywords) + r")\b"
                has_moto_main = bool(re.search(moto_pattern, lower_query))
                has_moto_sub = bool(re.search(moto_pattern, lower_trimmed))
                
                # Check for "ô tô" and other auto items with word boundaries
                auto_keywords = ["ô tô", "xe hơi", "xe tải", "xe khách", "xe buýt", "xe con", "xe ben", "xe bốn bánh", "xe 4 bánh"]
                auto_pattern = r"\b(?:" + "|".join(re.escape(x) for x in auto_keywords) + r")\b"
                has_auto_main_explicit = bool(re.search(auto_pattern, lower_query))
                has_auto_sub = bool(re.search(auto_pattern, lower_trimmed))
                
                # Highway inference: "cao tốc" implies ô tô ONLY when the user did NOT
                # explicitly specify another vehicle type (xe máy / mô tô).  If the user
                # already said "xe máy lùi trên cao tốc", we must NOT inject ô tô — the
                # user's explicit vehicle takes precedence.
                cao_toc_implies_auto = ("cao tốc" in lower_query) and not has_moto_main
                has_auto_main = has_auto_main_explicit or cao_toc_implies_auto
                
                # For sub-query level: only infer auto from cao_toc if main didn't have moto
                cao_toc_sub_implies_auto = ("cao tốc" in lower_trimmed) and not has_moto_main
                has_auto_sub = has_auto_sub or cao_toc_sub_implies_auto
                
                # Check for specific non-branching vehicle types with word boundaries
                # These should NEVER trigger ô tô/mô tô branching
                specific_vehicles = ["xe đạp điện", "xe đạp máy", "xe đạp", "đi bộ", "người đi bộ",
                                     "xe thô sơ", "xe lăn", "xe cứu thương", "xe cứu hỏa"]
                specific_pattern = r"\b(?:" + "|".join(re.escape(x) for x in specific_vehicles) + r")\b"
                has_specific_main = bool(re.search(specific_pattern, lower_query))
                has_specific_sub = bool(re.search(specific_pattern, lower_trimmed))
                
                # Normalize existing colloquial terms to formal legal terms to boost RAG scoring
                if re.search(r"\bô tô\b", lower_trimmed) and "xe ô tô" not in lower_trimmed:
                    trimmed = re.sub(r"\bô tô\b", "xe ô tô", trimmed, flags=re.IGNORECASE)
                    lower_trimmed = trimmed.lower()
                    has_auto_sub = True
                
                if any(x in lower_trimmed for x in ["xe máy", "mô tô", "xe gắn máy"]) and "xe mô tô" not in lower_trimmed:
                    trimmed = re.sub(r"\b(xe máy|mô tô|xe gắn máy)\b", "xe mô tô", trimmed, flags=re.IGNORECASE)
                    lower_trimmed = trimmed.lower()
                    has_moto_sub = True

                # If main query has a specific vehicle type, propagate it (don't branch!)
                if has_specific_main:
                    if not has_specific_sub and not has_moto_sub and not has_auto_sub:
                        # Find which specific vehicle to propagate
                        for v in specific_vehicles:
                            if v in lower_query:
                                vehicle_prefix = f"{v} "
                                break
                    sub_queries.append(f"{vehicle_prefix}{trimmed}")
                    continue
                
                if has_moto_sub and has_auto_sub:
                    # Vehicle Branching for explicit compound vehicles
                    sub_queries.append(f"xe ô tô {trimmed}")
                    sub_queries.append(f"xe mô tô {trimmed}")
                    continue

                # Only prepend a vehicle if the sub-query lacks BOTH moto and auto
                if not has_moto_sub and not has_auto_sub:
                    if not has_moto_main and not has_auto_main:
                        # No vehicle in main query either -> do NOT branch, just keep the trimmed part
                        sub_queries.append(trimmed)
                        continue
                    
                    if has_moto_main and has_auto_main:
                        # Main query has both, but this part has neither? Branch it!
                        sub_queries.append(f"xe ô tô {trimmed}")
                        sub_queries.append(f"xe mô tô {trimmed}")
                        continue
                    elif has_moto_main:
                        vehicle_prefix = "xe mô tô "
                    elif has_auto_main:
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
