"""TrafficQueryRewriter — translates colloquial/slang queries into formal legal terminology.

This module bridges the gap between everyday Vietnamese traffic slang and the rigid
legal vocabulary used in Nghị định 100/123/NĐ-CP and QCVN 41:2019/BGTVT.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from loguru import logger


class TrafficQueryRewriter:
    def __init__(self, synonym_dict: Optional[Dict[str, str]] = None):
        # Canonical Vietnamese traffic law synonyms
        # Note: All keys are stored in lowercase, plain string format (no regex anchors needed here).
        self.synonyms = synonym_dict or {
            "không đội mũ bảo hiểm": "không đội mũ bảo hiểm cho người đi mô tô, xe máy",
            "không đội nón bảo hiểm": "không đội mũ bảo hiểm cho người đi mô tô, xe máy",
            "không vượt đèn đỏ": "chấp hành đúng hiệu lệnh của đèn tín hiệu giao thông",
            "vượt đèn đỏ": "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
            "đèn đỏ": "hiệu lệnh của đèn tín hiệu giao thông",
            "xe gắn máy": "xe mô tô",
            "xe máy điện": "xe máy điện",
            "xe máy": "xe mô tô",
            "xe cúp": "xe mô tô",
            "xe hai bánh": "xe mô tô",
            "không mang bằng lái": "không mang theo Giấy phép lái xe",
            "không mang bằng": "không mang theo Giấy phép lái xe",
            "không có bằng lái": "không có Giấy phép lái xe",
            "không có bằng": "không có Giấy phép lái xe",
            "bằng lái xe": "Giấy phép lái xe",
            "bằng lái": "Giấy phép lái xe",
            "gplx": "Giấy phép lái xe",
            "ngược chiều": "đi ngược chiều của đường một chiều, đi ngược chiều trên đường có biển cấm đi ngược chiều",
            "đâm cán bộ": "chống người thi hành công vụ, cản trở người thi hành công vụ",
            "đấm công an": "chống người thi hành công vụ, cản trở người thi hành công vụ",
            "không đội mũ": "không đội mũ bảo hiểm cho người đi mô tô, xe máy",
            "không đội nón": "không đội mũ bảo hiểm cho người đi mô tô, xe máy",
            "mũ bảo hiểm": "mũ bảo hiểm cho người đi mô tô, xe máy",
            "nón bảo hiểm": "mũ bảo hiểm cho người đi mô tô, xe máy",
            "quá tốc độ": "điều khiển xe chạy quá tốc độ quy định",
            "bắn tốc độ": "điều khiển xe chạy quá tốc độ quy định",
            "chạy nhanh": "điều khiển xe chạy quá tốc độ quy định",
            "uống rượu": "điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn",
            "say rượu": "điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn",
            "nồng độ cồn": "điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn",
            "nhậu": "điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn",
            "điện thoại": "sử dụng điện thoại di động khi đang điều khiển xe",
            "nghe điện thoại": "sử dụng điện thoại di động khi đang điều khiển xe",
            "kéo đẩy": "bám, kéo, đẩy xe khác, vật khác, mang vác vật cồng kềnh",
            "chở ba": "chở theo từ 03 người trở lên trên xe mô tô, xe gắn máy",
            "kẹp ba": "chở theo từ 03 người trở lên trên xe mô tô, xe gắn máy",
            "chất kích thích": "chất ma túy, nồng độ cồn",
            "gây tai nạn": "gây tai nạn giao thông",
            "chở đi bệnh viện": "chở người bệnh đi cấp cứu",
            "đi bệnh viện": "chở người bệnh đi cấp cứu",
            "chở đi cấp cứu": "chở người bệnh đi cấp cứu",
            "đi cấp cứu": "chở người bệnh đi cấp cứu",
            
            # --- Slang / Colloquial Upgrades ---
            "xe cọp": "xe mô tô độ, xe mô tô thay đổi kết cấu",
            "xe độ": "xe mô tô độ, xe mô tô thay đổi kết cấu",
            "bồ câu": "Cảnh sát giao thông",
            "thông chốt": "không chấp hành hiệu lệnh, chỉ dẫn của người điều khiển giao thông hoặc người kiểm soát giao thông",
            "nẹt pô": "rú ga liên tục, nẹt pô, rú còi",
            "nẹt bô": "rú ga liên tục, nẹt pô, rú còi",
            "tờ sớ": "Giấy phép lái xe",
            "vài chén rượu": "điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn",
            "làm vài chén rượu": "điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn",
            "làm vài chén": "điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn",
            "chén rượu": "điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn",
            "wave tàu": "xe mô tô",
            "con wave": "xe mô tô",
            "giam xe": "tạm giữ phương tiện",
            
            # --- Automated Tester Additions ---
            "cớm": "Cảnh sát giao thông",
            "đóng bỉm": "không gắn biển số / không có biển số",
            "lượn lách": "lạng lách, đánh võng",
            "ăn bánh mì": "đút lót, hối lộ",
        }
        
        # Sort keys from longest to shortest to ensure that longer phrases match first
        sorted_keys = sorted(self.synonyms.keys(), key=len, reverse=True)
        
        # Compile a single unified regex pattern matching any of the keys on word boundaries
        self.pattern = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b",
            re.IGNORECASE
        )

    def rewrite(self, query: str) -> str:
        """Apply single-pass rule-based synonym mapping to standardize the query.

        This eliminates cascade/recursive matches since replaced text is never re-scanned.

        Args:
            query: The raw Vietnamese query from the user.

        Returns:
            The standardized query containing formal legal terminology.
        """
        if not query:
            return ""

        original_query = query.strip()

        def replace_fn(match):
            word = match.group(1)
            lower_word = word.lower()
            return self.synonyms.get(lower_word, word)

        rewritten_query = self.pattern.sub(replace_fn, original_query)

        if rewritten_query != original_query:
            logger.info(f"[query_rewriter] mapped query: '{original_query}' -> '{rewritten_query}'")
        else:
            logger.debug(f"[query_rewriter] no mapping applied for query: '{original_query}'")

        return rewritten_query
