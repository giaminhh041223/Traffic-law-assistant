"""Heuristic metadata tagger for Vietnamese traffic-law chunks.

Tags each chunk with:
  * vehicle_type   — list of vehicle categories the chunk applies to.
  * violation_type — list of violation categories the chunk addresses.
  * fine_amount    — extracted fine range, if any (min, max VND).
  * sanctions      — list of supplementary sanctions (license suspension, etc.).

This metadata is critical because Phase 2's ChromaDB retriever uses
metadata filters (e.g. "give me Khoản about xe máy + nồng độ cồn"),
which is dramatically faster and more precise than relying on
embedding similarity alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Vehicle taxonomy
# ---------------------------------------------------------------------------
# Each entry: (canonical_label, list_of_surface_patterns).
# Patterns are matched case-insensitively against the chunk text.
_VEHICLES: List[Tuple[str, List[str]]] = [
    ("o_to",          [r"\bxe ô tô\b", r"\bô tô\b"]),
    ("xe_may",        [r"\bxe máy\b", r"\bxe mô tô\b", r"\bmô tô\b", r"\bxe gắn máy\b"]),
    ("may_keo",       [r"\bmáy kéo\b"]),
    ("xe_chuyen_dung",[r"\bxe máy chuyên dùng\b", r"\bxe chuyên dùng\b"]),
    ("xe_dap",        [r"\bxe đạp\b", r"\bxe đạp điện\b", r"\bxe đạp máy\b"]),
    ("xe_tho_so",     [r"\bxe thô sơ\b"]),
    ("nguoi_di_bo",   [r"\bngười đi bộ\b"]),
    ("xe_buyt",       [r"\bxe buýt\b", r"\bxe khách\b"]),
    ("xe_tai",        [r"\bxe tải\b", r"\bxe ô tô tải\b"]),
]

# ---------------------------------------------------------------------------
# Violation taxonomy
# ---------------------------------------------------------------------------
_VIOLATIONS: List[Tuple[str, List[str]]] = [
    ("toc_do",              [r"\btốc độ\b", r"\bquá tốc độ\b", r"\bchạy quá\b.{0,30}km/h"]),
    ("nong_do_con",         [r"\bnồng độ cồn\b", r"\brượu\b.{0,10}bia\b", r"\bcó cồn\b"]),
    ("ma_tuy",              [r"\bma túy\b", r"\bchất ma túy\b"]),
    ("vuot_den_do",         [r"\bvượt đèn đỏ\b", r"\btín hiệu đèn\b", r"\bđèn tín hiệu\b"]),
    ("khong_mu_bao_hiem",   [r"\bmũ bảo hiểm\b", r"\bkhông đội mũ\b"]),
    ("sai_lan",             [r"\bsai làn\b", r"\bkhông đúng làn\b", r"\blàn đường\b"]),
    ("khong_gplx",          [r"\bgiấy phép lái xe\b", r"\bGPLX\b", r"\bkhông có (bằng|giấy)\b"]),
    ("dung_do_xe",          [r"\bdừng xe\b", r"\bđỗ xe\b"]),
    ("lui_xe",              [r"\blùi xe\b"]),
    ("quay_dau",            [r"\bquay đầu\b", r"\bquay đầu xe\b"]),
    ("vuot_xe",             [r"\bvượt xe\b", r"\bvượt không đúng\b"]),
    ("cho_qua_so_nguoi",    [r"\bchở quá số người\b", r"\bquá số người\b"]),
    ("cho_qua_tai",         [r"\bquá tải\b", r"\bquá trọng tải\b"]),
    ("dien_thoai",          [r"\bđiện thoại\b", r"\bdùng tay.{0,20}điện thoại\b"]),
    ("bao_hiem",            [r"\bbảo hiểm trách nhiệm\b", r"\bbảo hiểm bắt buộc\b"]),
    ("dang_kiem",           [r"\bđăng kiểm\b", r"\bgiấy chứng nhận kiểm định\b"]),
    ("bien_so",             [r"\bbiển số\b", r"\bbiển kiểm soát\b"]),
]

# ---------------------------------------------------------------------------
# Fine + sanction extractors
# ---------------------------------------------------------------------------
# "phạt tiền từ 800.000 đồng đến 1.000.000 đồng"
_FINE_RE = re.compile(
    r"phạt\s+tiền\s+từ\s+([\d\.\,]+)\s*đồng\s+đến\s+([\d\.\,]+)\s*đồng",
    re.IGNORECASE | re.UNICODE,
)

_SANCTION_PATTERNS: List[Tuple[str, str]] = [
    ("tuoc_gplx",        r"tước\s+(quyền sử dụng\s+)?(Giấy phép lái xe|GPLX)[^.]{0,80}"),
    ("tam_giu_phuong_tien", r"tạm giữ\s+phương tiện[^.]{0,80}"),
    ("tich_thu",         r"tịch thu[^.]{0,80}"),
    ("buoc_khac_phuc",   r"buộc\s+(khôi phục|tháo dỡ|thực hiện)[^.]{0,80}"),
]


@dataclass
class TaggedMetadata:
    vehicle_type: List[str]
    violation_type: List[str]
    fine_min: Optional[int]
    fine_max: Optional[int]
    sanctions: List[str]


class MetadataTagger:
    """Tag a chunk's free text with controlled-vocabulary labels."""

    @staticmethod
    def tag(text: str) -> TaggedMetadata:
        vehicles = MetadataTagger._match_labels(text, _VEHICLES)
        violations = MetadataTagger._match_labels(text, _VIOLATIONS)
        fine_min, fine_max = MetadataTagger._extract_fine(text)
        sanctions = MetadataTagger._extract_sanctions(text)
        return TaggedMetadata(
            vehicle_type=vehicles,
            violation_type=violations,
            fine_min=fine_min,
            fine_max=fine_max,
            sanctions=sanctions,
        )

    # ---------------- internals ----------------

    @staticmethod
    def _match_labels(text: str, taxonomy: List[Tuple[str, List[str]]]) -> List[str]:
        hits: List[str] = []
        lowered = text.lower()
        for label, patterns in taxonomy:
            for pat in patterns:
                if re.search(pat, lowered, flags=re.IGNORECASE | re.UNICODE):
                    hits.append(label)
                    break
        return hits

    @staticmethod
    def _parse_amount(raw: str) -> Optional[int]:
        digits = re.sub(r"[^\d]", "", raw)
        return int(digits) if digits else None

    @staticmethod
    def _extract_fine(text: str) -> Tuple[Optional[int], Optional[int]]:
        m = _FINE_RE.search(text)
        if not m:
            return None, None
        return MetadataTagger._parse_amount(m.group(1)), MetadataTagger._parse_amount(m.group(2))

    @staticmethod
    def _extract_sanctions(text: str) -> List[str]:
        out: List[str] = []
        for label, pat in _SANCTION_PATTERNS:
            if re.search(pat, text, flags=re.IGNORECASE | re.UNICODE):
                out.append(label)
        return out
