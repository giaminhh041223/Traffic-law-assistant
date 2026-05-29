"""Semantic chunker for Vietnamese legal documents.

Vietnamese legal hierarchy (decree / nghị định):

    Chương  (Chapter)   — e.g. "Chương I"          [Roman numerals, sometimes Arabic]
        Mục  (Section)  — e.g. "Mục 1"             [Arabic, optional]
            Điều (Article) — e.g. "Điều 5. <title>"
                Khoản (Clause) — e.g. "1. <text>"   [Arabic, sequential within Điều]
                    Điểm (Point) — e.g. "a) <text>" [VN letters, sequential within Khoản]

Why a state machine, not pure regex?
    A line `1.` or `2.` looks identical whether it is a real Khoản marker
    or a numbered enumeration inside Điều prose. Sequential validation
    (the next Khoản must be the previous + 1, reset per Điều) eliminates
    nearly all false positives.

    Likewise for Điểm: the next letter must be the next entry in the
    Vietnamese alphabet (which has 'đ' but no 'f', 'j', 'w', 'z').

Output: list of `LegalChunk` objects, one per Khoản (or Điều / Điểm,
        depending on `granularity` setting).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Regex — the heart of the chunker
# ---------------------------------------------------------------------------
# Compiled with re.UNICODE so Vietnamese characters (đ, ê, ô, ơ, ư, diacritics)
# behave correctly. We use re.IGNORECASE to absorb "ĐIỀU 5" (all caps) headings
# that appear in formal legal documents — Python's IGNORECASE handles VN
# case folding because str is Unicode.

# Chương I, Chương II, Chương 1 — Roman or Arabic
_CHUONG_RE = re.compile(
    r"^\s*Chương\s+([IVXLCDM]+|\d+)\b\s*[\.\-:]?\s*(.*)$",
    re.IGNORECASE | re.UNICODE,
)

# Mục 1, Mục 2 — section
_MUC_RE = re.compile(
    r"^\s*Mục\s+(\d+)\s*[\.\-:]?\s*(.*)$",
    re.IGNORECASE | re.UNICODE,
)

# Điều 5, Điều 5., Điều 5: <title>
# Captures (1) the number, (2) the (possibly empty) inline title.
_DIEU_RE = re.compile(
    r"^\s*Điều\s+(\d+)\s*[\.\-:]?\s*(.*)$",
    re.IGNORECASE | re.UNICODE,
)

# Khoản marker at line start: "1. ", "2) ", "10. "
# Note the REQUIRED trailing space — protects against decimals like "1.5".
_KHOAN_RE = re.compile(
    r"^\s*(\d{1,3})\s*[\.\)]\s+(.+)$",
    re.UNICODE,
)

# Điểm marker at line start: "a) ", "b) ", "đ) "
# Vietnamese point alphabet does NOT contain f / j / w / z.
_DIEM_RE = re.compile(
    r"^\s*([a-zđ])\s*\)\s+(.+)$",
    re.IGNORECASE | re.UNICODE,
)

# Inline "Khoản X Điều Y" citation — used only to suppress false-positive
# Khoản matches when the line is clearly a back-reference inside prose.
_INLINE_REF_RE = re.compile(
    r"\bKhoản\s+\d+\s+Điều\s+\d+",
    re.IGNORECASE | re.UNICODE,
)

# Vietnamese alphabet used for Điểm enumeration (in order).
# Source: Bộ luật Ban hành VBQPPL — Vietnamese legal drafting convention.
VN_POINT_ALPHABET: List[str] = [
    "a", "b", "c", "d", "đ", "e", "g", "h", "i", "k",
    "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class LegalChunk:
    """A single retrievable unit of legal text with full hierarchy context."""
    chunk_id: str
    source_doc: str            # "Nghị định 100/2019/NĐ-CP"
    doc_short: str             # "ND100"
    chuong: Optional[str] = None
    chuong_title: Optional[str] = None
    muc: Optional[str] = None
    muc_title: Optional[str] = None
    dieu: Optional[str] = None
    dieu_title: Optional[str] = None
    khoan: Optional[str] = None
    diem: Optional[str] = None
    text: str = ""
    full_citation: str = ""
    chunk_type: str = "khoan"  # dieu | khoan | diem | lead_in | table
    char_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
class SemanticChunker:
    """Stateful Vietnamese legal-hierarchy chunker."""

    def __init__(
        self,
        source_doc: str,
        doc_short: str,
        granularity: str = "khoan",
        max_chunk_chars: int = 1800,
        emit_dieu_when_no_khoan: bool = True,
        emit_structural_headings: bool = False,
    ):
        if granularity not in {"dieu", "khoan", "diem"}:
            raise ValueError("granularity must be one of: dieu, khoan, diem")
        self.source_doc = source_doc
        self.doc_short = doc_short
        self.granularity = granularity
        self.max_chunk_chars = max_chunk_chars
        self.emit_dieu_when_no_khoan = emit_dieu_when_no_khoan
        self.emit_structural_headings = emit_structural_headings

        # current hierarchy state
        self._chuong: Optional[str] = None
        self._chuong_title: Optional[str] = None
        self._muc: Optional[str] = None
        self._muc_title: Optional[str] = None
        self._dieu: Optional[str] = None
        self._dieu_title: Optional[str] = None
        self._khoan: Optional[str] = None
        self._diem: Optional[str] = None

        # sequential counters used to reject false-positive markers
        self._expected_next_khoan: int = 1
        self._expected_next_diem_idx: int = 0

        # buffer of text lines for the currently open chunk
        self._buffer: List[str] = []
        # whether we are inside an Điều header before the first Khoản
        self._dieu_has_emitted = False
        # output
        self._chunks: List[LegalChunk] = []

    # ----------------------------- public API -----------------------------

    def parse(self, text: str) -> List[LegalChunk]:
        """Walk `text` line by line, returning the list of chunks."""
        text = self._normalize(text)
        for raw_line in text.split("\n"):
            self._process_line(raw_line)
        self._flush()  # final flush
        return self._chunks

    # ----------------------------- normalisation -----------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """NFC-normalize Unicode + collapse whitespace within lines, preserve newlines."""
        text = unicodedata.normalize("NFC", text)
        # Replace common invisible chars with regular space.
        text = text.replace(" ", " ").replace("​", "")
        # Trim trailing whitespace per line, but keep line boundaries.
        text = "\n".join(re.sub(r"[ \t]+", " ", ln).rstrip() for ln in text.splitlines())
        return text

    # ----------------------------- per-line walker -----------------------------

    def _process_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            # Blank line — treat as a soft paragraph break inside the current chunk.
            if self._buffer and self._buffer[-1] != "":
                self._buffer.append("")
            return

        # --- 1. Chương ---
        m = _CHUONG_RE.match(stripped)
        if m and self._looks_like_heading(stripped, max_words=12):
            self._flush()
            self._chuong = m.group(1).strip()
            self._chuong_title = m.group(2).strip() or None
            # reset everything beneath chương
            self._reset_below("chuong")
            if self.emit_structural_headings:
                self._emit_heading_chunk("chuong")
            return

        # --- 2. Mục ---
        m = _MUC_RE.match(stripped)
        if m and self._looks_like_heading(stripped, max_words=14):
            self._flush()
            self._muc = m.group(1).strip()
            self._muc_title = m.group(2).strip() or None
            self._reset_below("muc")
            if self.emit_structural_headings:
                self._emit_heading_chunk("muc")
            return

        # --- 3. Điều ---
        m = _DIEU_RE.match(stripped)
        if m:
            self._flush()
            self._dieu = m.group(1).strip()
            self._dieu_title = (m.group(2).strip() or None)
            self._reset_below("dieu")
            self._dieu_has_emitted = False
            return

        # --- 4. Khoản ---  (validated by sequential numbering)
        m = _KHOAN_RE.match(stripped)
        if m and self._dieu is not None:
            num = int(m.group(1))
            # Reject if not the expected next Khoản — this is the most important guard.
            if num == self._expected_next_khoan and not _INLINE_REF_RE.search(stripped):
                if self.granularity == "dieu":
                    # Don't flush — fold Khoản content into the parent Điều chunk,
                    # but keep the "1. " marker visible in the text for citation lookup.
                    self._buffer.append(f"{num}. {m.group(2).strip()}")
                else:
                    self._flush()
                    self._khoan = str(num)
                    self._diem = None
                    self._buffer.append(m.group(2).strip())
                self._expected_next_khoan = num + 1
                self._expected_next_diem_idx = 0
                return

        # --- 5. Điểm --- (validated by VN alphabet ordering)
        m = _DIEM_RE.match(stripped)
        if m and self._khoan is not None:
            letter = m.group(1).lower()
            if self._is_expected_diem(letter):
                # In khoan-granularity, Điểm content becomes part of the parent Khoản chunk.
                if self.granularity == "diem":
                    self._flush()
                    self._diem = letter
                else:
                    # Just record we're inside Điểm `letter` and keep accumulating into the Khoản buffer.
                    self._diem = letter
                self._expected_next_diem_idx += 1
                # Prefix the Điểm line so the marker stays in the text for citation lookup.
                self._buffer.append(f"{letter}) {m.group(2).strip()}")
                return

        # --- 6. fallthrough: continuation text ---
        # If we haven't opened any Điều yet, this is preamble — drop it unless asked to keep.
        if self._dieu is None:
            return
        # If we have an Điều but no Khoản yet AND dieu_title is empty, treat this as
        # continuation of the title (Điều header spans multiple lines).
        if self._khoan is None and (self._dieu_title is None or len(self._dieu_title) < 5):
            self._dieu_title = (self._dieu_title + " " + stripped).strip() if self._dieu_title else stripped
            return
        self._buffer.append(stripped)

    # ----------------------------- helpers -----------------------------

    @staticmethod
    def _looks_like_heading(line: str, max_words: int) -> bool:
        """Heading lines are short and don't end with a period followed by lowercase prose."""
        words = line.split()
        if len(words) > max_words:
            return False
        return True

    def _is_expected_diem(self, letter: str) -> bool:
        if self._expected_next_diem_idx >= len(VN_POINT_ALPHABET):
            return False
        return letter == VN_POINT_ALPHABET[self._expected_next_diem_idx]

    def _reset_below(self, level: str) -> None:
        """Reset state strictly below `level`."""
        order = ["chuong", "muc", "dieu", "khoan", "diem"]
        idx = order.index(level)
        below = order[idx + 1:]
        for lvl in below:
            setattr(self, f"_{lvl}", None)
        if "khoan" in below:
            self._expected_next_khoan = 1
        if "diem" in below:
            self._expected_next_diem_idx = 0
            if "khoan" not in below:
                # New Khoản → reset Điểm counter
                self._expected_next_diem_idx = 0

    # ----------------------------- flushing -----------------------------

    def _flush(self) -> None:
        """Emit the currently buffered text (if any) as a chunk."""
        text = self._buffer_text()
        if not text and self._khoan is None:
            # Nothing to emit (e.g. blank Điều header before first Khoản).
            self._buffer.clear()
            return

        # Decide chunk type.
        if self._khoan is not None:
            chunk_type = "khoan"
        elif self._dieu is not None:
            if not self.emit_dieu_when_no_khoan:
                self._buffer.clear()
                return
            chunk_type = "dieu"
        else:
            self._buffer.clear()
            return

        chunk = self._build_chunk(text, chunk_type)
        # If the chunk exceeds max_chunk_chars and we are at Khoản granularity,
        # split by Điểm boundaries inside the text.
        if (
            self.granularity == "khoan"
            and chunk.char_count > self.max_chunk_chars
            and "\n" in text
        ):
            for sub in self._split_oversized_khoan(chunk):
                self._chunks.append(sub)
        else:
            self._chunks.append(chunk)

        self._buffer.clear()
        self._dieu_has_emitted = True

    def _buffer_text(self) -> str:
        text = "\n".join(self._buffer).strip()
        return re.sub(r"\n{3,}", "\n\n", text)

    def _build_chunk(self, text: str, chunk_type: str) -> LegalChunk:
        citation = self._format_citation()
        cid = self._make_chunk_id(chunk_type)
        return LegalChunk(
            chunk_id=cid,
            source_doc=self.source_doc,
            doc_short=self.doc_short,
            chuong=self._chuong,
            chuong_title=self._chuong_title,
            muc=self._muc,
            muc_title=self._muc_title,
            dieu=self._dieu,
            dieu_title=self._dieu_title,
            khoan=self._khoan if chunk_type != "dieu" else None,
            diem=self._diem if chunk_type == "diem" else None,
            text=text,
            full_citation=citation,
            chunk_type=chunk_type,
            char_count=len(text),
        )

    def _format_citation(self) -> str:
        parts: List[str] = []
        if self._diem and self.granularity == "diem":
            parts.append(f"Điểm {self._diem}")
        if self._khoan:
            parts.append(f"Khoản {self._khoan}")
        if self._dieu:
            parts.append(f"Điều {self._dieu}")
        parts.append(self.source_doc)
        return " ".join(parts)

    def _make_chunk_id(self, chunk_type: str) -> str:
        bits = [self.doc_short]
        if self._dieu:
            bits.append(f"Dieu_{self._dieu}")
        if self._khoan and chunk_type != "dieu":
            bits.append(f"Khoan_{self._khoan}")
        if self._diem and chunk_type == "diem":
            bits.append(f"Diem_{self._diem}")
        return "__".join(bits)

    def _split_oversized_khoan(self, chunk: LegalChunk) -> List[LegalChunk]:
        """Split a too-large Khoản chunk into Điểm-level sub-chunks."""
        sub_chunks: List[LegalChunk] = []
        # Split on Điểm markers we kept in the text: "a) ...", "b) ..."
        # Use a lookahead so the marker stays in each segment.
        segments = re.split(r"\n(?=[a-zđ]\)\s)", chunk.text, flags=re.IGNORECASE)
        # First segment is the Khoản lead-in (before any Điểm).
        lead = segments[0].strip()
        if lead:
            lead_chunk = LegalChunk(
                chunk_id=f"{chunk.chunk_id}__lead",
                source_doc=chunk.source_doc,
                doc_short=chunk.doc_short,
                chuong=chunk.chuong,
                chuong_title=chunk.chuong_title,
                muc=chunk.muc,
                muc_title=chunk.muc_title,
                dieu=chunk.dieu,
                dieu_title=chunk.dieu_title,
                khoan=chunk.khoan,
                diem=None,
                text=lead,
                full_citation=chunk.full_citation,
                chunk_type="khoan",
                char_count=len(lead),
            )
            sub_chunks.append(lead_chunk)
        for seg in segments[1:]:
            seg = seg.strip()
            if not seg:
                continue
            m = re.match(r"^([a-zđ])\)\s*(.*)$", seg, re.IGNORECASE | re.DOTALL)
            if not m:
                continue
            letter = m.group(1).lower()
            body = m.group(2).strip()
            # Build citation defensively: technical regulations (e.g. QCVN with
            # 6.1.2. style numbering) reach this branch with khoan=None — render
            # "Điểm a Điều X" instead of the nonsensical "Điểm a Khoản None Điều X".
            cit_parts = [f"Điểm {letter}"]
            if chunk.khoan:
                cit_parts.append(f"Khoản {chunk.khoan}")
            if chunk.dieu:
                cit_parts.append(f"Điều {chunk.dieu}")
            cit_parts.append(chunk.source_doc)
            cit = " ".join(cit_parts)
            enriched_text = f"{lead}\n{letter}) {body}"
            sub_chunks.append(LegalChunk(
                chunk_id=f"{chunk.chunk_id}__Diem_{letter}",
                source_doc=chunk.source_doc,
                doc_short=chunk.doc_short,
                chuong=chunk.chuong,
                chuong_title=chunk.chuong_title,
                muc=chunk.muc,
                muc_title=chunk.muc_title,
                dieu=chunk.dieu,
                dieu_title=chunk.dieu_title,
                khoan=chunk.khoan,
                diem=letter,
                text=enriched_text,
                full_citation=cit,
                chunk_type="diem",
                char_count=len(enriched_text),
            ))
        return sub_chunks or [chunk]

    def _emit_heading_chunk(self, level: str) -> None:
        """Optionally emit Chương / Mục as standalone navigational chunks."""
        if level == "chuong":
            text = f"Chương {self._chuong}" + (f". {self._chuong_title}" if self._chuong_title else "")
        else:
            text = f"Mục {self._muc}" + (f". {self._muc_title}" if self._muc_title else "")
        self._chunks.append(LegalChunk(
            chunk_id=f"{self.doc_short}__{level}_{getattr(self, '_' + level)}",
            source_doc=self.source_doc,
            doc_short=self.doc_short,
            chuong=self._chuong,
            chuong_title=self._chuong_title,
            muc=self._muc if level != "chuong" else None,
            muc_title=self._muc_title if level != "chuong" else None,
            text=text,
            full_citation=text + " " + self.source_doc,
            chunk_type=level,
            char_count=len(text),
        ))


# ---------------------------------------------------------------------------
# Convenience helper used by run_ingest.py
# ---------------------------------------------------------------------------
def chunk_document(
    text: str,
    source_doc: str,
    doc_short: str,
    **kwargs: Any,
) -> List[LegalChunk]:
    return SemanticChunker(source_doc=source_doc, doc_short=doc_short, **kwargs).parse(text)


def stable_hash(text: str) -> str:
    """Short stable hash — useful for table chunk IDs."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
