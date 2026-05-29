"""PDF → raw Vietnamese text via PyMuPDF (fitz).

Why PyMuPDF for text:
  * Fast, dependency-light, preserves reading order for narrative columns
    (Vietnamese decrees are single-column, which it handles well).
  * Keeps line breaks — important because our `SemanticChunker`
    relies on line-start anchors (^) for Điều / Khoản / Điểm markers.

Tables are extracted separately by `TableExtractor` (pdfplumber).
This parser RETURNS plain text with newlines intact and per-page boundaries
marked, so downstream code can route table regions to the correct article.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF


@dataclass
class ParsedPage:
    page_num: int           # 1-indexed
    text: str
    bbox: Tuple[float, float, float, float] = field(default=(0, 0, 0, 0))


@dataclass
class ParsedDocument:
    path: str
    pages: List[ParsedPage]

    @property
    def full_text(self) -> str:
        """All pages concatenated with double newline separators (preserves chunker anchors)."""
        return "\n\n".join(p.text for p in self.pages)


class PDFParser:
    """Lightweight PyMuPDF wrapper."""

    # Headers / footers we strip — they break Điều/Khoản anchors when they
    # land mid-page in extraction order.
    _BOILERPLATE_PATTERNS = [
        re.compile(r"^\s*\d+\s*$"),                               # bare page number
        re.compile(r"^\s*CÔNG BÁO.*$", re.IGNORECASE),
        re.compile(r"^\s*Trang\s*\d+(\s*/\s*\d+)?\s*$", re.IGNORECASE),
    ]

    def __init__(self, strip_boilerplate: bool = True):
        self.strip_boilerplate = strip_boilerplate

    # ---------------- public API ----------------

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        pages: List[ParsedPage] = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc, start=1):
                # `sort=True` enforces reading order based on geometry.
                raw = page.get_text("text", sort=True)
                cleaned = self._clean_page(raw)
                pages.append(ParsedPage(page_num=i, text=cleaned, bbox=tuple(page.rect)))
        return ParsedDocument(path=str(pdf_path), pages=pages)

    # ---------------- cleaning ----------------

    def _clean_page(self, text: str) -> str:
        # Fix common PDF extraction artefacts:
        #  - soft hyphen at line end:  "phương­\ntiện" -> "phươngtiện"
        #  - mid-word line break in justified text: "phương-\ntiện" -> "phươngtiện"
        text = re.sub(r"­\n", "", text)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # Normalize multiple newlines (keep at most one blank line — paragraph separator).
        text = re.sub(r"\n{3,}", "\n\n", text)
        if self.strip_boilerplate:
            kept = []
            for ln in text.splitlines():
                if any(p.match(ln) for p in self._BOILERPLATE_PATTERNS):
                    continue
                kept.append(ln)
            text = "\n".join(kept)
        return text.strip()
