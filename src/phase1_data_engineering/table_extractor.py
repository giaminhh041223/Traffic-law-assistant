"""Penalty-table extraction via pdfplumber.

Vietnamese traffic-law decrees contain dense penalty tables — e.g. fine
ranges by vehicle type + violation. We must NOT let these get mangled by
PyMuPDF's text extraction (which reads them row-by-row, losing cell
boundaries and conflating columns).

Strategy:
  1. For every page, ask pdfplumber to detect tables.
  2. Convert each table (List[List[str]]) into a Markdown table — the LLM
     reads Markdown tables natively and citation/answer quality improves.
  3. Each table becomes its own `LegalChunk` with `chunk_type="table"`,
     anchored to the nearest preceding Điều/Khoản (the chunker walks
     page-aware text, so we tag tables with a page number for fuzzy join
     in `run_ingest.py`).

Linearization fallback:
  If Markdown is unsuitable (e.g. heavily merged cells), we also export
  a "linear" form — one sentence per row — like:
     "Đối với xe ô tô, vi phạm tốc độ 5-10 km/h: phạt 800.000 - 1.000.000 đồng."
  This is stored alongside the markdown so we can choose at retrieval time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber


@dataclass
class ExtractedTable:
    table_id: str                   # e.g. "ND100__page12__t1"
    page_num: int
    rows: List[List[str]]
    markdown: str
    linear: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TableExtractor:
    """pdfplumber-based table extractor with Markdown + linear serialization."""

    # pdfplumber settings tuned for legal-document tables (clean black borders, dense text).
    _TABLE_SETTINGS = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_y_tolerance": 6,
        "intersection_x_tolerance": 6,
    }

    def __init__(self, doc_short: str):
        self.doc_short = doc_short

    # ---------------- public API ----------------

    def extract(self, pdf_path: str | Path) -> List[ExtractedTable]:
        pdf_path = Path(pdf_path)
        tables: List[ExtractedTable] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                page_tables = page.find_tables(table_settings=self._TABLE_SETTINGS)
                for t_idx, t in enumerate(page_tables, start=1):
                    rows = self._clean_rows(t.extract())
                    if not rows or not self._is_meaningful(rows):
                        continue
                    md = self._to_markdown(rows)
                    lin = self._to_linear(rows)
                    tables.append(ExtractedTable(
                        table_id=f"{self.doc_short}__page{page_idx}__t{t_idx}",
                        page_num=page_idx,
                        rows=rows,
                        markdown=md,
                        linear=lin,
                    ))
        return tables

    # ---------------- internals ----------------

    @staticmethod
    def _clean_rows(rows: List[List[Optional[str]]]) -> List[List[str]]:
        out: List[List[str]] = []
        for row in rows:
            cleaned = []
            for cell in row:
                if cell is None:
                    cleaned.append("")
                else:
                    # Collapse newlines inside cells, trim whitespace.
                    c = re.sub(r"\s+", " ", cell).strip()
                    cleaned.append(c)
            # Skip empty rows.
            if any(c for c in cleaned):
                out.append(cleaned)
        return out

    @staticmethod
    def _is_meaningful(rows: List[List[str]]) -> bool:
        """Reject 1x1 or all-empty pseudo-tables created by stray ruler lines."""
        if len(rows) < 2:
            return False
        if all(len(r) <= 1 for r in rows):
            return False
        total_chars = sum(len(c) for row in rows for c in row)
        return total_chars >= 20

    @staticmethod
    def _to_markdown(rows: List[List[str]]) -> str:
        if not rows:
            return ""
        n_cols = max(len(r) for r in rows)
        # Pad each row to n_cols.
        padded = [r + [""] * (n_cols - len(r)) for r in rows]
        header = padded[0]
        body = padded[1:]
        # Escape pipe chars inside cells.
        def esc(s: str) -> str:
            return s.replace("|", "\\|")
        md_lines = [
            "| " + " | ".join(esc(c) for c in header) + " |",
            "| " + " | ".join(["---"] * n_cols) + " |",
        ]
        for row in body:
            md_lines.append("| " + " | ".join(esc(c) for c in row) + " |")
        return "\n".join(md_lines)

    @staticmethod
    def _to_linear(rows: List[List[str]]) -> str:
        """Produce a row-by-row natural-language linearization.

        For a table like:
            | Hành vi vi phạm | Mức phạt | Hình thức bổ sung |
            | Vượt đèn đỏ     | 4-6 triệu | Tước GPLX 1-3 tháng |
        Returns:
            "Hành vi vi phạm: Vượt đèn đỏ. Mức phạt: 4-6 triệu.
             Hình thức bổ sung: Tước GPLX 1-3 tháng."
        """
        if len(rows) < 2:
            return " ".join(" ".join(r) for r in rows)
        header, *body = rows
        sentences: List[str] = []
        for row in body:
            parts = []
            for col_idx, cell in enumerate(row):
                if not cell:
                    continue
                col_name = header[col_idx] if col_idx < len(header) else f"Cột {col_idx + 1}"
                parts.append(f"{col_name}: {cell}")
            if parts:
                sentences.append(". ".join(parts) + ".")
        return "\n".join(sentences)
