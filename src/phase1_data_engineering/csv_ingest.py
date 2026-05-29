"""CSV ingest path — for the pre-chunked Luật TTATGT 2024 dataset.

Why a separate module?
    The structured CSV (`metadata_luat.csv`) is already split row-by-row
    into Điều (article) units, but the `content` column is one long
    string where Khoản (1. 2. 3.) and Điểm (a) b) c)) markers run inline
    without line breaks. Our `SemanticChunker` is a line-based state
    machine, so we preprocess: insert `\\n` before every Khoản / Điểm
    marker, reconstruct a synthetic legal-format document, then route it
    through the SAME chunker that processes the PDFs.

    Doing it this way:
      * keeps `chunk_id`, `full_citation`, and metadata schema identical
        across PDF- and CSV-sourced chunks → downstream BM25 / dense
        indexes treat them uniformly;
      * adds explicit `chapter_number` / `chapter_title` / `article_number`
        / `article_title` to the chunk metadata for downstream filtering.

Schema expected in the CSV:
    chapter_number  — e.g. "Chương I"
    chapter_title   — e.g. "NHỮNG QUY ĐỊNH CHUNG"
    article_number  — e.g. 5
    article_title   — e.g. "Tuyên truyền, phổ biến pháp luật ..."
    content         — body of the article, with inline "1. … 2. … a) … b) …"
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Union

import pandas as pd
from loguru import logger

from .semantic_chunker import LegalChunk, SemanticChunker

# Inline Khoản marker:   "...end. 2. Next..." → newline before "2. ".
# Lookbehind requires a non-space char so we don't double-split at line starts.
_KHOAN_INLINE = re.compile(r"(?<=\S)\s+(\d{1,3})\.\s+")

# Inline Điểm marker — only Vietnamese-legal alphabet letters.
# Lowercase only; uppercase point markers don't occur in Vietnamese decrees.
_DIEM_INLINE = re.compile(r"(?<=\S)\s+([abcdđeghiklmnopqrstuv])\)\s+")


def _normalize_row_content(text: str) -> str:
    """Insert newlines before Khoản / Điểm markers so the chunker can detect them."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").strip()
    text = _KHOAN_INLINE.sub(r"\n\1. ", text)
    text = _DIEM_INLINE.sub(r"\n\1) ", text)
    return text


def build_synthetic_document(csv_path: Path) -> str:
    """Reconstruct a synthetic Vietnamese-legal-format document from the CSV.

    Output mimics how a decree reads top-to-bottom so the line-based
    `SemanticChunker` can walk it exactly as it walks PDF text:

        Chương I. NHỮNG QUY ĐỊNH CHUNG
        Điều 1. <title>
        <body lines …>
        Điều 2. <title>
        …
        Chương II. <title>
        Điều N. <title>
        …
    """
    df = pd.read_csv(csv_path)
    required = {"chapter_number", "chapter_title", "article_number", "article_title", "content"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    parts: List[str] = []
    current_chapter: str = ""
    for _, row in df.iterrows():
        chap = str(row["chapter_number"]).strip()
        chap_title = str(row["chapter_title"]).strip()
        if chap and chap != current_chapter:
            # Heading line — the chunker's _CHUONG_RE will pick this up.
            parts.append(f"{chap}. {chap_title}")
            current_chapter = chap

        art_num = str(row["article_number"]).strip()
        art_title = str(row["article_title"]).strip()
        content = _normalize_row_content(str(row["content"]))

        # Heading line — _DIEU_RE picks this up.
        parts.append(f"Điều {art_num}. {art_title}")
        if content:
            parts.append(content)
        parts.append("")  # blank line between articles

    return "\n".join(parts)


def ingest_csv(
    csv_path: Union[str, Path],
    source_doc: str,
    doc_short: str,
    granularity: str = "khoan",
    max_chunk_chars: int = 1800,
    emit_dieu_when_no_khoan: bool = True,
    emit_structural_headings: bool = False,
) -> List[LegalChunk]:
    """End-to-end: CSV → list of `LegalChunk` objects ready to be JSONL-dumped."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV source not found: {csv_path}")
    logger.info(f"Parsing CSV: {csv_path.name}")
    df_preview = pd.read_csv(csv_path)
    logger.info(f"  → {len(df_preview)} rows × {len(df_preview.columns)} columns")

    text = build_synthetic_document(csv_path)
    logger.info(f"  → synthetic document: {len(text):,} chars")

    chunker = SemanticChunker(
        source_doc=source_doc,
        doc_short=doc_short,
        granularity=granularity,
        max_chunk_chars=max_chunk_chars,
        emit_dieu_when_no_khoan=emit_dieu_when_no_khoan,
        emit_structural_headings=emit_structural_headings,
    )
    chunks = chunker.parse(text)
    logger.info(f"  → {len(chunks)} semantic chunks")
    return chunks
