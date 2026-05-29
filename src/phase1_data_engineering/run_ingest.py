"""Phase 1 CLI: ingest all configured PDFs → chunks.jsonl + tables.jsonl.

Usage:
    python -m src.phase1_data_engineering.run_ingest \
        --config configs/settings.yaml \
        [--doc ND_100_2019]           # optional: ingest a single document
        [--granularity khoan]          # override config
        [--dry-run]                    # print stats, don't write files
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from loguru import logger

from src.utils.io import load_yaml, write_jsonl
from src.utils.logging import setup_logger

from .csv_ingest import ingest_csv
from .metadata_tagger import MetadataTagger
from .pdf_parser import PDFParser
from .semantic_chunker import LegalChunk, SemanticChunker, stable_hash
from .table_extractor import ExtractedTable, TableExtractor


def attach_metadata(chunk: LegalChunk) -> Dict:
    """Run the metadata tagger and merge results into the chunk's `metadata` dict."""
    text_to_tag = f"{chunk.dieu_title or ''}\n{chunk.text}"
    tagged = MetadataTagger.tag(text_to_tag)
    chunk.metadata.update({
        "vehicle_type": tagged.vehicle_type,
        "violation_type": tagged.violation_type,
        "fine_min": tagged.fine_min,
        "fine_max": tagged.fine_max,
        "sanctions": tagged.sanctions,
    })
    return asdict(chunk)


def table_to_chunk(table: ExtractedTable, source_doc: str, doc_short: str) -> Dict:
    """Wrap an extracted table as a JSONL-friendly chunk record."""
    text = table.markdown if table.markdown else table.linear
    tagged = MetadataTagger.tag(table.linear or " ".join(" ".join(r) for r in table.rows))
    return {
        "chunk_id": table.table_id,
        "source_doc": source_doc,
        "doc_short": doc_short,
        "chunk_type": "table",
        "page_num": table.page_num,
        "text": text,
        "linear_form": table.linear,
        "markdown": table.markdown,
        "rows": table.rows,
        "full_citation": f"Bảng tại trang {table.page_num} - {source_doc}",
        "char_count": len(text),
        "metadata": {
            "vehicle_type": tagged.vehicle_type,
            "violation_type": tagged.violation_type,
            "fine_min": tagged.fine_min,
            "fine_max": tagged.fine_max,
            "sanctions": tagged.sanctions,
            "row_count": len(table.rows),
            "hash": stable_hash(text),
        },
    }


def ingest_document(
    pdf_path: Path,
    source_doc: str,
    doc_short: str,
    granularity: str,
    max_chunk_chars: int,
    emit_dieu_when_no_khoan: bool,
    emit_structural_headings: bool,
) -> tuple[List[Dict], List[Dict]]:
    logger.info(f"Parsing PDF: {pdf_path.name}")
    parsed = PDFParser().parse(pdf_path)
    logger.info(f"  → {len(parsed.pages)} pages, {sum(len(p.text) for p in parsed.pages):,} chars")

    chunker = SemanticChunker(
        source_doc=source_doc,
        doc_short=doc_short,
        granularity=granularity,
        max_chunk_chars=max_chunk_chars,
        emit_dieu_when_no_khoan=emit_dieu_when_no_khoan,
        emit_structural_headings=emit_structural_headings,
    )
    chunks: List[LegalChunk] = chunker.parse(parsed.full_text)
    logger.info(f"  → {len(chunks)} semantic chunks")

    chunk_records = [attach_metadata(c) for c in chunks]

    logger.info(f"Extracting tables: {pdf_path.name}")
    tables = TableExtractor(doc_short=doc_short).extract(pdf_path)
    logger.info(f"  → {len(tables)} tables")
    table_records = [table_to_chunk(t, source_doc, doc_short) for t in tables]

    return chunk_records, table_records


def ingest_csv_source(
    csv_path: Path,
    source_doc: str,
    doc_short: str,
    granularity: str,
    max_chunk_chars: int,
    emit_dieu_when_no_khoan: bool,
    emit_structural_headings: bool,
) -> List[Dict]:
    """Process a structured CSV source (e.g. metadata_luat.csv) into chunk records."""
    chunks = ingest_csv(
        csv_path=csv_path,
        source_doc=source_doc,
        doc_short=doc_short,
        granularity=granularity,
        max_chunk_chars=max_chunk_chars,
        emit_dieu_when_no_khoan=emit_dieu_when_no_khoan,
        emit_structural_headings=emit_structural_headings,
    )
    return [attach_metadata(c) for c in chunks]


def main():
    parser = argparse.ArgumentParser(description="Phase 1 ingestion CLI")
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--doc", default=None, help="Ingest a single registered doc (key in settings.documents or settings.csv_sources)")
    parser.add_argument("--granularity", default=None, choices=["dieu", "khoan", "diem"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    setup_logger(cfg.get("paths", {}).get("log_dir", "logs"), cfg.get("logging", {}).get("level", "INFO"))

    raw_dir = Path(cfg["paths"]["raw_dir"])
    chunks_out = Path(cfg["paths"]["chunks_file"])
    tables_out = Path(cfg["paths"]["tables_file"])

    chunking_cfg = cfg.get("chunking", {})
    granularity = args.granularity or chunking_cfg.get("granularity", "khoan")
    max_chars = chunking_cfg.get("max_chunk_chars", 1800)
    emit_dieu_when_no_khoan = chunking_cfg.get("emit_dieu_when_no_khoan", True)
    emit_structural_headings = chunking_cfg.get("emit_structural_headings", False)

    docs_cfg = cfg.get("documents", {}) or {}
    csv_cfg = cfg.get("csv_sources", {}) or {}

    if args.doc:
        if args.doc in docs_cfg:
            pdf_targets = {args.doc: docs_cfg[args.doc]}
            csv_targets = {}
        elif args.doc in csv_cfg:
            pdf_targets = {}
            csv_targets = {args.doc: csv_cfg[args.doc]}
        else:
            logger.error(
                f"Unknown doc key: {args.doc}. Available PDF docs: {list(docs_cfg.keys())}; "
                f"Available CSV sources: {list(csv_cfg.keys())}"
            )
            sys.exit(2)
    else:
        pdf_targets = docs_cfg
        csv_targets = csv_cfg

    all_chunks: List[Dict] = []
    all_tables: List[Dict] = []

    # ---------------- PDF sources ----------------
    for key, meta in pdf_targets.items():
        pdf_path = raw_dir / f"{key}.pdf"
        if not pdf_path.exists():
            logger.warning(f"Missing PDF: {pdf_path} — skipping")
            continue
        chunks, tables = ingest_document(
            pdf_path=pdf_path,
            source_doc=meta["title"],
            doc_short=meta["short"],
            granularity=granularity,
            max_chunk_chars=max_chars,
            emit_dieu_when_no_khoan=emit_dieu_when_no_khoan,
            emit_structural_headings=emit_structural_headings,
        )
        all_chunks.extend(chunks)
        all_tables.extend(tables)

    # ---------------- CSV sources ----------------
    for key, meta in csv_targets.items():
        csv_path = raw_dir / meta["file"]
        if not csv_path.exists():
            logger.warning(f"Missing CSV: {csv_path} — skipping")
            continue
        chunks = ingest_csv_source(
            csv_path=csv_path,
            source_doc=meta["title"],
            doc_short=meta["short"],
            granularity=granularity,
            max_chunk_chars=max_chars,
            emit_dieu_when_no_khoan=emit_dieu_when_no_khoan,
            emit_structural_headings=emit_structural_headings,
        )
        all_chunks.extend(chunks)

    # ---- Disambiguate duplicate chunk_ids ----
    # In long consolidated decrees (VBHN) the chunker can occasionally collide
    # on (Điều, Khoản, Điểm) — typically when a Mục boundary or a line-break
    # artifact tricks the state machine into resetting the Điểm sequence. The
    # text itself is real and we want to KEEP it for the retrieval index, but
    # Chroma requires unique IDs, so we append a `__occN` suffix to collisions.
    seen_ids: Dict[str, int] = {}
    n_disambiguated = 0
    for c in all_chunks:
        base = c["chunk_id"]
        if base in seen_ids:
            seen_ids[base] += 1
            c["chunk_id"] = f"{base}__occ{seen_ids[base]}"
            n_disambiguated += 1
        else:
            seen_ids[base] = 1
    if n_disambiguated:
        logger.warning(
            f"Disambiguated {n_disambiguated} duplicate chunk_id(s) by appending __occN"
        )
    for c in all_tables:
        base = c["chunk_id"]
        if base in seen_ids:
            seen_ids[base] += 1
            c["chunk_id"] = f"{base}__occ{seen_ids[base]}"
        else:
            seen_ids[base] = 1

    # ---- Report ----
    logger.info(f"TOTAL chunks: {len(all_chunks)}")
    logger.info(f"TOTAL tables: {len(all_tables)}")
    by_type = {}
    for c in all_chunks:
        by_type[c["chunk_type"]] = by_type.get(c["chunk_type"], 0) + 1
    logger.info(f"By chunk_type: {by_type}")

    if args.dry_run:
        logger.info("--dry-run: not writing output files.")
        # Show a sample.
        for c in all_chunks[:2]:
            logger.info(f"SAMPLE chunk:\n  citation: {c['full_citation']}\n  text: {c['text'][:180]}…")
        return

    n_c = write_jsonl(chunks_out, all_chunks)
    n_t = write_jsonl(tables_out, all_tables)
    logger.info(f"Wrote {n_c} chunks → {chunks_out}")
    logger.info(f"Wrote {n_t} tables → {tables_out}")


if __name__ == "__main__":
    main()
