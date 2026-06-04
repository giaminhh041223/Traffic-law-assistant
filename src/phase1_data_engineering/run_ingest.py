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
from typing import Dict, List, Optional

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
    filter_doc_short: Optional[str] = None,
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
        filter_doc_short=filter_doc_short,
    )
    return [attach_metadata(c) for c in chunks]


def main():
    parser = argparse.ArgumentParser(description="Phase 1 ingestion CLI")
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--doc", default=None, help="Ingest a single registered doc (key in csv_sources, or a specific doc_short)")
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

    csv_cfg = cfg.get("csv_sources", {}) or {}
    metadata_cfg = csv_cfg.get("metadata", {})

    if not metadata_cfg:
        logger.error("No 'metadata' CSV source registered in configs/settings.yaml")
        sys.exit(2)

    csv_path = raw_dir / metadata_cfg["file"]
    if not csv_path.exists():
        logger.error(f"Missing consolidated CSV file at: {csv_path}")
        sys.exit(2)

    # Determine filter_doc_short
    filter_doc_short = None
    if args.doc and args.doc not in ["metadata", "CONSOLIDATED"]:
        filter_doc_short = args.doc

    all_chunks: List[Dict] = []
    all_tables: List[Dict] = []

    # Process text chunks from consolidated CSV
    logger.info(f"Ingesting text chunks from consolidated CSV: {csv_path.name}")
    chunks = ingest_csv_source(
        csv_path=csv_path,
        source_doc=metadata_cfg["title"],
        doc_short=metadata_cfg["short"],
        granularity=granularity,
        max_chunk_chars=max_chars,
        emit_dieu_when_no_khoan=emit_dieu_when_no_khoan,
        emit_structural_headings=emit_structural_headings,
        filter_doc_short=filter_doc_short,
    )
    all_chunks.extend(chunks)

    # Process tables from PDFs
    PDF_STEM_MAPPING = {
        "QCVN41_TT": "QCVN_41_2024",
        "QCVN41_2024": "QCVN_41_2024_attachment",
    }
    DOC_TITLE_MAPPING = {
        "QCVN41_TT": "Thông tư 51/2024/TT-BGTVT ban hành QCVN 41:2024/BGTVT",
        "QCVN41_2024": "QCVN 41:2024/BGTVT — Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ",
        "LUAT_TTATGT_2024": "Luật Trật tự, An toàn Giao thông Đường bộ (Luật 36/2024/QH15)",
        "LUAT_DUONGBO_2024": "Luật Đường bộ 2024 (Luật 35/2024/QH15)",
        "ND168_2024": "Nghị định 168/2024/NĐ-CP quy định xử phạt vi phạm hành chính về trật tự, an toàn giao thông đường bộ; trừ điểm, phục hồi điểm giấy phép lái xe",
        "TT30_2024": "Thông tư 30/2024/TT-BGTVT sửa đổi, bổ dung một số điều về kiểm định an toàn kỹ thuật và bảo vệ môi trường xe cơ giới",
        "ND336_2025": "Nghị định 336/2025/NĐ-CP quy định xử phạt vi phạm hành chính trong hoạt động đường bộ",
        "TT38_2024": "Thông tư 38/2024/TT-BGTVT quy định về tốc độ và khoảng cách an toàn của xe cơ giới, xe máy chuyên dùng",
        "ND94_2026": "Nghị định 94/2026/NĐ-CP quy định về hoạt động đào tạo và sát hạch lái xe",
        "TT73_2024": "Thông tư 73/2024/TT-BCA quy định công tác tuần tra, kiểm soát, xử lý vi phạm pháp luật về trật tự, an toàn giao thông đường bộ của CSGT",
        "ND67_2023": "Nghị định 67/2023/NĐ-CP quy định về bảo hiểm bắt buộc trách nhiệm dân sự của chủ xe cơ giới, bảo hiểm cháy, nổ bắt buộc, bảo hiểm bắt buộc trong hoạt động đầu tư xây dựng",
        "TT36_2024": "Thông tư 36/2024/TT-BYT quy định về tiêu chuẩn sức khỏe, việc khám sức khỏe đối với người lái xe, người điều khiển xe máy chuyên dùng; việc khám sức khỏe định kỳ đối với người hành nghề lái xe ô tô; cơ sở dữ liệu về sức khỏe của người lái xe, người điều khiển xe máy chuyên dùng",
        "TT35_2024": "Thông tư 35/2024/TT-BGTVT quy định về đào tạo, sát hạch, cấp giấy phép lái xe; cấp, sử dụng giấy phép lái xe quốc tế; đào tạo, kiểm tra, cấp chứng chỉ bồi dưỡng kiến thức pháp luật về giao thông đường bộ",
        "TT47_2024": "Thông tư 47/2024/TT-BGTVT quy định trình tự, thủ tục kiểm định, miễn kiểm định lần đầu cho xe cơ giới, xe máy chuyên dùng; trình tự, thủ tục chứng nhận an toàn kỹ thuật và bảo vệ môi trường đối với xe cơ giới cải tạo, xe máy chuyên dùng cải tạo; trình tự, thủ tục kiểm định khí thải xe mô tô, xe gắn máy",
        "ND166_2024": "Nghị định 166/2024/NĐ-CP quy định về điều kiện kinh doanh dịch vụ kiểm định xe cơ giới; tổ chức, hoạt động của cơ sở đăng kiểm; niên hạn sử dụng của xe cơ giới",
        "ND158_2024": "Nghị định 158/2024/NĐ-CP quy định về hoạt động vận tải đường bộ",
        "ND165_2024": "Nghị định 165/2024/NĐ-CP quy định chi tiết, hướng dẫn thi hành một số điều của Luật Đường bộ và Điều 77 Luật Trật tự, an toàn giao thông đường bộ",
        "TT72_2024": "Thông tư 72/2024/TT-BCA quy định công tác điều tra, giải quyết tai nạn giao thông đường bộ của lực lượng CSGT",
        "TT69_2024": "Thông tư 69/2024/TT-BCA quy định về công tác chỉ huy, điều khiển giao thông đường bộ của lực lượng CSGT",
        "TT65_2024": "Thông tư 65/2024/TT-BCA quy định về kiểm tra kiến thức pháp luật về trật tự, an toàn giao thông đường bộ",
        "TT39_2024": "Thông tư 39/2024/TT-BGTVT quy định về tải trọng, khổ giới hạn của đường bộ; lưu hành xe quá tải trọng, xe quá khổ giới hạn, xe bánh xích; vận chuyển hàng siêu trường, siêu trọng",
        "ND161_2024": "Nghị định 161/2024/NĐ-CP quy định về vận chuyển hàng hóa nguy hiểm bằng phương tiện giao thông cơ giới đường bộ và phương tiện thủy nội địa",
        "TT51_2025": "Thông tư 51/2025/TT-BCA sửa đổi, bổ sung một số điều của Thông tư 79/2024/TT-BCA quy định về cấp, thu hồi chứng nhận đăng ký xe, biển số xe cơ giới, xe máy chuyên dùng",
        "TT79_2024": "Thông tư 79/2024/TT-BCA quy định về cấp, thu hồi chứng nhận đăng ký xe, biển số xe cơ giới, xe máy chuyên dùng",
        "ND160_2024": "Nghị định 160/2024/NĐ-CP quy định về đào tạo và sát hạch lái xe",
        "ND151_2024": "Nghị định 151/2024/NĐ-CP quy định chi tiết một số điều của Luật Trật tự, an toàn giao thông đường bộ về giáo dục kiến thức pháp luật, cơ sở dữ liệu và trách nhiệm quản lý nhà nước",
    }

    active_docs = set()
    if filter_doc_short:
        active_docs = {filter_doc_short}
    else:
        active_docs = set(DOC_TITLE_MAPPING.keys())

    for d_short in active_docs:
        stem = PDF_STEM_MAPPING.get(d_short)
        if stem:
            pdf_path = raw_dir / f"{stem}.pdf"
            if pdf_path.exists():
                logger.info(f"Extracting tables for {d_short} from PDF: {pdf_path.name}")
                tables = TableExtractor(doc_short=d_short).extract(pdf_path)
                logger.info(f"  → {len(tables)} tables")
                table_records = [table_to_chunk(t, DOC_TITLE_MAPPING[d_short], d_short) for t in tables]
                all_tables.extend(table_records)

    # ---- Disambiguate duplicate chunk_ids ----
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
        for c in all_chunks[:2]:
            logger.info(f"SAMPLE chunk:\n  citation: {c['full_citation']}\n  text: {c['text'][:180]}…")
        return

    n_c = write_jsonl(chunks_out, all_chunks)
    n_t = write_jsonl(tables_out, all_tables)
    logger.info(f"Wrote {n_c} chunks → {chunks_out}")
    logger.info(f"Wrote {n_t} tables → {tables_out}")


if __name__ == "__main__":
    main()
