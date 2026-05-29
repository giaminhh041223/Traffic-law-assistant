"""Phase 2 CLI: build both BM25 and dense (ChromaDB) indexes from Phase 1's
chunks.jsonl (+ optionally tables.jsonl).

Usage:
    python -m src.phase2_hybrid_search.build_indexes \
        --settings configs/settings.yaml \
        --retrieval configs/retrieval.yaml \
        [--include-tables / --no-include-tables]   # default: include
        [--rebuild]                                 # wipe existing Chroma collection
        [--bm25-only / --dense-only]                # build just one of the two
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from loguru import logger

from src.utils.io import load_yaml, read_jsonl
from src.utils.logging import setup_logger

from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever


# ---------------------------------------------------------------------------
# Embed-text builder — what goes into BOTH indexes
# ---------------------------------------------------------------------------
def build_embed_text(chunk: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """Construct the text we'll index for a given chunk.

    Strategy:
      * Prefix with `dieu_title` (when present) so the article topic is
        always a strong signal for both lexical and semantic retrieval.
      * For TABLE chunks, prefer `linear_form` (clean sentences) over the
        raw markdown — markdown pipes & dashes pollute BM25 tokenisation.
      * Optionally prepend the full citation (`Khoản 1 Điều 5 …`) so
        queries that reference legal anchors get an extra match signal.
    """
    template = cfg.get("embedding_text_template", {})
    include_title = template.get("include_dieu_title", True)
    include_citation = template.get("include_citation_prefix", False)
    sep = template.get("join_separator", ". ")

    parts: List[str] = []
    if include_citation and chunk.get("full_citation"):
        parts.append(chunk["full_citation"])
    if include_title and chunk.get("dieu_title"):
        parts.append(chunk["dieu_title"])

    if chunk.get("chunk_type") == "table" and chunk.get("linear_form"):
        parts.append(chunk["linear_form"])
    else:
        parts.append(chunk.get("text", ""))

    # Lexicon Expansion: Map metadata tags back to fluent, natural Vietnamese sentences.
    # This highly aligns with the pre-training data of Cross-Encoders (PhoRanker),
    # dramatically improving semantic matching for colloquial queries.
    meta = chunk.get("metadata") or {}
    natural_parts: List[str] = []
    
    violation_mapping = {
        "toc_do": "chạy quá tốc độ quy định, vượt tốc độ giới hạn",
        "nong_do_con": "vi phạm nồng độ cồn, uống rượu bia khi lái xe, sử dụng rượu bia",
        "ma_tuy": "sử dụng chất ma túy, lái xe khi phê ma túy",
        "vuot_den_do": "vượt đèn đỏ, vượt đèn vàng, không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
        "khong_mu_bao_hiem": "không đội mũ bảo hiểm, không cài quai mũ bảo hiểm theo quy định",
        "sai_lan": "đi sai làn đường, đi không đúng phần đường làn đường quy định",
        "khong_gplx": "không có giấy phép lái xe, không có bằng lái xe, không mang gplx",
        "dung_do_xe": "dừng xe đỗ xe sai quy định, dừng đỗ xe trái phép",
        "lui_xe": "lùi xe nguy hiểm, lùi xe không đúng quy định",
        "quay_dau": "quay đầu xe sai quy định, quay đầu xe nơi cấm quay đầu",
        "vuot_xe": "vượt xe sai quy định, vượt ẩu nguy hiểm",
        "cho_qua_so_nguoi": "chở quá số người quy định, chở quá tải người",
        "cho_qua_tai": "chở hàng quá trọng tải, xe quá tải",
        "dien_thoai": "sử dụng điện thoại khi đang lái xe, dùng điện thoại di động",
        "bao_hiem": "không có bảo hiểm bắt buộc trách nhiệm dân sự",
        "dang_kiem": "hết hạn đăng kiểm, không đăng kiểm xe",
        "bien_so": "lỗi biển số xe, không gắn biển số xe",
    }
    
    vehicle_mapping = {
        "o_to": "xe ô tô, xe hơi",
        "xe_may": "xe máy, xe mô tô, xe gắn máy, xe máy điện",
        "may_keo": "máy kéo",
        "xe_chuyen_dung": "xe máy chuyên dùng, xe chuyên dùng",
        "xe_dap": "xe đạp, xe đạp điện, xe đạp máy",
        "xe_tho_so": "xe thô sơ",
        "nguoi_di_bo": "người đi bộ",
    }

    # Add violation sentence
    vios = [violation_mapping[v] for v in meta.get("violation_type") or [] if v in violation_mapping]
    if vios:
        natural_parts.append(f"Quy định này áp dụng cho lỗi {', '.join(vios)}.")
        
    # Vehicle sentence
    vehs = [vehicle_mapping[v] for v in meta.get("vehicle_type") or [] if v in vehicle_mapping]
    if vehs:
        natural_parts.append(f"Áp dụng cho người điều khiển {', '.join(vehs)}.")

    if natural_parts:
        parts.append(" ".join(natural_parts))

    return sep.join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Metadata pruning — strip non-serialisable / large fields
# ---------------------------------------------------------------------------
_TOP_LEVEL_METADATA_KEYS = (
    "doc_short", "source_doc", "chunk_type",
    "chuong", "muc", "dieu", "khoan", "diem",
    "dieu_title", "full_citation", "page_num", "char_count",
)


def build_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Merge top-level chunk fields with nested `metadata` for retrieval-time filtering.

    The result preserves list-typed taxonomies (vehicle_type, etc.) in their
    original shape — DenseRetriever flattens them to Chroma flags internally,
    while BM25Retriever uses them as-is for in-memory filtering.
    """
    meta: Dict[str, Any] = {}
    for k in _TOP_LEVEL_METADATA_KEYS:
        v = chunk.get(k)
        if v is not None:
            meta[k] = v
    nested = chunk.get("metadata") or {}
    for k, v in nested.items():
        if v is None:
            continue
        meta[k] = v
    return meta


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_documents(
    chunks_path: Path,
    tables_path: Path,
    include_tables: bool,
    retrieval_cfg: Dict[str, Any],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Read chunks (+ tables) and pack as (chunk_id, embed_text, metadata) tuples."""
    docs: List[Tuple[str, str, Dict[str, Any]]] = []
    n_chunks = 0
    for c in read_jsonl(chunks_path):
        text = build_embed_text(c, retrieval_cfg)
        if not text.strip():
            continue
        docs.append((c["chunk_id"], text, build_metadata(c)))
        n_chunks += 1
    logger.info(f"Loaded {n_chunks:,} semantic chunks from {chunks_path}")

    if include_tables and tables_path.exists():
        n_tables = 0
        for c in read_jsonl(tables_path):
            text = build_embed_text(c, retrieval_cfg)
            if not text.strip():
                continue
            docs.append((c["chunk_id"], text, build_metadata(c)))
            n_tables += 1
        logger.info(f"Loaded {n_tables:,} table chunks from {tables_path}")
    elif include_tables:
        logger.warning(f"--include-tables requested but {tables_path} not found; skipping.")

    if not docs:
        logger.error("No documents to index. Did you run Phase 1 ingestion?")
        sys.exit(2)

    return docs


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_bm25_index(
    docs: List[Tuple[str, str, Dict[str, Any]]],
    bm25_cfg: Dict[str, Any],
    out_path: Path,
) -> None:
    logger.info("Building BM25 index …")
    retriever = BM25Retriever(
        variant=bm25_cfg.get("variant", "okapi"),
        k1=bm25_cfg.get("k1", 1.5),
        b=bm25_cfg.get("b", 0.75),
        epsilon=bm25_cfg.get("epsilon", 0.25),
        tokenizer_backend=bm25_cfg.get("tokenizer_backend", "pyvi"),
        lowercase=bm25_cfg.get("lowercase", True),
        remove_punctuation=bm25_cfg.get("remove_punctuation", True),
    )
    retriever.index(docs)
    retriever.save(out_path)
    logger.info(f"BM25 index ({len(retriever):,} chunks) saved → {out_path}")


def build_dense_index(
    docs: List[Tuple[str, str, Dict[str, Any]]],
    retrieval_cfg: Dict[str, Any],
    rebuild: bool,
) -> None:
    logger.info("Building dense index (ChromaDB) …")
    dense = DenseRetriever.from_config(retrieval_cfg)
    if rebuild:
        dense.reset()
    dense.index(docs, upsert=True)
    logger.info(f"Dense collection size: {dense.count():,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build BM25 + dense indexes from chunks.jsonl")
    ap.add_argument("--settings", default="configs/settings.yaml")
    ap.add_argument("--retrieval", default="configs/retrieval.yaml")
    ap.add_argument("--include-tables", dest="include_tables", action="store_true", default=True)
    ap.add_argument("--no-include-tables", dest="include_tables", action="store_false")
    ap.add_argument("--rebuild", action="store_true", help="Wipe the Chroma collection before indexing.")
    ap.add_argument("--bm25-only", action="store_true")
    ap.add_argument("--dense-only", action="store_true")
    args = ap.parse_args()

    if args.bm25_only and args.dense_only:
        ap.error("--bm25-only and --dense-only are mutually exclusive.")

    settings = load_yaml(args.settings)
    retrieval = load_yaml(args.retrieval)
    setup_logger(settings.get("paths", {}).get("log_dir", "logs"),
                 settings.get("logging", {}).get("level", "INFO"))

    chunks_path = Path(settings["paths"]["chunks_file"])
    tables_path = Path(settings["paths"]["tables_file"])
    bm25_index_path = Path(settings["paths"]["bm25_index"])

    docs = load_documents(chunks_path, tables_path, args.include_tables, retrieval)

    if not args.dense_only:
        build_bm25_index(docs, retrieval["bm25"], bm25_index_path)

    if not args.bm25_only:
        build_dense_index(docs, retrieval, rebuild=args.rebuild)

    logger.info("Done. Phase 2 indexes are ready for hybrid search.")


if __name__ == "__main__":
    main()
