"""Phase 1 — Data Engineering: PDF parsing, table extraction, semantic chunking.

We deliberately avoid eager imports of PDF backends (fitz / pdfplumber)
at package import time — that lets you import the pure-Python chunker
(`from src.phase1_data_engineering.semantic_chunker import SemanticChunker`)
without having PyMuPDF / pdfplumber installed (useful in CI / unit tests).
"""
__all__ = [
    "SemanticChunker",
    "LegalChunk",
    "PDFParser",
    "TableExtractor",
    "MetadataTagger",
]
