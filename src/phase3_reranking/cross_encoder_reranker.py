"""Cross-Encoder re-ranker — Phase 2's Top-10 → Top-3 for the LLM in Phase 4.

Why cross-encoder, not just keep the bi-encoder's top-10?
    The dense retriever scores query-vs-document via two INDEPENDENT
    embeddings (`cos(E(q), E(d))`). The encoder never sees q and d in the
    same forward pass, so subtle cross-references — e.g. "with xe ô tô"
    in the query vs "đối với xe mô tô" in the doc — are easy to confuse.

    A cross-encoder concatenates q ⊕ d and runs both through ONE
    transformer with full bidirectional self-attention. This is much more
    accurate but ~100× slower per pair, which is exactly why we run it
    on the top-10 candidates only, not the whole corpus.

Two-stage retrieval pattern (academic textbook):
    Stage 1  — Recall:  Hybrid BM25 + dense  →  Top-10 candidates  (cheap)
    Stage 2  — Precision: Cross-encoder      →  Top-3              (accurate)

Vietnamese-specific note:
    `itdainb/PhoRanker` is PhoBERT-based; it expects word-segmented input
    (multi-syllable words joined by '_'). We apply the SAME pyvi
    segmentation used in Phase 2 to keep retrieval and re-ranking aligned.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from src.utils.io import load_yaml
from src.utils.vn_tokenizer import VnTokenizer


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class CrossEncoderReranker:
    """Re-ranks Phase 2 candidates using a Vietnamese cross-encoder."""

    def __init__(
        self,
        model_name: str = "itdainb/PhoRanker",
        max_length: int = 256,
        device: str = "auto",
        batch_size: int = 16,
        segment_input: bool = True,
        include_dieu_title: bool = True,
        include_citation_prefix: bool = False,
        # Test hook — pass a callable that takes List[[q, d]] and returns List[float].
        predictor: Optional[Callable[[List[List[str]]], List[float]]] = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.segment_input = segment_input
        self.include_dieu_title = include_dieu_title
        self.include_citation_prefix = include_citation_prefix
        self._predictor = predictor       # for tests / mocks
        self._model = None
        self._tokenizer: Optional[VnTokenizer] = None

    # ----------------------------- lazy resources -----------------------------

    @property
    def model(self):
        if self._model is None and self._predictor is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder '{self.model_name}' on {self.device} …")
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
        return self._model

    @property
    def tokenizer(self) -> VnTokenizer:
        if self._tokenizer is None:
            # No lowercase / punctuation stripping — PhoBERT-family handles casing internally.
            self._tokenizer = VnTokenizer(backend="pyvi", lowercase=False, remove_punctuation=False)
        return self._tokenizer

    # ----------------------------- text construction -----------------------------

    def build_doc_text(self, chunk: Dict[str, Any]) -> str:
        """Same template as Phase 2's `build_embed_text` — keeps the surface form aligned."""
        parts: List[str] = []
        if self.include_citation_prefix and chunk.get("full_citation"):
            parts.append(chunk["full_citation"])
        if self.include_dieu_title and chunk.get("dieu_title"):
            parts.append(chunk["dieu_title"])
        # For TABLE chunks, prefer the linear form (no markdown noise).
        if chunk.get("chunk_type") == "table" and chunk.get("linear_form"):
            parts.append(chunk["linear_form"])
        else:
            parts.append(chunk.get("text", ""))

        # Lexicon Expansion: Map metadata tags back to fluent, natural Vietnamese sentences.
        # This keeps the Cross-Encoder's text aligned with the indexes.
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

        return ". ".join(p for p in parts if p)

    # ----------------------------- inference -----------------------------

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Re-rank `candidates` (output of Phase 2's HybridSearcher) by cross-encoder score.

        Each returned chunk gets two new fields:
            * cross_encoder_score — raw logit from the cross-encoder
            * pre_rerank_rank     — its position (1-indexed) in the input candidates
        """
        if not candidates:
            return []

        # Build (q, d) pairs with the same segmentation used at training time.
        if self.segment_input:
            q = self.tokenizer.segment(query)
            doc_texts = [self.tokenizer.segment(self.build_doc_text(c)) for c in candidates]
        else:
            q = query
            doc_texts = [self.build_doc_text(c) for c in candidates]

        pairs: List[List[str]] = [[q, d] for d in doc_texts]

        # Score — real model or injected predictor.
        if self._predictor is not None:
            scores = list(self._predictor(pairs))
        else:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            scores = [float(s) for s in scores]

        if len(scores) != len(candidates):
            raise RuntimeError(
                f"Cross-encoder returned {len(scores)} scores for {len(candidates)} candidates."
            )

        # Attach scores, preserving original metadata, then sort.
        reranked: List[Dict[str, Any]] = []
        for i, (c, s) in enumerate(zip(candidates, scores), start=1):
            new_c = dict(c)
            new_c["cross_encoder_score"] = float(s)
            new_c["pre_rerank_rank"] = i
            reranked.append(new_c)

        reranked.sort(key=lambda x: -x["cross_encoder_score"])
        return reranked[:top_k]

    # ----------------------------- factory -----------------------------

    @classmethod
    def from_config(
        cls,
        retrieval_yaml: Union[str, Path] = "configs/retrieval.yaml",
        **overrides: Any,
    ) -> "CrossEncoderReranker":
        cfg = load_yaml(retrieval_yaml)["cross_encoder"]
        params = dict(
            model_name=cfg["model_name"],
            max_length=cfg.get("max_length", 256),
            device=cfg.get("device", "auto"),
            batch_size=cfg.get("batch_size", 16),
            segment_input=cfg.get("segment_input", True),
            include_dieu_title=cfg.get("include_dieu_title", True),
            include_citation_prefix=cfg.get("include_citation_prefix", False),
        )
        params.update(overrides)
        return cls(**params)
