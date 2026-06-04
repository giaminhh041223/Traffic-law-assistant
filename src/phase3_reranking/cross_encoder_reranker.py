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

# Move heavy imports to top level to avoid thread deadlocks during chromadb execution
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")
try:
    import torch
    try:
        torch.classes.__path__ = []
    except Exception:
        pass
    from sentence_transformers import CrossEncoder
except ImportError as e:
    logger.warning(f"Could not import torch/CrossEncoder at top level: {e}")

from src.utils.io import load_yaml
from src.utils.vn_tokenizer import VnTokenizer


def _resolve_device(device: str, min_free_mb: int = 300) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu' with VRAM awareness.

    On small-VRAM GPUs (e.g. MX550, 2 GB) we must avoid OOM by checking
    free memory before committing to CUDA.  If fewer than ``min_free_mb``
    MB are available the function falls back to CPU and logs a warning.
    """
    if device != "auto":
        return device
    try:
        if not torch.cuda.is_available():
            return "cpu"
        free, total = torch.cuda.mem_get_info(0)
        free_mb = free / (1024 ** 2)
        total_mb = total / (1024 ** 2)
        if free_mb < min_free_mb:
            logger.warning(
                f"GPU VRAM thấp ({free_mb:.0f}/{total_mb:.0f} MB free) "
                f"— fallback sang CPU để tránh OOM"
            )
            return "cpu"
        logger.info(f"GPU VRAM OK ({free_mb:.0f}/{total_mb:.0f} MB free) → dùng CUDA")
        return "cuda"
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
            logger.info(f"Loading CrossEncoder '{self.model_name}' on {self.device} …")
            try:
                self._model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device=self.device,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if self.device == "cuda" and ("out of memory" in str(e).lower() or "CUDA" in str(e)):
                    logger.warning(f"GPU OOM khi load CrossEncoder — fallback sang CPU: {e}")
                    torch.cuda.empty_cache()
                    self.device = "cpu"
                    self._model = CrossEncoder(
                        self.model_name,
                        max_length=self.max_length,
                        device="cpu",
                    )
                else:
                    raise
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
        chunk_text = ""
        # For TABLE chunks, prefer the linear form (no markdown noise).
        if chunk.get("chunk_type") == "table" and chunk.get("linear_form"):
            chunk_text = chunk.get("linear_form")
        else:
            chunk_text = chunk.get("text", "")

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

        if chunk_text:
            parts.append(chunk_text)

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
            try:
                model = self.model
                scores = model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )
                scores = [float(s) for s in scores]
            except Exception as e:
                # If using cuda, retry on CPU
                if "cuda" in str(self.device).lower():
                    logger.warning(f"GPU Error/OOM during CrossEncoder prediction. Fallback to CPU: {e}")
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    try:
                        self.device = "cpu"
                        self._model = None  # Force reload on CPU
                        model = self.model
                        scores = model.predict(
                            pairs,
                            batch_size=self.batch_size,
                            show_progress_bar=False,
                        )
                        scores = [float(s) for s in scores]
                    except Exception as inner_e:
                        logger.error(f"CrossEncoder CPU fallback retry failed: {inner_e}")
                        # Fallback to RRF ordering
                        scores = [float(len(candidates) - i) for i in range(len(candidates))]
                else:
                    logger.warning(f"CrossEncoder reranking failed, falling back gracefully to original RRF ranking: {e}")
                    # Mock scores matching RRF ranks to preserve input ordering
                    scores = [float(len(candidates) - i) for i in range(len(candidates))]

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
