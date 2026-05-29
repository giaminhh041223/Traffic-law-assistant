"""Dense retriever — Vietnamese SBERT embeddings stored in ChromaDB.

Why segment Vietnamese input before encoding?
    `keepitreal/vietnamese-sbert` is fine-tuned on PhoBERT, which was
    pre-trained on word-segmented Vietnamese text (multi-syllable words
    joined by '_'). Passing raw text degrades quality silently — the
    tokenizer falls back to subword splits that don't match training.

    We apply the SAME pyvi segmentation as in BM25 (via VnTokenizer.segment)
    so that the corpus and queries share lexical preprocessing — sparse and
    dense retrievers see the same surface form.

ChromaDB metadata limitations:
    Chroma metadata values MUST be scalar (str | int | float | bool).
    Lists are rejected. We flatten list-valued fields (vehicle_type,
    violation_type, sanctions) into boolean flags using prefixes from
    `flatten_list_fields` in retrieval.yaml — e.g.
        vehicle_type=["o_to","xe_may"] → vt_o_to=True, vt_xe_may=True
    Search-time filters in their natural form (e.g. {"violation_type":
    "nong_do_con"}) are translated back to {"vio_nong_do_con": True}.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

from src.utils.vn_tokenizer import VnTokenizer

FilterValue = Union[str, int, float, bool, List[Union[str, int, float, bool]]]
FilterDict = Dict[str, FilterValue]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_device(device: str) -> str:
    """auto → cuda if available, else cpu."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _coerce_scalar(v: Any) -> Optional[Any]:
    """Return value if Chroma-storable as scalar, else None."""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    return None  # drop lists / dicts / numpy types


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class DenseRetriever:
    """SBERT → ChromaDB retriever with metadata filtering."""

    def __init__(
        self,
        model_name: str = "keepitreal/vietnamese-sbert",
        persist_dir: str = "vector_store/chroma",
        collection_name: str = "traffic_law_chunks",
        distance: str = "cosine",                # cosine | l2 | ip
        device: str = "auto",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        max_seq_length: int = 256,
        segment_input: bool = True,
        flatten_list_fields: Optional[Dict[str, str]] = None,
        hnsw_construction_ef: int = 200,
        hnsw_M: int = 32,
    ):
        self.model_name = model_name
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.distance = distance
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length
        self.segment_input = segment_input
        self.flatten_list_fields = flatten_list_fields or {}
        self.hnsw_construction_ef = hnsw_construction_ef
        self.hnsw_M = hnsw_M

        # Lazy resources — first access triggers loading.
        self._model = None
        self._tokenizer: Optional[VnTokenizer] = None
        self._client = None
        self._collection = None

    # ----------------------------- lazy resources -----------------------------

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            dev = _resolve_device(self.device)
            logger.info(f"Loading SBERT model '{self.model_name}' on {dev} …")
            if dev == "cpu":
                threads = min(8, os.cpu_count() or 4)
                try:
                    torch.set_num_threads(threads)
                except Exception as e:
                    logger.debug(f"Failed to set_num_threads: {e}")
                try:
                    torch.set_num_interop_threads(threads)
                except Exception as e:
                    logger.debug(f"Failed to set_num_interop_threads: {e}")
                logger.info(f"Configured PyTorch CPU threads: num_threads={threads}, interop_threads={threads}")
            self._model = SentenceTransformer(self.model_name, device=dev)
            self._model.max_seq_length = self.max_seq_length
        return self._model

    @property
    def tokenizer(self) -> VnTokenizer:
        if self._tokenizer is None:
            # For dense we don't strip punctuation / lowercase aggressively;
            # we only want word segmentation. PhoBERT-based models do their
            # own casing internally.
            self._tokenizer = VnTokenizer(backend="pyvi", lowercase=False, remove_punctuation=False)
        return self._tokenizer

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            metadata = {
                "hnsw:space": self.distance,
                "hnsw:construction_ef": self.hnsw_construction_ef,
                "hnsw:M": self.hnsw_M,
            }
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata=metadata,
            )
        return self._collection

    # ----------------------------- indexing -----------------------------

    def reset(self) -> None:
        """Delete the collection and recreate it empty."""
        import chromadb
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        try:
            self._client.delete_collection(self.collection_name)
            logger.info(f"Deleted existing collection '{self.collection_name}'")
        except Exception:
            pass
        self._collection = None  # force re-creation on next access

    def index(
        self,
        docs: List[Tuple[str, str, Dict[str, Any]]],
        upsert: bool = True,
    ) -> int:
        """Embed and add `docs` to the Chroma collection. Returns count added.

        Args:
            docs: list of (chunk_id, text, metadata) tuples.
            upsert: if True, use Chroma's upsert (idempotent re-runs); if False, add() (errors on duplicates).
        """
        if not docs:
            return 0

        ids = [d[0] for d in docs]
        texts = [d[1] for d in docs]
        metadatas = [self._to_chroma_metadata(d[2]) for d in docs]

        if self.segment_input:
            texts_to_encode = [self.tokenizer.segment(t) for t in texts]
        else:
            texts_to_encode = texts

        logger.info(f"Encoding {len(texts_to_encode)} chunks (batch_size={self.batch_size}) …")
        embeddings = self.model.encode(
            texts_to_encode,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # ChromaDB has a default max add batch of ~5000. Chunk just in case.
        coll = self.collection
        added = 0
        BATCH = 1024
        for start in range(0, len(ids), BATCH):
            sl = slice(start, start + BATCH)
            kwargs = dict(
                ids=ids[sl],
                embeddings=embeddings[sl].tolist(),
                metadatas=metadatas[sl],
                documents=texts[sl],   # store the raw (un-segmented) text for display
            )
            if upsert:
                coll.upsert(**kwargs)
            else:
                coll.add(**kwargs)
            added += len(ids[sl])
        logger.info(f"Indexed {added} chunks into '{self.collection_name}' "
                    f"(collection size now: {coll.count()})")
        return added

    # ----------------------------- search -----------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[FilterDict] = None,
    ) -> List[Tuple[str, float]]:
        """Return [(chunk_id, similarity)] — similarity in [-1, 1] for cosine."""
        if not query.strip():
            return []
        q = self.tokenizer.segment(query) if self.segment_input else query
        q_emb = self.model.encode(
            [q],
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        where = self._translate_filters(filters) if filters else None
        coll = self.collection

        res = coll.query(
            query_embeddings=[q_emb.tolist()],
            n_results=min(top_k, max(1, coll.count())),
            where=where,
        )

        out: List[Tuple[str, float]] = []
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, d in zip(ids, dists):
            sim = self._distance_to_similarity(float(d))
            out.append((cid, sim))
        return out

    # ----------------------------- helpers -----------------------------

    def _distance_to_similarity(self, dist: float) -> float:
        """Translate ChromaDB distance back into a [0, 1]-ish similarity for inspection."""
        if self.distance == "cosine":
            # Chroma returns 1 - cosine_similarity → similarity = 1 - dist
            return 1.0 - dist
        if self.distance == "ip":
            # inner product: higher is better; Chroma negates
            return -dist
        return -dist  # l2: lower distance = higher similarity (sign-flip for monotonicity)

    def _to_chroma_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten list fields → boolean flags; drop None / unsupported types."""
        out: Dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, list):
                prefix = self.flatten_list_fields.get(k)
                if prefix is None:
                    continue  # silently drop unknown list-valued fields
                for item in v:
                    if isinstance(item, (str, int)):
                        out[f"{prefix}{item}"] = True
                continue
            scalar = _coerce_scalar(v)
            if scalar is not None:
                out[k] = scalar
        return out

    def _translate_filters(self, filters: FilterDict) -> Optional[Dict[str, Any]]:
        """User-facing filter dict → ChromaDB `where` clause.

        Each key in `filters` is either:
          * a flattened list-field (e.g. "violation_type") — translated via prefix to flags
          * a scalar field (e.g. "doc_short") — passed through with $eq / $in
        Multiple keys combined with $and (Chroma's implicit AND).
        """
        if not filters:
            return None
        clauses: List[Dict[str, Any]] = []
        for key, expected in filters.items():
            prefix = self.flatten_list_fields.get(key)
            if prefix is not None:
                values = expected if isinstance(expected, list) else [expected]
                flag_clauses = [{f"{prefix}{v}": True} for v in values]
                if len(flag_clauses) == 1:
                    clauses.append(flag_clauses[0])
                else:
                    clauses.append({"$or": flag_clauses})
            else:
                if isinstance(expected, list):
                    clauses.append({key: {"$in": list(expected)}})
                else:
                    clauses.append({key: expected})
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    # ----------------------------- introspection -----------------------------

    def count(self) -> int:
        return self.collection.count()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "DenseRetriever":
        """Construct from a parsed retrieval.yaml dict."""
        dense = cfg["dense"]
        vs = cfg["vector_store"]
        return cls(
            model_name=dense["model_name"],
            persist_dir=vs["persist_dir"],
            collection_name=vs["collection_name"],
            distance=vs.get("distance", "cosine"),
            device=dense.get("device", "auto"),
            batch_size=dense.get("batch_size", 32),
            normalize_embeddings=dense.get("normalize_embeddings", True),
            max_seq_length=dense.get("max_seq_length", 256),
            segment_input=dense.get("segment_input", True),
            flatten_list_fields=cfg.get("flatten_list_fields", {}),
            hnsw_construction_ef=vs.get("hnsw_construction_ef", 200),
            hnsw_M=vs.get("hnsw_M", 32),
        )
