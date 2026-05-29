"""Traffic Law Assistant — Streamlit chat UI.

Run from the repo root with:

    streamlit run app/streamlit_app.py

The UI is deliberately thin — every business decision (retrieval, reranking,
generation, validation) happens inside `TrafficLawRAG`. This file:

    1. Lazy-loads the pipeline ONCE per session (cached via @st.cache_resource).
    2. Lets the user toggle pipeline parameters in the sidebar (top_k, rerank
       on/off, LLM backend, low-confidence gate). Hits to these widgets rebuild
       the relevant components — cheap, since only knobs change.
    3. Renders chat history with `st.chat_message`.
    4. For every answer, expands an evidence panel showing the Top-3 chunks
       used, their RRF + cross-encoder scores, and the citation validator's
       verdict. THIS panel is what proves transparency to a viva committee.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---- Ensure project root is on sys.path ----
# Streamlit's script runner does NOT add the project root automatically.
# This file lives in  <project>/app/streamlit_app.py  →  parent.parent = project root.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.phase2_hybrid_search.hybrid_search import HybridSearcher
from src.phase3_reranking.cross_encoder_reranker import CrossEncoderReranker
from src.phase4_generation.citation_validator import CitationValidator
from src.phase4_generation.llm_loader import load_llm
from src.phase4_generation.rag_chain import RAGResult, TrafficLawRAG
from src.utils.io import load_yaml
from src.utils import session_manager

# ---- Settings paths ----
settings_yaml = "configs/settings.yaml"
retrieval_yaml = "configs/retrieval.yaml"
generation_yaml = "configs/generation.yaml"

# Load default backend from environment or config
try:
    gen_cfg = load_yaml(generation_yaml)
    default_backend = os.environ.get("LLM_BACKEND") or gen_cfg.get("llm", {}).get("backend", "hf")
    if default_backend == "transformers":
        default_backend = "hf"
except Exception:
    default_backend = "hf"

options = ["hf", "ollama", "openai"]
default_backend_index = options.index(default_backend) if default_backend in options else 0


# ===========================================================================
# Page setup
# ===========================================================================
st.set_page_config(
    page_title="Trợ lý Pháp luật Giao thông Việt Nam",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# Chat state (Multi-Session Persistence)
# ===========================================================================
# Read saved sessions from disk
_sessions = session_manager.list_sessions()
if not _sessions:
    # Bootstrap initial session if no sessions exist
    _init_sid = session_manager.create_session()
    _sessions = session_manager.list_sessions()

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = _sessions[0]["conversation_id"]

if "messages" not in st.session_state:
    try:
        _sess_data = session_manager.load_session(st.session_state.active_session_id)
        st.session_state.messages = _sess_data.get("messages", [])
    except Exception:
        st.session_state.messages = []


# Slim custom CSS — keeps the UI feeling modern without dragging in a heavy theme.
st.markdown(
    """
    <style>
      .small-mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85rem; color: #6b7280; }
      .chunk-card { padding: 0.6rem 0.8rem; border-left: 3px solid #2563eb; background: #f8fafc; border-radius: 4px; margin-bottom: 0.4rem; }
      .pill       { display:inline-block; padding:1px 8px; border-radius:9999px; font-size:0.75rem; margin-right:4px; }
      .pill-ok    { background:#dcfce7; color:#166534; }
      .pill-warn  { background:#fef3c7; color:#92400e; }
      .pill-bad   { background:#fee2e2; color:#991b1b; }
      .pill-info  { background:#e0f2fe; color:#075985; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚦 Trợ lý Pháp luật Giao thông Việt Nam")
st.caption(
    "Hỏi đáp về Nghị định 100/2019/NĐ-CP, Nghị định 123/2021/NĐ-CP và QCVN 41:2019/BGTVT. "
    "Mỗi câu trả lời đều được kiểm chứng trích dẫn và hiển thị nguồn pháp lý gốc."
)


# ===========================================================================
# Pipeline construction (cached for the session)
# ===========================================================================
@st.cache_resource(show_spinner="Đang tải pipeline (BM25 + dense + cross-encoder + LLM) …")
def _load_pipeline(
    settings_yaml: str,
    retrieval_yaml: str,
    generation_yaml: str,
    llm_backend: str,
) -> TrafficLawRAG:
    """Build the full TrafficLawRAG once and cache it.

    Note: only the LLM backend toggle invalidates this cache (it's part of the
    cache key). The cross-encoder on/off and top-k sliders are applied at
    .run() time via a thin wrapper — no rebuild needed.
    """
    return TrafficLawRAG.from_configs(
        settings_yaml=settings_yaml,
        retrieval_yaml=retrieval_yaml,
        generation_yaml=generation_yaml,
        llm_backend_override=llm_backend,
    )


# ===========================================================================
# Sidebar — pipeline controls
# ===========================================================================
with st.sidebar:
    st.header("💬 Cuộc trò chuyện")
    
    # "New Chat" button
    if st.button("➕ Cuộc hội thoại mới", use_container_width=True, type="primary"):
        new_sid = session_manager.create_session()
        st.session_state.active_session_id = new_sid
        st.session_state.messages = []
        st.rerun()

    # List saved sessions from disk
    sessions_list = session_manager.list_sessions()
    
    # Render session buttons
    for s in sessions_list:
        sid = s["conversation_id"]
        title = s["title"]
        # Use an indicator for the active session
        btn_label = f"📌 {title}" if sid == st.session_state.active_session_id else f"💬 {title}"
        
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(btn_label, key=f"sess_{sid}", use_container_width=True):
                st.session_state.active_session_id = sid
                sess_data = session_manager.load_session(sid)
                st.session_state.messages = sess_data.get("messages", [])
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{sid}", help="Xoá cuộc hội thoại này"):
                session_manager.delete_session(sid)
                if sid == st.session_state.active_session_id:
                    st.session_state.pop("active_session_id", None)
                    st.session_state.pop("messages", None)
                st.rerun()

    st.markdown("---")
    st.subheader("⚙️ Cấu hình pipeline")

    # --- LLM backend ---
    llm_backend = st.selectbox(
        "LLM backend",
        options=["hf", "ollama", "openai"],
        index=default_backend_index,
        help=(
            "hf: on-device HuggingFace (4-bit quantized).  "
            "ollama: local HTTP API.  "
            "openai: any OpenAI-compatible endpoint."
        ),
    )

    st.markdown("---")
    st.subheader("Truy hồi (Retrieval)")
    hybrid_top_k = st.slider("Hybrid top-K (Phase 2)", 5, 30, 10, step=1,
                             help="Số chunk lấy từ BM25 + dense + RRF trước khi rerank.")
    final_top_k = st.slider("Final top-K cho LLM (Phase 3)", 1, 10, 3, step=1,
                            help="Số chunk sau cross-encoder sẽ được đưa vào prompt.")

    enable_rerank = st.toggle("Bật Cross-Encoder rerank (Phase 3)", value=True,
                              help="Tắt để so sánh chất lượng khi chỉ dùng hybrid.")

    st.markdown("---")
    st.subheader("Cổng tin cậy")
    use_score_gate = st.toggle("Bật ngưỡng từ chối", value=False,
                               help="Nếu điểm cross-encoder cao nhất dưới ngưỡng → bot từ chối trả lời.")
    min_score = st.slider("Ngưỡng cross-encoder", -10.0, 10.0, 0.0, step=0.1,
                          disabled=not use_score_gate)

    st.markdown("---")
    st.subheader("Bộ lọc metadata (tuỳ chọn)")
    filter_vehicle = st.selectbox(
        "Loại phương tiện",
        options=["(tất cả)", "o_to", "xe_may", "xe_dap"],
        index=0,
    )

    st.markdown("---")
    if st.button("🗑️ Xoá toàn bộ cuộc trò chuyện này", use_container_width=True):
        session_manager.delete_session(st.session_state.active_session_id)
        st.session_state.pop("active_session_id", None)
        st.session_state.pop("messages", None)
        st.rerun()

    # (Paths are defined globally at the top of the file)

    # Try to surface a quick "system health" indicator.
    with st.expander("🩺 Trạng thái hệ thống", expanded=False):
        try:
            cfg = load_yaml(retrieval_yaml)
            st.write(f"• BM25 variant: `{cfg['bm25']['variant']}`")
            st.write(f"• Dense model: `{cfg['dense']['model_name']}`")
            st.write(f"• Cross-encoder: `{cfg['cross_encoder']['model_name']}`")
        except Exception as e:
            st.warning(f"Không đọc được cấu hình: {e}")


# ===========================================================================
# Load pipeline (cached) + apply sidebar overrides at call time
# ===========================================================================
try:
    rag: TrafficLawRAG = _load_pipeline(
        settings_yaml=settings_yaml,
        retrieval_yaml=retrieval_yaml,
        generation_yaml=generation_yaml,
        llm_backend=llm_backend,
    )
except FileNotFoundError as e:
    st.error(
        f"Không tìm thấy index hoặc file dữ liệu: {e}\n\n"
        "Bạn cần chạy Phase 1 (chunking) và Phase 2 (build_indexes.py) trước khi mở UI."
    )
    st.stop()
except Exception as e:
    st.error(f"Lỗi khi khởi tạo pipeline: {e}")
    st.stop()


def _apply_sidebar_overrides() -> None:
    """Patch the cached pipeline with the user's current sidebar choices."""
    rag.hybrid_top_k = int(hybrid_top_k)
    rag.final_top_k = int(final_top_k)
    rag.min_cross_encoder_score = float(min_score) if use_score_gate else None
    # Cross-encoder toggle: a stub reranker that bypasses scoring keeps the
    # first `final_top_k` candidates from the hybrid step, preserving order.
    if enable_rerank:
        rag.reranker = _real_reranker
    else:
        rag.reranker = _passthrough_reranker


@st.cache_resource
def _build_passthrough_reranker() -> CrossEncoderReranker:
    """A reranker that returns the input order unchanged (zero scores)."""
    return CrossEncoderReranker(predictor=lambda pairs: [0.0] * len(pairs),
                                segment_input=False)


_real_reranker = rag.reranker
_passthrough_reranker = _build_passthrough_reranker()






# ===========================================================================
# Render history
# ===========================================================================
def _validation_badge(result: RAGResult) -> str:
    if result.refused_due_to_low_confidence:
        return '<span class="pill pill-info">↪ Từ chối (không đủ căn cứ)</span>'
    if result.validation is None:
        return ""
    if result.validation.is_valid:
        n_sup = len(result.validation.supported_citations)
        return f'<span class="pill pill-ok">✓ Trích dẫn hợp lệ ({n_sup})</span>'
    fails = ", ".join(result.validation.failures) or "không xác định"
    return f'<span class="pill pill-bad">✗ Cảnh báo: {fails}</span>'


def _render_sources(result: RAGResult) -> None:
    """The transparency panel — Top-K chunks + scores. The slide gold."""
    if not result.sources:
        st.info("Không có chunk nào được sử dụng.")
        return

    for i, c in enumerate(result.sources, start=1):
        with st.container(border=True):
            cit_bits: List[str] = []
            if c.get("diem"):
                cit_bits.append(f"Điểm {c['diem']}")
            if c.get("khoan"):
                cit_bits.append(f"Khoản {c['khoan']}")
            if c.get("dieu"):
                cit_bits.append(f"Điều {c['dieu']}")
            doc_short = c.get("doc_short", "?")
            cit_label = ", ".join(cit_bits) if cit_bits else "—"

            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown(f"**#{i}.  {cit_label} ({doc_short})**")
                if c.get("dieu_title"):
                    st.caption(c["dieu_title"])
            with col2:
                pills: List[str] = []
                if c.get("rrf_score") is not None:
                    pills.append(f'<span class="pill pill-info">RRF {c["rrf_score"]:.4f}</span>')
                if c.get("cross_encoder_score") is not None:
                    pills.append(f'<span class="pill pill-ok">CE {c["cross_encoder_score"]:.3f}</span>')
                if c.get("retrievers"):
                    pills.append(f'<span class="pill pill-warn">{c["retrievers"]}</span>')
                if c.get("bm25_rank") is not None and c.get("dense_rank") is not None:
                    pills.append(
                        f'<span class="small-mono">BM25 #{c["bm25_rank"]} · Dense #{c["dense_rank"]}</span>'
                    )
                st.markdown(" ".join(pills), unsafe_allow_html=True)

            body = c.get("linear_form") if c.get("chunk_type") == "table" else c.get("text", "")
            st.markdown(body)


def _render_timings(result: RAGResult) -> None:
    if not result.timings:
        return
    cols = st.columns(len(result.timings))
    for col, (k, v) in zip(cols, result.timings.items()):
        col.metric(label=f"⏱ {k}", value=f"{v*1000:.0f} ms")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        result: Optional[RAGResult] = msg.get("result")
        if result is not None:
            st.markdown(_validation_badge(result), unsafe_allow_html=True)
            with st.expander("📑 Nguồn trích dẫn / Căn cứ pháp lý (Top-K chunks)", expanded=False):
                _render_sources(result)
                st.markdown("---")
                _render_timings(result)
                if result.usage:
                    st.caption(
                        f"Token usage — prompt: {result.usage.get('prompt_tokens', '—')}, "
                        f"completion: {result.usage.get('completion_tokens', '—')}"
                    )


# ===========================================================================
# Input
# ===========================================================================
user_input = st.chat_input(
    "Hỏi về luật giao thông — ví dụ: Lái ô tô vượt đèn đỏ bị phạt bao nhiêu?"
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "result": None})
    with st.chat_message("user"):
        st.markdown(user_input)

    _apply_sidebar_overrides()

    filters: Optional[Dict[str, Any]] = None
    if filter_vehicle != "(tất cả)":
        filters = {"vehicle_type": filter_vehicle}

    # Extract historical turns (excluding the user query that was just appended)
    past_history = st.session_state.messages[:-1]

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu và tổng hợp …"):
            t0 = time.perf_counter()
            try:
                result = rag.run(user_input, chat_history=past_history, filters=filters)
            except Exception as e:
                st.error(f"Lỗi: {e}")
                st.stop()
            elapsed = time.perf_counter() - t0

        st.markdown(result.answer or "_(không có câu trả lời)_")
        st.markdown(_validation_badge(result), unsafe_allow_html=True)

        with st.expander("📑 Nguồn trích dẫn / Căn cứ pháp lý (Top-K chunks)", expanded=True):
            _render_sources(result)
            st.markdown("---")
            _render_timings(result)
            st.caption(f"Tổng thời gian: {elapsed*1000:.0f} ms")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.answer,
        "result": result,
    })

    # ---- Persist the updated conversation to disk ----
    _current_sess = session_manager.load_session(st.session_state.active_session_id)
    _title = _current_sess.get("title", "Cuộc hội thoại mới")
    
    # Auto-generate title from the first question if it was named "Cuộc hội thoại mới"
    if _title == "Cuộc hội thoại mới" and st.session_state.messages:
        _first_q = st.session_state.messages[0]["content"]
        _title = _first_q[:30] + "..." if len(_first_q) > 30 else _first_q
        _title = _title.replace("\n", " ").strip()

    _session_data = {
        "conversation_id": st.session_state.active_session_id,
        "title": _title,
        "timestamp": _current_sess.get("timestamp", ""),
        "messages": st.session_state.messages,
    }
    session_manager.save_session(st.session_state.active_session_id, _session_data)
    
    # Force rerun to refresh the sidebar title instantly
    st.rerun()
