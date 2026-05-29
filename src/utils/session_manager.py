"""Session Manager — handles file-based persistence for independent conversations.

This allows chat history to be preserved across Streamlit updates and restarts,
creating a ChatGPT-like multitasking chat interface.
Optimized for multi-user web environments to prevent race conditions.
"""
from __future__ import annotations

import json
import uuid
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from loguru import logger

# Import Streamlit defensively
try:
    import streamlit as st
except ImportError:
    st = None

from src.phase4_generation.citation_validator import (
    ExtractedCitation,
    ValidationReport,
)

HISTORY_DIR = Path("data/chat_history")


def _get_history_dir() -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORY_DIR


def get_session_file_path(session_id: str) -> Path:
    """Helper to generate a clean, safe path under data/chat_history/session_{session_id}.json."""
    # Ensure we don't double-prefix
    clean_id = session_id
    if clean_id.startswith("session_"):
        clean_id = clean_id[len("session_"):]
    return _get_history_dir() / f"session_{clean_id}.json"


# ---------------------------------------------------------------------------
# Serialization Helpers (dataclass/custom classes -> dict)
# ---------------------------------------------------------------------------
def to_json_serializable(obj: Any) -> Any:
    """Recursively convert custom objects/dataclasses into JSON-friendly formats."""
    if isinstance(obj, list):
        return [to_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return to_json_serializable(asdict(obj))
    if hasattr(obj, "__dict__"):
        return to_json_serializable(obj.__dict__)
    if isinstance(obj, tuple):
        return list(obj)
    return obj


# ---------------------------------------------------------------------------
# Deserialization Helpers (dict -> dataclass/custom classes)
# ---------------------------------------------------------------------------
def dict_to_citation(c_d: Dict[str, Any]) -> ExtractedCitation | None:
    if not c_d:
        return None
    nd = c_d.get("nghi_dinh")
    if nd and isinstance(nd, list):
        nd = tuple(nd)
    return ExtractedCitation(
        dieu=c_d.get("dieu"),
        khoan=c_d.get("khoan"),
        diem=c_d.get("diem"),
        nghi_dinh=nd,
        qcvn=c_d.get("qcvn"),
    )


def dict_to_rag_result(d: Dict[str, Any]) -> RAGResult | None:
    if not d:
        return None
    from src.phase4_generation.rag_chain import RAGResult
    val_d = d.get("validation")
    validation = None
    if val_d:
        validation = ValidationReport(
            is_valid=val_d.get("is_valid", True),
            citations_found=[dict_to_citation(c) for c in val_d.get("citations_found", [])],
            supported_citations=[
                dict_to_citation(c) for c in val_d.get("supported_citations", [])
            ],
            hallucinated_citations=[
                dict_to_citation(c) for c in val_d.get("hallucinated_citations", [])
            ],
            failures=val_d.get("failures", []),
            forbidden_phrases_hit=val_d.get("forbidden_phrases_hit", []),
        )
    return RAGResult(
        query=d.get("query", ""),
        answer=d.get("answer", ""),
        thought=d.get("thought", ""),
        sources=d.get("sources", []),
        validation=validation,
        timings=d.get("timings", {}),
        usage=d.get("usage", {}),
        messages=d.get("messages", []),
        refused_due_to_low_confidence=d.get("refused_due_to_low_confidence", False),
    )


# ---------------------------------------------------------------------------
# Public Session API
# ---------------------------------------------------------------------------
def get_or_create_streamlit_session_id() -> str:
    """Retrieves the active session ID from st.session_state, or registers a new unique UUID.
    
    Guarantees that every user accessing the web application receives an isolated session.
    """
    if st is not None and hasattr(st, "session_state"):
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
            logger.info(f"[session_manager] Initialized new Streamlit Session State ID: {st.session_state.session_id}")
        return st.session_state.session_id
    return str(uuid.uuid4())


def create_session(first_query: str = "") -> str:
    """Create a new independent conversation session and write it to disk.

    Args:
        first_query: Optional first user question to generate a clean title.

    Returns:
        The unique string UUID of the session.
    """
    session_id = get_or_create_streamlit_session_id()
    title = (
        first_query[:35] + "..."
        if len(first_query) > 35
        else (first_query or "Cuộc hội thoại mới")
    )
    title = title.replace("\n", " ").strip()

    data = {
        "conversation_id": session_id,
        "title": title,
        "timestamp": datetime.now().isoformat(),
        "messages": [],
    }
    save_session(session_id, data)
    return session_id


def save_session(session_id: str, data: Dict[str, Any]) -> None:
    """Serialize the conversation session data to a JSON file.

    Args:
        session_id: UUID of the session.
        data: The session data dictionary containing 'title', 'timestamp', and 'messages'.
    """
    path = get_session_file_path(session_id)
    serializable_data = to_json_serializable(data)
    
    # Thread-safe write to prevent multi-user race conditions (writing to tmp and renaming)
    temp_path = path.with_suffix(".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        temp_path.replace(path)
    except Exception as e:
        logger.error(f"[session_manager] Failed thread-safe session save for {session_id}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise e


def load_session(session_id: str) -> Dict[str, Any]:
    """Read a conversation session file from disk and reconstruct its dataclasses.

    Args:
        session_id: UUID of the session.

    Returns:
        The session data dictionary.
    """
    path = get_session_file_path(session_id)
    if not path.exists():
        # Fallback to legacy path format if present
        legacy_path = _get_history_dir() / f"{session_id}.json"
        if legacy_path.exists():
            path = legacy_path
        else:
            raise FileNotFoundError(f"Session {session_id} not found on disk at {path}.")
            
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Reconstruct RAGResult objects in messages list
        messages = []
        for msg in data.get("messages", []):
            res_dict = msg.get("result")
            result = dict_to_rag_result(res_dict) if res_dict else None
            messages.append({
                "role": msg.get("role"),
                "content": msg.get("content"),
                "result": result,
            })
        data["messages"] = messages
        return data


def list_sessions() -> List[Dict[str, Any]]:
    """List all saved conversation sessions on disk, sorted newest first.

    Returns:
        A list of dictionaries containing 'conversation_id', 'title', and 'timestamp'.
    """
    sessions = []
    # Search for files starting with 'session_' or matching legacy '.json' files
    for file in _get_history_dir().glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                d = json.load(f)
                conv_id = d.get("conversation_id")
                if conv_id:
                    sessions.append({
                        "conversation_id": conv_id,
                        "title": d.get("title", "Hội thoại không tiêu đề"),
                        "timestamp": d.get("timestamp", ""),
                    })
        except Exception:
            continue
    # Sort sessions by timestamp desc (newest first)
    sessions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return sessions


def delete_session(session_id: str) -> None:
    """Delete a conversation session file from disk.

    Args:
        session_id: UUID of the session to delete.
    """
    path = get_session_file_path(session_id)
    if path.exists():
        path.unlink()
    # Delete legacy file if it exists
    legacy_path = _get_history_dir() / f"{session_id}.json"
    if legacy_path.exists():
        legacy_path.unlink()
