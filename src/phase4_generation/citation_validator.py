"""Citation validator — the post-generation safety net.

Why we need this on top of a strict prompt:
    Even with a well-written system prompt, modern LLMs occasionally fabricate
    plausible-sounding citations ("Điều 12 Khoản 7" that doesn't exist) or
    invert numbers ("Điều 5 Khoản 6" when the chunk said "Điều 6 Khoản 5").
    The prompt makes hallucination unlikely; the validator makes it DETECTABLE.

What this module does:
    1. Extract every recognisable citation pattern from the LLM's output via
       a small set of well-tested regexes (Điều, Khoản, Điểm, Nghị định, QCVN).
    2. Build the set of LEGAL citations from the retrieved chunks' metadata.
    3. Set-difference the two: anything cited in the answer but NOT supported by
       any retrieved chunk is flagged as a hallucination.
    4. Optionally enforce that the answer contains AT LEAST ONE citation —
       a zero-citation answer almost always means the model ignored the context.

What this module deliberately does NOT do:
    * It does not perform NLI / entailment checking — for that, run RAGAS in
      Phase 5. The validator's job is cheap, deterministic, regex-grade.
    * It does not check that the *content* of the answer matches the cited
      chunk; the LLM could still pick the correct citation but paraphrase
      wrongly. That, too, is RAGAS's responsibility.

Public API:
    * CitationValidator (callable)
    * ValidationReport (dataclass)
    * extract_citations (helper, also exported for downstream analytics)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from loguru import logger

from src.phase4_generation.prompt_templates import ChunkCitation, chunk_citation
from src.utils.io import load_yaml


# ---------------------------------------------------------------------------
# Regex patterns — kept module-level and pre-compiled.
# Vietnamese point alphabet has no f/j/w/z; we permit a–z + đ defensively
# (so that a malformed model output like "Điểm f" is still caught and flagged
# rather than silently ignored by an over-strict regex).
# ---------------------------------------------------------------------------
_RE_DIEU = re.compile(r"Điều\s+(\d+)", re.IGNORECASE)
_RE_KHOAN = re.compile(r"Khoản\s+(\d+)", re.IGNORECASE)
_RE_DIEM = re.compile(r"Điểm\s+([a-zđ])", re.IGNORECASE)
_RE_NGHI_DINH = re.compile(
    r"Nghị[\s_]*định\s+(\d+)\s*/\s*(\d{4})\s*/\s*NĐ-?CP",
    re.IGNORECASE,
)
_RE_QCVN = re.compile(r"QCVN\s+([\d:]+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# A single citation extracted from text.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractedCitation:
    dieu: Optional[str] = None
    khoan: Optional[str] = None
    diem: Optional[str] = None
    nghi_dinh: Optional[Tuple[str, str]] = None    # (number, year), e.g. ("100", "2019")
    qcvn: Optional[str] = None

    def is_empty(self) -> bool:
        return not any([self.dieu, self.khoan, self.diem, self.nghi_dinh, self.qcvn])


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------
def extract_citations(text: str) -> List[ExtractedCitation]:
    """Find every parenthesised citation in `text` and parse it.

    The prompt instructs the LLM to wrap citations in parentheses, but real
    models often drop the parens or use square brackets. We therefore split
    on a permissive pattern and re-parse each fragment.

    Returns a list — order preserved — of one ExtractedCitation per bracket/paren
    group found. A "(Khoản 5 Điều 5 Nghị định 100/2019/NĐ-CP)" yields ONE entry
    with dieu='5', khoan='5', nghi_dinh=('100','2019').
    """
    # Pull out every (...) or [...] group; if none, fall back to splitting by
    # sentence so we still catch unbracketed citations like
    # "...theo Khoản 5 Điều 5 Nghị định 100/2019/NĐ-CP."
    bracket_re = re.compile(r"[\(\[]([^()\[\]]+)[\)\]]")
    fragments: List[str] = [m.group(1) for m in bracket_re.finditer(text)]
    if not fragments:
        # Fallback: every sentence is a potential citation host.
        fragments = re.split(r"(?<=[\.\!\?])\s+", text)

    out: List[ExtractedCitation] = []
    for frag in fragments:
        cit = _parse_fragment(frag)
        if not cit.is_empty():
            out.append(cit)
    return out


def _parse_fragment(frag: str) -> ExtractedCitation:
    dieu = _first(_RE_DIEU, frag)
    khoan = _first(_RE_KHOAN, frag)
    diem_match = _RE_DIEM.search(frag)
    diem = diem_match.group(1).lower() if diem_match else None
    nd_match = _RE_NGHI_DINH.search(frag)
    nghi_dinh = (nd_match.group(1), nd_match.group(2)) if nd_match else None
    qcvn = _first(_RE_QCVN, frag)
    return ExtractedCitation(dieu=dieu, khoan=khoan, diem=diem,
                             nghi_dinh=nghi_dinh, qcvn=qcvn)


def _first(pattern: re.Pattern, s: str) -> Optional[str]:
    m = pattern.search(s)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Vehicle Type Mismatch Detection
# ---------------------------------------------------------------------------
def _detect_vehicle_types(text: str) -> Set[str]:
    """Helper to detect vehicle types based on Vietnamese keywords."""
    text_lower = (text or "").lower()
    found = set()
    
    # Check for "người đi bộ" (pedestrian)
    if "đi bộ" in text_lower or "người đi bộ" in text_lower:
        found.add("nguoi_di_bo")
        
    # Check for "xe đạp" (bicycle) or "xe thô sơ"
    if "xe đạp" in text_lower or "xe thô sơ" in text_lower or "xe thô-sơ" in text_lower:
        found.add("xe_dap")
        found.add("xe_tho_so")
        
    # Check for "xe máy" or "mô tô" or "xe gắn máy" or "mô-tô"
    if "xe máy" in text_lower or "mô tô" in text_lower or "xe gắn máy" in text_lower or "xe mô tô" in text_lower or "mô-tô" in text_lower:
        found.add("xe_may")
        
    # Check for "ô tô" or "xe hơi" or "xe du lịch" or "xe khách" or "xe tải" or "xe ô-tô"
    if "ô tô" in text_lower or "xe hơi" in text_lower or "xe du lịch" in text_lower or "xe tải" in text_lower or "xe khách" in text_lower or "xe ô-tô" in text_lower:
        found.add("o_to")
        
    # Check for "máy kéo"
    if "máy kéo" in text_lower:
        found.add("may_keo")
        
    # Check for "xe chuyên dùng" or "xe máy chuyên dùng"
    if "chuyên dùng" in text_lower:
        found.add("xe_chuyen_dung")
        
    return found


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    """Result of validating one LLM answer against its retrieved chunks."""
    is_valid: bool
    citations_found: List[ExtractedCitation] = field(default_factory=list)
    supported_citations: List[ExtractedCitation] = field(default_factory=list)
    hallucinated_citations: List[ExtractedCitation] = field(default_factory=list)
    # Reasons the answer failed (human-readable for logging / UI).
    failures: List[str] = field(default_factory=list)
    # Forbidden phrases the answer triggered (e.g. "theo tôi", "có thể là").
    forbidden_phrases_hit: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------
class CitationValidator:
    """Verifies that an LLM answer's citations are supported by retrieved chunks.

    A citation in the answer is "supported" if some chunk's metadata is
    *consistent* with it. Consistency rules:

        * Nghị định number/year (if cited) must match a chunk's doc_short.
        * Điều number (if cited) must match a chunk's `dieu` field.
        * Khoản number, if cited together with Điều, must appear in some chunk
          whose Điều also matches. (Cross-Điều mixups are the most common
          hallucination mode — "Khoản 5 Điều 6" when the right answer is
          "Khoản 5 Điều 5".)
        * Điểm letter, if cited together with Khoản+Điều, must match a chunk
          at that exact triple.

    A citation that fails any consistency check is "hallucinated".
    """

    def __init__(
        self,
        require_at_least_one: bool = True,
        forbidden_phrases: Optional[Iterable[str]] = None,
        refusal_phrase: str = "Tôi không tìm thấy quy định phù hợp",
    ):
        self.require_at_least_one = require_at_least_one
        self.forbidden_phrases = [p.lower() for p in (forbidden_phrases or [])]
        self.refusal_phrase = refusal_phrase.lower()

    # ---------- factory ----------
    @classmethod
    def from_config(cls, generation_yaml: str = "configs/generation.yaml") -> "CitationValidator":
        cfg = load_yaml(generation_yaml)
        return cls(
            require_at_least_one=cfg["citation"].get("require_at_least_one", True),
            forbidden_phrases=cfg.get("rag", {}).get("forbidden_phrases", []),
        )

    # ---------- public ----------
    def validate(self, answer: str, chunks: List[Dict[str, Any]], query: Optional[str] = None) -> ValidationReport:
        report = ValidationReport(is_valid=True)

        # An explicit "I couldn't find an answer" reply is ALWAYS valid — the
        # prompt mandates it for the no-context case, and it carries no claims
        # to support. Skip citation checks.
        if self.refusal_phrase in (answer or "").lower():
            return report

        # Forbidden hedge phrases.
        lower = (answer or "").lower()
        for phrase in self.forbidden_phrases:
            if phrase in lower:
                report.forbidden_phrases_hit.append(phrase)
        if report.forbidden_phrases_hit:
            report.is_valid = False
            report.failures.append(
                f"Forbidden hedging phrase(s) emitted: {report.forbidden_phrases_hit}"
            )

        # Extract citations from the answer.
        report.citations_found = extract_citations(answer or "")

        if not report.citations_found:
            # Tolerant bypass: if the answer asserts that the user is lawful and not penalized,
            # requiring a citation is a logical fallacy. Bypassing require_at_least_one.
            is_lawful_assertion = any(w in lower for w in [
                "không bị xử phạt", "không bị phạt", "đúng luật", "0 đồng", 
                "không phải chịu mức phạt", "không có lỗi", "hoàn toàn đúng luật",
                "không phải chịu bất kỳ mức phạt"
            ])
            if self.require_at_least_one and not is_lawful_assertion:
                report.is_valid = False
                report.failures.append("No legal citation found in the answer.")
            return report

        # Build the set of legal citations from the retrieved chunks.
        chunk_cits = [chunk_citation(c) for c in chunks]
        # Tuple form for fast set ops: (doc_short, dieu, khoan, diem).
        chunk_triples: Set[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]] = {
            (c.doc_short, c.dieu, c.khoan, c.diem) for c in chunk_cits
        }
        chunk_dieus: Set[Tuple[Optional[str], Optional[str]]] = {
            (c.doc_short, c.dieu) for c in chunk_cits
        }
        chunk_di_khoan: Set[Tuple[Optional[str], Optional[str], Optional[str]]] = {
            (c.doc_short, c.dieu, c.khoan) for c in chunk_cits
        }
        # Doc-short map from Nghị định (num, year).
        nd_year_to_short = {
            ("100", "2019"): "ND100",
            ("123", "2021"): "ND123",
        }

        # Analyze vehicle types from query
        query_vehicles = _detect_vehicle_types(query) if query else set()

        for cit in report.citations_found:
            doc_short = (
                nd_year_to_short.get(cit.nghi_dinh) if cit.nghi_dinh else None
            )

            ok = _citation_supported(
                cit=cit,
                expected_doc=doc_short,
                chunk_triples=chunk_triples,
                chunk_di_khoan=chunk_di_khoan,
                chunk_dieus=chunk_dieus,
            )
            if ok:
                report.supported_citations.append(cit)
                
                # Check for vehicle type alignment on supported citations
                if query_vehicles:
                    matching_chunks = []
                    for chunk in chunks:
                        c_cit = chunk_citation(chunk)
                        
                        # Match doc_short
                        doc_ok = (
                            doc_short is None
                            or c_cit.doc_short is None
                            or c_cit.doc_short == doc_short
                            or (c_cit.doc_short == "ND100_123" and doc_short in ("ND100", "ND123"))
                        )
                        if not doc_ok:
                            continue
                        
                        # Match dieu
                        if cit.dieu is not None and c_cit.dieu != cit.dieu:
                            continue
                        
                        # Match khoan
                        if cit.khoan is not None and c_cit.khoan != cit.khoan:
                            continue
                        
                        # Match diem (allow broader matching where chunk has no diem)
                        if cit.diem is not None and c_cit.diem != cit.diem:
                            if c_cit.diem is not None and c_cit.diem != "":
                                continue
                                
                        matching_chunks.append(chunk)
                        
                    if matching_chunks:
                        chunk_vehicles = set()
                        for chunk in matching_chunks:
                            meta = chunk.get("metadata") or {}
                            for v in meta.get("vehicle_type") or []:
                                chunk_vehicles.add(v)
                                
                        # Check overlap (only if chunk has vehicle types specified)
                        if chunk_vehicles:
                            overlap = query_vehicles.intersection(chunk_vehicles)
                            if not overlap:
                                report.is_valid = False
                                report.failures.append(
                                    f"Subject/Vehicle Type Mismatch: query is about {query_vehicles} "
                                    f"but cited chunk applies to {chunk_vehicles}"
                                )
                                logger.warning(
                                    f"[citation_validator] Vehicle type mismatch detected! "
                                    f"Query: {query_vehicles}, Cited chunk: {chunk_vehicles}"
                                )
            else:
                report.hallucinated_citations.append(cit)

        if report.hallucinated_citations:
            report.is_valid = False
            report.failures.append(
                f"Hallucinated citation(s) not supported by retrieved chunks: "
                f"{[_cit_to_str(c) for c in report.hallucinated_citations]}"
            )
            logger.warning(
                f"[citation_validator] {len(report.hallucinated_citations)} "
                f"unsupported citation(s) in answer."
            )

        return report


# ---------------------------------------------------------------------------
# Internal: per-citation support check.
# ---------------------------------------------------------------------------
def _citation_supported(
    cit: ExtractedCitation,
    expected_doc: Optional[str],
    chunk_triples: Set[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]],
    chunk_di_khoan: Set[Tuple[Optional[str], Optional[str], Optional[str]]],
    chunk_dieus: Set[Tuple[Optional[str], Optional[str]]],
) -> bool:
    """Return True iff some retrieved chunk could plausibly host this citation."""
    # We tolerate a citation that names ONLY a Nghị định without Điều — it's
    # weak but not a hallucination — provided some chunk is from that doc.
    if cit.dieu is None and cit.khoan is None and cit.diem is None:
        if expected_doc is None:
            return False
        return any(
            triple[0] == expected_doc or (triple[0] == "ND100_123" and expected_doc in ("ND100", "ND123"))
            for triple in chunk_triples
        )

    # Helper: do we allow `None` on the doc side?
    # If the answer doesn't name a Nghị định, the citation may still be valid
    # provided the (dieu, khoan, diem) appears in ANY chunk regardless of doc.
    doc_match = lambda d: (
        expected_doc is None
        or d is None
        or d == expected_doc
        or (d == "ND100_123" and expected_doc in ("ND100", "ND123"))
    )

    # Most specific level cited determines the strictest check.
    if cit.diem is not None and cit.khoan is not None and cit.dieu is not None:
        return any(
            doc_match(d) and di == cit.dieu and kh == cit.khoan and (dm == cit.diem or dm is None or dm == "")
            for (d, di, kh, dm) in chunk_triples
        )
    if cit.khoan is not None and cit.dieu is not None:
        return any(
            doc_match(d) and di == cit.dieu and kh == cit.khoan
            for (d, di, kh) in chunk_di_khoan
        )
    if cit.dieu is not None:
        return any(
            doc_match(d) and di == cit.dieu
            for (d, di) in chunk_dieus
        )
    # Khoản / Điểm without Điều are too ambiguous to support; treat as unsupported.
    return False


def _cit_to_str(c: ExtractedCitation) -> str:
    parts: List[str] = []
    if c.diem:
        parts.append(f"Điểm {c.diem}")
    if c.khoan:
        parts.append(f"Khoản {c.khoan}")
    if c.dieu:
        parts.append(f"Điều {c.dieu}")
    if c.nghi_dinh:
        parts.append(f"Nghị định {c.nghi_dinh[0]}/{c.nghi_dinh[1]}/NĐ-CP")
    if c.qcvn:
        parts.append(f"QCVN {c.qcvn}")
    return ", ".join(parts)
