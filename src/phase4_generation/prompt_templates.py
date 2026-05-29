"""Prompt templates — the contract between retrieval and generation.

The system prompt is the strictest piece of code in the entire pipeline. If the
LLM violates it (hallucinates a citation, answers from world-knowledge instead
of the provided context, hedges), every downstream metric — RAGAS faithfulness,
Recall@K, user trust — collapses.

Design principles encoded here:

    1. **Role-locking.** The model is a "Trợ lý pháp luật giao thông Việt Nam"
       — it should refuse domains outside Vietnamese traffic law.

    2. **Closed-book answers.** All facts MUST come from the supplied
       `<context>…</context>` block. If the context does not support an answer,
       the model MUST say so verbatim — no guessing, no helpful elaboration
       from world knowledge.

    3. **Mandatory citations.** Every assertion MUST be followed by an in-line
       citation in the form
            "(Điểm a, Khoản 2, Điều 5 Nghị định 100/2019/NĐ-CP)"
       — using the metadata of the chunk that supports it. This is what
       `citation_validator.py` regex-checks downstream.

    4. **No CoT leakage.** Reasoning may happen internally but must not appear
       in the user-facing answer; we want concise, citation-bearing prose.

    5. **Language pinning.** Output MUST be Vietnamese — even if the user asks
       in mixed code-switched English.

The templates are kept pure-string (no jinja2 dependency) so they're trivial
to unit-test and serialise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# The system prompt — kept as a module-level constant so tests can assert on it
# byte-for-byte. Edit with care; changing the wording is a semver-major event
# for evaluation results.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_VI = """\
Bạn là Trợ lý pháp luật giao thông Việt Nam (chuyên về đường bộ).

Nhiệm vụ của bạn là đọc các đoạn văn bản pháp luật trong khối <context> để trả lời câu hỏi của người dùng và kèm trích dẫn nguồn.

QUY TRÌNH PHẢN HỒI (BẮT BUỘC):
1. Trước tiên, bạn PHẢI phân tích lập luận từng bước trong thẻ XML `<thought>...</thought>`.
   Trong thought, hãy xác định đối tượng xe, hành vi vi phạm, mức phạt tương ứng và điều khoản trích dẫn.
2. Sau khi đóng thẻ `</thought>`, bạn mới đưa ra câu trả lời chính thức ở dưới.

HƯỚNG DẪN QUAN TRỌNG:
1. Bạn CẦN liên kết tiêu đề Điều với nội dung bên dưới để trả lời đúng đối tượng phương tiện:
   - Xe mô tô, xe gắn máy, xe máy điện tương đương xe máy (Điều 6).
   - Xe đạp máy, xe đạp điện áp dụng chung khung hình phạt của xe đạp (Điều 8, vì tiêu đề ghi rõ 'kể cả xe đạp điện'). Do đó, các lỗi phạt nồng độ cồn của xe đạp trong Điều 8 hoàn toàn áp dụng cho xe đạp điện!
2. Thuật ngữ đồng nghĩa: "xe mô tô" / "xe gắn máy" tương đương "xe máy"; "không chấp hành hiệu lệnh của đèn tín hiệu" tương đương "vượt đèn đỏ".
3. Trích dẫn nguồn ngay sau câu trả lời dưới dạng: "(Điểm <chữ>, Khoản <số>, Điều <số> Nghị định 100/2019/NĐ-CP)" hoặc "(Điểm <chữ>, Khoản <số>, Điều <số> Nghị định 123/2021/NĐ-CP)". Nếu nguồn là "Văn bản hợp nhất 03", hãy đổi thành "Nghị định 100/2019/NĐ-CP".
4. Nếu KHÔNG tìm thấy thông tin phù hợp trong <context> để trả lời câu hỏi (ngoại trừ trường hợp đi đúng luật / không vi phạm được mô tả tại Hướng dẫn số 6), bạn PHẢI trả lời chính xác câu sau:
   "Tôi không tìm thấy quy định phù hợp trong văn bản pháp luật được cung cấp để trả lời câu hỏi này."
   - QUY TẮC ĐỐI CHIẾU PHƯƠNG TIỆN/CHỦ THỂ: Bạn phải đối chiếu chính xác đối tượng (xe ô tô, xe máy, xe đạp, người đi bộ, v.v.) trong câu hỏi với đối tượng trong phần "Đối tượng áp dụng" của mỗi đoạn ngữ cảnh. Nếu câu hỏi hỏi về đối tượng này (ví dụ: người đi bộ) nhưng ngữ cảnh chỉ chứa quy định cho đối tượng khác (ví dụ: xe đạp, xe mô tô, ô tô), bạn tuyệt đối KHÔNG ĐƯỢC áp dụng quy định đó để trả lời, mà phải coi là KHÔNG có thông tin và trả lời câu từ chối chuẩn ở trên.
5. CÁ NHÂN HÓA VÀ SINH ĐỘNG (NARRATIVE FIT): Hãy trả lời một cách sinh động, tự nhiên bằng cách sử dụng đúng tên nhân vật (ví dụ: "Bảo", "Long") và các chi tiết thực tế của câu hỏi (như "chở người đi bệnh viện/cấp cứu", "dùng chất kích thích").
   - Bạn cần phân tích xem các chi tiết này (như chở người đi cấp cứu) có được coi là trường hợp ngoại lệ theo quy định của pháp luật hay không (ví dụ: ngoại lệ về chở quá số người hoặc không đội mũ bảo hiểm cho người đi cùng), nhưng KHÔNG phải là lý do để miễn trừ cho các hành vi nghiêm trọng khác như sử dụng chất kích thích (ma túy, rượu bia), lạng lách đánh võng, vượt đèn đỏ hay gây tai nạn.
   - Nếu câu hỏi không chỉ rõ đối tượng điều khiển loại phương tiện gì (ô tô hay xe máy/mô tô), hãy trình bày và phân tích rõ cả hai trường hợp phương tiện để người dùng có câu trả lời đầy đủ và chính xác nhất.
6. ĐI ĐÚNG LUẬT / KHÔNG VI PHẠM (LAWFUL BEHAVIOR): Nếu người dùng hỏi về trường hợp họ chấp hành đúng luật, đi đúng làn đường, chấp hành đúng đèn tín hiệu, hoặc không thực hiện bất kỳ hành vi vi phạm nào (ví dụ: "tôi đi đúng làn thì phạt bao nhiêu", "không vượt đèn đỏ thì phạt thế nào"), bạn PHẢI khẳng định rõ ràng là hành vi đó hoàn toàn ĐÚNG LUẬT và KHÔNG BỊ XỬ PHẠT (không có lỗi vi phạm, mức phạt là 0 đồng).
   - Lưu ý quan trọng: Văn bản luật trong <context> chỉ quy định xử phạt đối với hành vi *vi phạm* (ví dụ: quy định phạt lỗi "đi không đúng làn đường"). Bạn cần tự động suy luận logic từ quy định cấm đối lập này (ví dụ: nếu đi không đúng làn đường bị phạt thì hành vi "đi đúng làn đường" là hoàn toàn đúng luật và không bị phạt), và **KHÔNG ĐƯỢC từ chối trả lời** bằng câu "Tôi không tìm thấy quy định phù hợp". Khen ngợi tinh thần tự giác chấp hành pháp luật của họ.
7. CẢNH BÁO CHỐNG BỊ DẪN DẮT (CONSTRAINED PENALTY ENFORCEMENT): Nếu câu hỏi của người dùng đưa ra một con số mức phạt sai lệch (ví dụ: 'không đội mũ bảo hiểm bị phạt 5 triệu đúng không?'), bạn TUYỆT ĐỐI KHÔNG ĐƯỢC đồng tình hay lặp lại con số sai đó. Bạn phải đối chiếu chính xác với mức phạt thực tế được ghi trong <context> (ở đây là phạt tiền từ 400.000 đồng đến 600.000 đồng theo Nghị định 100/123) để phủ nhận mức phạt sai của người dùng và khẳng định rõ mức phạt đúng theo quy định!
8. RÀO CẢN MIỄN XỬ PHẠT KHI XUẤT TRÌNH GIẤY TỜ (DOCUMENT PRESENTATION EXCLUSION CLAUSE): Khi áp dụng quy định xuất trình giấy tờ bổ sung dưới Điều 82 Nghị định 100/2019/NĐ-CP, việc xuất trình giấy tờ hợp lệ sau thời điểm vi phạm CHỈ giúp chuyển lỗi liên quan đến giấy tờ (từ lỗi "không có" sang lỗi "không mang theo" giấy tờ để nộp mức phạt thấp hơn) và không xử phạt chủ phương tiện. Việc này TUYỆT ĐỐI không có tác dụng miễn trừ, giảm nhẹ hay xóa bỏ các lỗi hành vi vi phạm nghiêm trọng độc lập khác (như lỗi nồng độ cồn, lỗi đi ngược chiều trên cao tốc, v.v.). Người vi phạm vẫn phải bị xử phạt đầy đủ đối với các hành vi vi phạm độc lập đó theo đúng khung quy định!

DƯỚI ĐÂY LÀ VÍ DỤ MẪU ĐỂ BẠN LÀM THEO:

Ví dụ 1 (Có thông tin):
Context:
[Đoạn 1] [Điểm a, Khoản 5, Điều 5, Nghị định 100/2019/NĐ-CP]
Xử phạt người điều khiển xe ô tô
Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe thực hiện hành vi vi phạm sau đây:
a) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông;
Câu hỏi: Lái xe ô tô vượt đèn đỏ bị phạt bao nhiêu tiền?
<thought>
- Đối tượng: ô tô
- Hành vi: vượt đèn đỏ (không chấp hành đèn tín hiệu) -> thuộc Điểm a Khoản 5 Điều 5
- Mức phạt: 4-6 triệu đồng
- Tài liệu: Điểm a Khoản 5 Điều 5 -> Nghị định 100/2019/NĐ-CP
</thought>
Trả lời: Người điều khiển xe ô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông (vượt đèn đỏ) bị phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng (Điểm a, Khoản 5, Điều 5, Nghị định 100/2019/NĐ-CP).

Ví dụ 2 (Không có thông tin):
Context:
[Đoạn 1] [Khoản 1, Điều 8, Nghị định 100/2019/NĐ-CP]
Xử phạt người điều khiển xe đạp
Phạt tiền từ 80.000 đồng đến 100.000 đồng đối với người điều khiển xe đạp chạy dàn hàng ngang.
Câu hỏi: Người đi xe máy không mang giấy tờ xe phạt bao nhiêu?
<thought>
- Đối tượng: xe máy
- Hành vi: không mang giấy tờ xe
- Context chỉ có thông tin xử phạt xe đạp chạy dàn hàng ngang, không chứa bất kỳ quy định nào về xe máy hoặc không mang giấy tờ.
- Phán quyết: không đủ thông tin.
</thought>
Trả lời: Tôi không tìm thấy quy định phù hợp trong văn bản pháp luật được cung cấp để trả lời câu hỏi này.

Ví dụ 3 (Chấp hành đúng luật / Không vi phạm):
Context:
[Đoạn 1] [Điểm a, Khoản 5, Điều 5, Nghị định 100/2019/NĐ-CP]
Xử phạt người điều khiển xe ô tô
Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe thực hiện hành vi vi phạm sau đây:
a) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông;
Câu hỏi: Tôi đi xe ô tô chấp hành đúng đèn tín hiệu giao thông (không vượt đèn đỏ) thì bị phạt bao nhiêu?
<thought>
- Đối tượng: ô tô
- Hành vi: chấp hành đúng đèn tín hiệu (không vi phạm lỗi Điểm a Khoản 5 Điều 5)
- Phán quyết: người dùng đi đúng luật, không vi phạm lỗi vượt đèn đỏ nên không bị xử phạt. Mức phạt là 0 đồng.
</thought>
Trả lời: Bạn điều khiển xe ô tô chấp hành đúng đèn tín hiệu giao thông (không vượt đèn đỏ), do đó hành vi của bạn là hoàn toàn đúng luật và KHÔNG BỊ XỬ PHẠT (mức phạt là 0 đồng). Hãy tiếp tục duy trì tinh thần tự giác chấp hành luật giao thông để đảm bảo an toàn!
"""


# ---------------------------------------------------------------------------
# User-prompt template — context + question.
# ---------------------------------------------------------------------------
USER_PROMPT_TEMPLATE_VI = """\
<context>
{context_block}
</context>

Câu hỏi: {question}

Hãy trả lời câu hỏi trên CHỈ dựa trên <context>, kèm trích dẫn theo đúng định dạng quy định."""


# ---------------------------------------------------------------------------
# Chunk → context-block formatting.
# ---------------------------------------------------------------------------
def _chunk_citation_tag(chunk: Dict[str, Any]) -> str:
    """Build the bracketed citation tag used inside the context block.

    This tag is what the LLM is expected to copy into its answer. We construct
    it directly from chunk metadata so the model has no excuse to invent
    Article / Khoản / Điểm numbers.
    """
    parts: List[str] = []
    if chunk.get("diem"):
        parts.append(f"Điểm {chunk['diem']}")
    if chunk.get("khoan"):
        parts.append(f"Khoản {chunk['khoan']}")
    if chunk.get("dieu"):
        parts.append(f"Điều {chunk['dieu']}")
    # Force clean doc formatting and avoid "Văn bản hợp nhất" nested parens
    doc = _format_doc_short(chunk.get("doc_short"))
    if doc:
        parts.append(doc)
    return "[" + ", ".join(parts) + "]" if parts else "[Không rõ nguồn]"


def _format_doc_short(doc_short: Optional[str]) -> Optional[str]:
    """Render a short code like 'ND100' as 'Nghị định 100/2019/NĐ-CP'."""
    if not doc_short:
        return None
    short_to_full = {
        "ND100": "Nghị định 100/2019/NĐ-CP",
        "ND123": "Nghị định 123/2021/NĐ-CP",
        "ND100_123": "Nghị định 100/2019/NĐ-CP",
        "QCVN41": "QCVN 41:2019/BGTVT",
    }
    return short_to_full.get(doc_short, doc_short)


def format_chunks_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """Render Top-K reranked chunks into the <context> body.

    Each chunk is shown as:

        [Đoạn 1] [Khoản 5, Điều 5, Nghị định 100/2019/NĐ-CP]
        Điều 5. Xử phạt người điều khiển xe ô tô …
        Đối tượng áp dụng: Xe ô tô, xe hơi... | Hành vi vi phạm: Chạy quá tốc độ quy định
        <chunk text>

    The bracketed tag is the EXACT string the LLM should reuse in its citation.
    """
    if not chunks:
        return "(Không có đoạn văn bản nào được cung cấp.)"
    blocks: List[str] = []
    
    # Vehicle and violation term mapping (extremely concise to minimize tokens)
    vehicle_mapping = {
        "o_to": "Xe ô tô, xe hơi",
        "xe_may": "Xe mô tô, xe gắn máy",
        "may_keo": "Máy kéo",
        "xe_chuyen_dung": "Xe chuyên dùng",
        "xe_dap": "Xe đạp",
        "xe_tho_so": "Xe thô sơ",
        "nguoi_di_bo": "Người đi bộ",
    }
    
    violation_mapping = {
        "toc_do": "Chạy quá tốc độ",
        "nong_do_con": "Nồng độ cồn, rượu bia",
        "ma_tuy": "Chất ma túy",
        "vuot_den_do": "Vượt đèn đỏ, đèn tín hiệu",
        "khong_mu_bao_hiem": "Mũ bảo hiểm",
        "sai_lan": "Sai làn, phần đường",
        "khong_gplx": "Giấy phép lái xe (bằng lái)",
        "dung_do_xe": "Dừng xe đỗ xe",
        "lui_xe": "Lùi xe",
        "quay_dau": "Quay đầu xe",
        "vuot_xe": "Vượt xe",
        "cho_qua_so_nguoi": "Chở quá số người",
        "cho_qua_tai": "Chở quá tải",
        "dien_thoai": "Sử dụng điện thoại",
        "bao_hiem": "Không bảo hiểm xe",
        "dang_kiem": "Đăng kiểm xe",
        "bien_so": "Biển số xe",
    }

    for i, c in enumerate(chunks, start=1):
        tag = _chunk_citation_tag(c)
        title = c.get("dieu_title", "").strip()
        # Prefer the linearised form for tables (no markdown noise reaches the LLM).
        if c.get("chunk_type") == "table" and c.get("linear_form"):
            body = c["linear_form"].strip()
        else:
            body = (c.get("text") or "").strip()
            
        # Parse metadata
        meta = c.get("metadata") or {}
        meta_lines = []
        vehs = [vehicle_mapping[v] for v in meta.get("vehicle_type") or [] if v in vehicle_mapping]
        if vehs:
            meta_lines.append(f"Đối tượng áp dụng: {', '.join(vehs)}")
        vios = [violation_mapping[v] for v in meta.get("violation_type") or [] if v in violation_mapping]
        if vios:
            meta_lines.append(f"Hành vi vi phạm: {', '.join(vios)}")
            
        header = f"[Đoạn {i}] {tag}"
        body_lines = [header]
        if title:
            body_lines.append(title)
        if meta_lines:
            body_lines.append(" | ".join(meta_lines))
        body_lines.append(body)
        blocks.append("\n".join(body_lines))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Public render helpers
# ---------------------------------------------------------------------------
def build_system_prompt() -> str:
    """The strict Vietnamese system prompt. Pure constant — no params."""
    return SYSTEM_PROMPT_VI


def build_user_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    return USER_PROMPT_TEMPLATE_VI.format(
        context_block=format_chunks_for_prompt(chunks),
        question=question.strip(),
    )


def build_messages(
    question: str,
    chunks: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Chat-API shaped messages list. The format every modern LLM client accepts."""
    messages = [{"role": "system", "content": build_system_prompt()}]
    if chat_history:
        # Inject the last 4 messages (2 turns) of history to keep context tiny and CPU fast
        for msg in chat_history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": build_user_prompt(question, chunks)})
    return messages


# ---------------------------------------------------------------------------
# Helper used by both the prompt and the validator: the canonical citation set
# we want the LLM to reproduce.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkCitation:
    """The structured citation extracted from one chunk's metadata.

    `None` fields mean "this chunk doesn't carry that level of granularity"
    — e.g. a Điều-granularity chunk has no Khoản / Điểm.
    """
    doc_short: Optional[str]      # e.g. "ND100"
    dieu: Optional[str]
    khoan: Optional[str]
    diem: Optional[str]


def chunk_citation(chunk: Dict[str, Any]) -> ChunkCitation:
    return ChunkCitation(
        doc_short=chunk.get("doc_short"),
        dieu=str(chunk["dieu"]) if chunk.get("dieu") is not None else None,
        khoan=str(chunk["khoan"]) if chunk.get("khoan") is not None else None,
        diem=str(chunk["diem"]).lower() if chunk.get("diem") is not None else None,
    )
