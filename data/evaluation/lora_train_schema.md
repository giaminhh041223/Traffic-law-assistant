# LoRA Fine-Tuning Dataset Schema

Consumed by `src/phase4_generation/train_lora.py`. The format mirrors the
prompt used at inference time, so the fine-tune literally trains the model on
the format it will be served in.

## File format

JSONL, UTF-8. One JSON object per line.

```json
{
  "system":   "optional — override the default SYSTEM_PROMPT_VI",
  "context":  "string — the retrieved chunk text (with citation tag header)",
  "question": "string — the user's natural-language question",
  "answer":   "string — gold answer with inline citations in the exact format"
}
```

### Field semantics

| Field      | Required | Notes                                                                                       |
|------------|----------|---------------------------------------------------------------------------------------------|
| `system`   | no       | If omitted, the trainer uses `prompt_templates.SYSTEM_PROMPT_VI`. Override only for ablations. |
| `context`  | yes      | Same `[Đoạn 1] [Khoản …, Điều …, Nghị định …]` shape produced by `format_chunks_for_prompt`. |
| `question` | yes      | The user's natural-language Vietnamese question. Do NOT pre-segment.                         |
| `answer`   | yes      | Gold answer. MUST contain at least one citation in the required format. MUST refuse politely with the exact refusal sentence if `context` is insufficient. |

## What makes a good gold `answer`

1. **Citation format is non-negotiable.** Use parentheses, in-line, immediately
   after each legal assertion:
   `"... bị phạt từ 4.000.000 đồng đến 6.000.000 đồng (Điểm a, Khoản 5, Điều 5 Nghị định 100/2019/NĐ-CP)."`
2. **No hedging.** Avoid "theo tôi", "có thể là", "thường thì".
3. **Refusal must be verbatim.** When the context is insufficient, the answer
   must be exactly:
   `"Tôi không tìm thấy quy định phù hợp trong văn bản pháp luật được cung cấp để trả lời câu hỏi này."`
4. **Length.** 1–3 sentences for simple lookups, ≤ 6 sentences for multi-Khoản answers.

## Curation tips

* Sample real Top-3 reranked outputs from the live pipeline, then hand-edit
  the answer. The context this produces is the same shape the model sees at inference.
* Include ~10–20 % "refusal" examples where the context is deliberately
  irrelevant — this is the only way to teach the model not to confabulate.
* Cover both **citation-style queries** (`"Khoản 5 Điều 5 Nghị định 100..."`) and
  **natural questions** (`"Lái xe ô tô vượt đèn đỏ bị phạt bao nhiêu?"`).

See `lora_train.jsonl` in this directory for a starter set.
