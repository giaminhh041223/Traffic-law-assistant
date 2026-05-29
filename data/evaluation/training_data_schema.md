# Cross-Encoder Training Data Schema

This file documents the JSONL format consumed by `src/phase3_reranking/train_cross_encoder.py`.

The training script reads a file like `cross_encoder_train.jsonl`, mines hard negatives
on the fly via BM25 (see `hard_negative_mining.py`), and fine-tunes the cross-encoder
with MultipleNegativesRankingLoss.

---

## File format

One JSON object per line. UTF-8. No trailing comma. No comments.

```json
{
  "query": "string — user-facing Vietnamese question",
  "positive_chunk_id": "string — chunk_id of the gold-answer chunk",
  "extra_positive_ids": ["optional list of chunk_ids that are ALSO valid answers"],
  "manual_negative_ids": ["optional list of chunk_ids to use as hard negatives, no mining"]
}
```

### Field semantics

| Field                  | Required | Meaning                                                                                                |
|------------------------|----------|--------------------------------------------------------------------------------------------------------|
| `query`                | yes      | The user query in natural Vietnamese. Do NOT pre-tokenise / pre-segment — the trainer handles it.       |
| `positive_chunk_id`    | yes      | ID of the chunk that correctly answers `query`. Must exist in `data/processed/chunks.jsonl`.            |
| `extra_positive_ids`   | no       | Other chunks that *also* answer the query. These are excluded from hard-negative candidates (never sampled as negatives) but are NOT trained as additional positives. Use when one query has multiple valid gold chunks. |
| `manual_negative_ids`  | no       | Curator-chosen hard negatives. Used **as-is** (no mining). Useful for tricky cases where BM25 misses a known-difficult distractor (e.g. same violation, wrong vehicle type). |

### How negatives are produced at training time

For each line:

1. Start with whatever is in `manual_negative_ids` (up to `num_per_positive`).
2. If we still need more, BM25-search the query, skip the positive(s) and any
   already-collected negatives, skip near-duplicates (token-Jaccard ≥ `dedup_jaccard_threshold`
   against the positive), and take the top-K from there.
3. If we STILL don't have enough and `fill_with_random: true`, fill with random chunks
   from the corpus. The mining log will record `random` provenance for these — useful
   diagnostic that your dataset is too small or BM25 is too weak.

All three configurable in `configs/retrieval.yaml → cross_encoder_training.hard_negatives`.

---

## How to curate a good training set

For a university-defence-grade demo of a Vietnamese traffic-law RAG, aim for:

* **30–200 high-quality (query, positive) pairs.** More is better, but each one
  should be carefully checked — a single mislabelled positive can poison MNR
  because the model is asked to push the "right" answer above all alternatives.
* **Diverse vehicle types.** Cover xe ô tô / xe mô tô / xe máy / xe đạp / xe gắn máy.
  These are the most common BM25 hard-negative confusions.
* **Diverse violation types.** Vượt đèn đỏ, nồng độ cồn, không đội mũ bảo hiểm,
  vượt tốc độ, đỗ xe sai quy định, etc.
* **Citation-style queries AND natural-language queries.** A user might ask
  "Khoản 5 Điều 5 Nghị định 100 phạt bao nhiêu?" OR "Vượt đèn đỏ xe ô tô phạt
  bao nhiêu tiền?" — train on both.
* **At least 10 cases with `manual_negative_ids`** for queries where you know
  BM25's top-K will miss a critical distractor (e.g. when the distractor uses
  different wording but is the wrong vehicle).

### Splitting train/dev

Hold out ~10–20% as `cross_encoder_dev.jsonl`. The trainer mines dev negatives
the same way and reports listwise MNR loss every `eval_steps`. Lower dev loss =
the model is better at ranking the gold answer ABOVE its hard distractors.

---

## Validation checklist (run before training)

Before kicking off a real fine-tune, verify:

```python
from src.phase3_reranking.hard_negative_mining import load_training_pairs
from src.utils.io import read_jsonl

pairs   = load_training_pairs("data/evaluation/cross_encoder_train.jsonl")
chunks  = {c["chunk_id"]: c for c in read_jsonl("data/processed/chunks.jsonl")}

bad = [p for p in pairs if p.positive_chunk_id not in chunks]
print(f"{len(bad)} pairs reference missing chunk_ids")  # should be 0
```

If `bad` is non-empty, your `chunks.jsonl` is out of date (re-run Phase 1) or
the labelled `positive_chunk_id`s use a stale ID scheme.

---

## Example record (annotated)

```jsonc
{
  // The natural-language query a user might type.
  "query": "Vượt đèn đỏ đối với xe ô tô bị phạt bao nhiêu tiền?",

  // The correct answer: Khoản 5 Điều 5 NĐ100 — phạt 4–6 triệu cho xe ô tô.
  "positive_chunk_id": "ND100_dieu5_khoan5_a",

  // Another chunk that also discusses xe ô tô vượt đèn đỏ (e.g. tước GPLX).
  "extra_positive_ids": ["ND100_dieu5_khoan11_b"],

  // We KNOW the model will try to confuse this with the xe-máy version
  // (same violation, wrong vehicle). Force it in as a negative.
  "manual_negative_ids": ["ND100_dieu6_khoan4_e"]
}
```

See `cross_encoder_train.jsonl` in this directory for a complete starter set.
