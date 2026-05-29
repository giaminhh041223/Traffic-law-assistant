# Golden QA Set — Schema

Consumed by both:
* `src/phase5_evaluation/retrieval_metrics.py` — needs `question` + `relevant_chunk_ids`.
* `src/phase5_evaluation/ragas_eval.py`        — needs `question` + `ground_truth`
  (and optionally `answer` + `contexts` when running in offline / no-pipeline mode).

## File format

JSONL, UTF-8. One JSON object per line.

```json
{
  "question": "string — user's natural-language question",
  "ground_truth": "string — the canonical correct answer (with citations)",
  "relevant_chunk_ids": ["chunk_id_1", "chunk_id_2"],
  "filters": { "vehicle_type": "o_to" },         // optional, narrows retrieval
  "answer":   "string — (offline RAGAS only)",
  "contexts": ["string", "string"]                // (offline RAGAS only)
}
```

### Field semantics

| Field                | Required for IR | Required for RAGAS live | Required for RAGAS offline |
|----------------------|:--------------:|:----------------------:|:--------------------------:|
| `question`           | ✓              | ✓                      | ✓                          |
| `relevant_chunk_ids` | ✓              | —                      | —                          |
| `ground_truth`       | —              | ✓                      | ✓                          |
| `filters`            | optional       | optional               | optional                   |
| `answer`             | —              | —                      | ✓                          |
| `contexts`           | —              | —                      | ✓                          |

### Curation notes

1. **30–100 questions is plenty** for the panel to see consistent trends.
2. **Cover the failure modes you want to advertise**: same-violation-different-vehicle
   pairs, multi-step queries that span two Khoản, out-of-scope refusals, and
   citation-style queries.
3. **`relevant_chunk_ids` may contain more than one ID** if multiple chunks
   are valid answers (e.g. the base fine in one Khoản + the additional sanction
   in another). The retrieval metrics use a set-based comparison.
4. **`ground_truth`** must follow the same citation discipline as the model's
   target output — it's the reference RAGAS uses for context_recall.
