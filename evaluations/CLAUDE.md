# evaluations/

One evaluation file per dataset.  Each file exposes a single top-level
function that receives a loaded model and a loaded dataset, runs inference,
and returns `(metrics, artifacts)`.

---

## Convention

```python
metrics, artifacts = evaluate_<dataset>(
    model,          # BaseModelWrapper subclass
    dataset,        # corresponding dataset class from data_provider/
    batch_size=1,
    num_shots=0,
    retriever=None, # optional — enables k-shot ICL
)
```

**`metrics`** — flat/nested dict of scalars for logging:
- `accuracy` — primary metric, always present
- `balanced_accuracy`, `f1_macro`, `f1_weighted`, `precision_*`, `recall_*` — sklearn metrics (omitted if sklearn unavailable)
- `invalid_rate` — fraction of samples where no option letter was extractable
- `total_samples`, `num_shots`
- `accuracy_by_category`, `accuracy_by_subcategory`, `accuracy_by_difficulty` — nested dicts of group → float

**`artifacts`** — per-sample lists (aligned by index) for debugging:
- `item_ids`, `input_prompts`, `generated_texts`
- `predicted_answers`, `gold_answers`, `correct`
- `categories`, `subcategories`, `difficulties`
- `input_ts`

---

## Retriever contract

When `retriever` is not None, `evaluate_*` calls:

```python
demo_items = retriever.retrieve(query_item, k=num_shots)
```

`demo_items` is a list of k item dicts in the same format as the dataset
returns.  `build_icl_prompt(query_item, demo_items)` then prepends them to
the query prompt and concatenates their `input_ts` arrays, so both combined
and separate input modes are handled automatically.

---

## Files

| File | Dataset | Eval function |
|------|---------|---------------|
| `timeseriesexam_eval.py` | TimeSeriesExam (TSE) | `evaluate_tse` |

---

## Label extraction

`_extract_label(response, options)` is shared logic that recovers an option
letter from free-form model output.  It collects all positions where an
option appears under multiple patterns (bare letter, `A)`, `Answer: A`, etc.)
and returns the option whose **last** match appears latest — the final answer
rather than one cited in reasoning.  If no pattern fires it returns the
sentinel `INVALID_PREDICTION`, which counts as wrong in all metrics.
