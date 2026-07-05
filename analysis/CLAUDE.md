# analysis/

Local analysis notebooks and helpers for the DR-TSR retriever comparison experiment.
All analysis reads directly from `results/` — no W&B API calls needed, since every
completed job saves a per-sample JSON to that directory.

---

## Data source

Each file in `results/` is a JSON array of 746 records (one per TimeSeriesExam question)
with the following fields:

| Field | Description |
|-------|-------------|
| `method` | Model ID (e.g. `Qwen/Qwen3-8B`) |
| `retriever` | Retriever ID (`none`, `random`, `text`, `vision_ts`, `ts`) |
| `num_shots` | Number of demonstrations (0–3) |
| `seed` | Random seed (0, 1, 2021) |
| `correct` | bool — whether the model answered correctly |
| `predicted_answer` | Extracted option letter |
| `gold_answer` | Ground-truth option letter |
| `category` | Top-level question category |
| `subcategory` | Fine-grained question type |
| `difficulty` | `easy` / `medium` / `hard` |
| `tid` | Question template ID (1–104) |
| `retrieved_demo_ids` | IDs of the k demonstrations used (empty for 0-shot) |

Filename convention: `{exp_id}__{method}__{task_id}__seed{seed}__{retriever}__{k}shot.json`

---

## Files

| File | Purpose |
|------|---------|
| `retriever_comparison.ipynb` | Main analysis notebook — accuracy vs shots, retriever heatmaps, per-category/difficulty breakdowns, seed-averaged with error bars |

---

## Notebook structure (`retriever_comparison.ipynb`)

1. **Load** — glob all `results/retriever_comparison_v1__*.json`, build a flat pandas DataFrame
2. **Accuracy vs shots** — line plot per retriever, one subplot per model; seed mean ± std
3. **Retriever × model heatmap** — at each shot count, who wins?
4. **Per-category breakdown** — which question types benefit most from retrieval?
5. **Per-difficulty breakdown** — does retrieval help more on hard questions?
6. **Per-tid breakdown** — per-template heatmap (104 templates × retriever)

---

## Coverage (as of 2026-07-04)

| Model | 0-shot | k-shot (all retrievers) |
|-------|--------|------------------------|
| `bytedance-research/ChatTS-8B` | 3 seeds | 3 seeds complete |
| `Qwen/Qwen3-VL-8B-Instruct` | 3 seeds | 3 seeds complete |
| `random_baseline` | 3 seeds | mostly complete (a few `vision_ts` files missing) |
| `Qwen/Qwen3-8B` | 3 seeds | in progress (k-shot jobs resubmitted after OOM fix) |
