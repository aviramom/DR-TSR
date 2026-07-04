# DR-TSR — Demonstration Retrieval for Time Series Reasoning

Research project studying utility-aware in-context learning (ICL) for time series
reasoning benchmarks. The central question: given a fixed budget of k demonstrations,
which examples maximally improve a frozen LLM's accuracy on a new query?

See `research_proposal.md` for the full research context, related work, and experiment plan.

---

## Repository Layout

```
DR-TSR/
├── run_exp.py                  ← Main experiment entry point
├── configs/
│   └── data_paths.yaml         ← task_id → dataset file path
├── data_provider/              ← Dataset wrappers (batch contract)
├── evaluations/                ← Per-dataset eval loops + ICL prompt builder
├── models/                     ← BaseModelWrapper + all model/baseline classes
├── loggers/                    ← W&B + tqdm logging (CompositeLogger)
├── utils/
│   ├── args.py                 ← Global CLI argument parser
│   └── model.py                ← method_wrapper_dict (method ID → model class)
├── third_party/
│   └── TimeSeriesExam/         ← Upstream dataset generation code (not used at runtime)
└── qa_dataset.json             ← Pre-generated TimeSeriesExam dataset (746 questions)
```

Each subdirectory has its own `CLAUDE.md` with detailed documentation.

---

## Running an Experiment

```bash
python run_exp.py \
    --method random_baseline \
    --task_id TimeSeriesExam \
    --num_shots 0 \
    --num_samples 50 \
    --display_samples 3
```

`--task_id` is the only data argument needed — the path is resolved from
`configs/data_paths.yaml`. Add a new dataset by adding one line there and a
corresponding branch in `run_exp.py:_build_dataset` and `_get_eval_fn`.

**With W&B logging:**
```bash
python run_exp.py --method Qwen/Qwen3-4B-Instruct-2507 \
    --task_id TimeSeriesExam --num_shots 1 \
    --use_wandb 1 --exp_id exp1_baseline
```

---

## Key Abstractions

### Batch contract

Every dataset item and every model input/output flows through a single dict format:

| Field | Type | Description |
|-------|------|-------------|
| `input_text` | `str` | Full prompt with `<ts><ts/>` placeholders for each series |
| `input_ts` | `list[list[float]]` | Raw float arrays, one per series — always populated |
| `output_text` | `str` | Gold option letter (`'A'`, `'B'`, …) |
| `task_id` | `str` | Dataset identifier |
| `options` | `list[str]` | Valid option letters for this sample |

### TS serialization

The dataset always produces `<ts><ts/>` placeholders. Text-only models call `fill_ts_placeholders(input_text, input_ts)` from `utils/ts_serialize.py` inside `generate()` to substitute numeric arrays before inference. Multimodal models (image, ChatTS) consume the placeholders natively.

### Model registration

All models live in `utils/model.py:method_wrapper_dict`. Pass the key as `--method`:

```python
"random_baseline"              → RandomBaseline
"Qwen/Qwen3-4B-Instruct-2507" → InstructModel
"anthropic"                    → APIModelWrapper
```

---

## Current Status

| Component | Status |
|-----------|--------|
| TimeSeriesExam dataset wrapper | Done |
| TSE evaluation loop (`evaluate_tse`) | Done |
| All model wrappers + baselines | Done |
| W&B logging | Done |
| `run_exp.py` orchestration | Done |
| Retriever (TS index, text index) | Not yet |
| Utility scorer | Not yet |
