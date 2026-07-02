# DR-TSR — Demonstration Retrieval for Time Series Reasoning

## Project Overview

Research project studying **utility-aware demonstration retrieval for in-context learning (ICL)** on time series reasoning tasks. Given a pool of labeled (time series, question, answer) triples, the goal is to select which *k* demonstrations to place in a frozen LLM's context to maximize MCQ accuracy on a new, unseen query — without any gradient updates.

The core hypothesis: *similarity* (shape distance or embedding proximity) is a weak proxy for *utility* (whether a demonstration actually helps the model produce the correct answer). The project systematically tests this through retrieval baseline experiments on TimeSeriesExam.

See `research_proposal.md` for full motivation, related work, proposed approach, and open questions.

**Current phase: Experiment 1** — cross-template baseline study comparing six retrieval conditions (zero-shot, random, TS-only, text-only, fusion, oracle) on TimeSeriesExam.

---

## Directory Structure

```
DR-TSR/
├── run_retrieval.py          # Entry point — Experiment 1 orchestrator
├── qa_dataset.json           # TimeSeriesExam dataset (763 instances, 98 templates)
├── research_proposal.md      # Full research proposal
│
├── retrieval/                # Core retrieval module
│   ├── dataset.py            # TSEItem dataclass, load_tse_items(), cross_template_split()
│   ├── indices.py            # TSIndex (L2-normalized cosine), TextIndex (sentence-transformers)
│   ├── prompt.py             # build_retrieval_prompt() → (prompt_str, option_letters)
│   └── strategies.py        # retrieve() router + ALL_STRATEGIES list
│
├── evaluations/
│   ├── tse_retrieval_eval.py # run_evaluation_retrieval() — eval loop for one (strategy, k)
│   └── icl_ucr_eval.py       # _extract_predicted_label() — shared MCQ label extraction
│
├── models/                   # Model wrappers — all implement BaseModelWrapper
├── utils/
│   ├── args.py               # get_retrieval_parser() — CLI args for Experiment 1
│   └── model.py              # method_wrapper_dict registry
├── loggers/                  # W&B + print logging (setup_logger entry point)
├── scripts/
│   └── generate_tse_augmented.py  # Generate N TS variants per option per template
├── third_party/TimeSeriesExam/    # TSE generation code (git submodule, exam_generation branch)
└── retrieval_results/             # Output JSONs (smoke-test references committed; others gitignored)
```

---

## Dataset

### TimeSeriesExam (`qa_dataset.json`)

- **Size**: 763 MCQ instances, 98 templates, 5 categories
- **Categories**: Pattern Recognition, Noise Understanding, Anomaly Detection, Similarity Analysis, Causality Analysis
- **Format**: each instance has `tid` (template id), `question`, `options` (list of display names), `answer` (correct option text), `category`, `subcategory`, `difficulty`, `ts` (list of floats, length 1024), optionally `ts1`/`ts2` for two-series templates
- **Loaded by**: `retrieval/dataset.py:load_tse_items()` → list of `TSEItem` objects

**Cross-template split** (`retrieval/dataset.py:cross_template_split()`): stratified by category, holds out `test_fraction` of tids entirely for query set. Pool and query items never share a tid.

---

## Pipeline — Experiment 1

```
qa_dataset.json
    ↓
load_tse_items()                      ← retrieval/dataset.py
    ↓
cross_template_split(items, seed)     ← retrieval/dataset.py
    → pool_items, query_items
    ↓
TSIndex(pool_items)                   ← retrieval/indices.py
TextIndex(pool_items).build()         ← retrieval/indices.py
    ↓
For each (strategy, k, [alpha]):
    retrieve(query, pool, strategy, k, ...)  ← retrieval/strategies.py
        → k demonstrations
    build_retrieval_prompt(query, demos)     ← retrieval/prompt.py
        → (prompt_str, option_letters)
    model.generate(batch)                    ← models/
    _extract_predicted_label(output, opts)   ← evaluations/icl_ucr_eval.py
        → MCQ accuracy per condition
    ↓
Aggregate mean ± std across splits
Save to retrieval_results/<exp_id>.json
```

---

## Retrieval Conditions

| Strategy | Key | Description |
|---|---|---|
| Zero-shot | `zero_shot` | k=0, no demonstrations |
| Random | `random` | k demos sampled uniformly from pool |
| TS-only | `ts_only` | Top-k by L2-normalized cosine of TS vectors |
| Text-only | `text_only` | Top-k by sentence-embedding cosine of question text |
| Fusion | `fusion` | Score = α·TS_sim + (1−α)·text_sim, sweep over α |
| Oracle | `oracle` | Pool restricted to same category as query |

`ALL_STRATEGIES = ["zero_shot", "random", "ts_only", "text_only", "fusion", "oracle"]`

---

## Key CLI Arguments (`utils/args.py:get_retrieval_parser()`)

```bash
--method random_baseline          # Model to use (see Models section)
--qa_dataset_path qa_dataset.json # Path to TSE dataset
--retrieval_strategies all        # "all" or subset: random zero_shot ts_only text_only fusion oracle
--retrieval_k 0 1 2 3 5 8        # Demo budgets to evaluate
--fusion_alpha 0.25 0.5 0.75     # α values to sweep for fusion condition
--n_splits 5                      # Number of cross-template splits
--test_fraction 0.2               # Fraction of templates held out for query set
--text_encoder all-MiniLM-L6-v2  # Sentence-transformer model for TextIndex
--ts_max_len 128                  # TS subsampled to this length in prompt text
--random_seed 42
--use_wandb 1
--project aviramom-/DR-TSR
--exp_id retrieval_baseline
--cache_dir /cs/azencot_fsas/aviramom
--quantization none|4bit|8bit
--batch_size 1
--display_samples 3
```

---

## Models

### Registered in `utils/model.py:method_wrapper_dict`

| Method ID | Class | Backend | Notes |
|---|---|---|---|
| `random_baseline` | `RandomBaseline` | CPU | Smoke tests |
| `Qwen/Qwen3-4B-Instruct-2507` | `InstructModel` | HF | Primary small text model |
| `Qwen/Qwen3.6-27B` | `LargeInstructModel` | vLLM | Large text model |
| `Qwen/Qwen3.6-27B-image-ts` | `ImageInstructModel` | vLLM | TS → plot → VL model |
| `Qwen/Qwen3-VL-8B-Instruct` | `QwenVLImageModel` | HF Vision2Seq | TS → plot |
| `Qwen/Qwen3-VL-8B-Thinking` | `QwenVLThinkingModel` | HF + thinking | TS → plot + CoT |
| `Qwen/Qwen3-VL-8B-Thinking-vllm` | `QwenVLThinkingVLLMModel` | vLLM + thinking | Preferred thinking model |
| `bytedance-research/ChatTS-8B` | `ChatTSHFWrapper` | HF | TS patch embeddings |
| `bytedance-research/ChatTS-8B-vllm` | `ChatTSVLLMWrapper` | vLLM | ChatTS vLLM variant |
| `anthropic` / `openai` / `gemini` | `APIModelWrapper` | REST API | External API |

**Batch contract**: all models expect:
```python
{
    "input_text": [str, ...],   # Full prompt (TS already embedded as "[v1, v2, ...]")
    "input_ts":   [list, ...],  # Raw TS arrays (empty lists if TS is in text)
    "output_text": [str, ...],  # Gold label (for reference, not used by model)
    "options":    [list, ...],  # Valid option letters per sample
    "task_id":    [str, ...],   # "retrieval"
}
```

**ChatTS caveat**: `ChatTSHFWrapper` requires non-None `input_ts`. In the current retrieval pipeline, TS is embedded in `input_text` and `input_ts` is passed as `[[], ...]` (empty lists). ChatTS will fail until `input_ts` is populated with real TS arrays in `tse_retrieval_eval.py`.

See `models/CLAUDE.md` for full details.

---

## Label Extraction (`evaluations/icl_ucr_eval.py:_extract_predicted_label()`)

Maps free-text model response to one of the known option letters (A/B/C/D). Matches across 7 patterns; returns the option whose **last** match appears latest in the response (handles thinking-mode models). Returns `"INVALID_PREDICTION"` if no pattern matches.

---

## Running Experiments

```bash
# Smoke test — no GPU
python run_retrieval.py \
  --method random_baseline \
  --n_splits 2 \
  --retrieval_strategies random zero_shot oracle \
  --retrieval_k 0 1

# Full Experiment 1 — small GPU
python run_retrieval.py \
  --method Qwen/Qwen3-4B-Instruct-2507 \
  --cache_dir /cs/azencot_fsas/aviramom \
  --n_splits 5 --retrieval_strategies all \
  --retrieval_k 0 1 2 3 5 8 --fusion_alpha 0.25 0.5 0.75 \
  --use_wandb 1 --project aviramom-/DR-TSR --exp_id retrieval_baseline
```

---

## W&B Logging

- **Project**: `aviramom-/DR-TSR`
- **Enable**: `--use_wandb 1 --project aviramom-/DR-TSR --exp_id <name>`
- **Keys per run**: `retrieval_strategy`, `fusion_alpha`, `k`, `split_idx`, `accuracy`
- **Deduplication**: `get_matching_run()` skips runs already finished with the same config

See `loggers/CLAUDE.md` for auth setup.

---

## Cluster Setup

- Model weights cache: `/cs/azencot_fsas/aviramom`
- Small GPU (RTX 4090, 24 GB): use `multits` conda env
- Large GPU (2× RTX 6000, vLLM): use `multits_large` conda env (sets `NCCL_P2P_DISABLE=1`)

---

## Adding a New Model

1. Create `models/my_model.py` — implement `BaseModelWrapper`:
   - `load_model(args)` — load weights
   - `generate(batch, max_new_tokens, **kwargs) → List[str]` — inference
2. Register in `utils/model.py:method_wrapper_dict`: `"my_model_id": MyModel`
3. Test: `python run_retrieval.py --method my_model_id --retrieval_strategies random --retrieval_k 1 --n_splits 1`
