# DR-TSR — Demonstration Retrieval for Time Series Reasoning

Research project studying **utility-aware demonstration retrieval for in-context learning (ICL)** on time series reasoning tasks. Given a pool of labeled (time series, question, answer) triples, the goal is to select which *k* demonstrations to place in a frozen LLM's context to maximize MCQ accuracy on a new, unseen query — without any gradient updates.

See [`research_proposal.md`](research_proposal.md) for the full motivation, related work, and experimental design.

---

## Quick Start — Experiment 1 (Retrieval Baseline Study)

```bash
# Smoke test: random baseline, no GPU needed
python run_retrieval.py \
  --method random_baseline \
  --n_splits 2 \
  --retrieval_strategies random zero_shot oracle \
  --retrieval_k 0 1 3 \
  --display_samples 2

# Full Experiment 1: all 6 conditions × k ∈ {0,1,2,3,5,8} × 5 splits
python run_retrieval.py \
  --method Qwen/Qwen3-4B-Instruct-2507 \
  --cache_dir /cs/azencot_fsas/aviramom \
  --n_splits 5 \
  --retrieval_strategies all \
  --retrieval_k 0 1 2 3 5 8 \
  --fusion_alpha 0.25 0.5 0.75 \
  --use_wandb 1 --project aviramom-/DR-TSR --exp_id retrieval_baseline
```

---

## Dataset

**TimeSeriesExam** (`qa_dataset.json`) — 763 MCQ instances across 98 templates, 5 categories.

To generate additional TS variants per template (for augmented pool):
```bash
python scripts/generate_tse_augmented.py \
  --tse_repo third_party/TimeSeriesExam \
  --output qa_dataset_augmented.json \
  --num_variants 10
```

---

## Retrieval Conditions (Experiment 1)

| Condition | `--retrieval_strategies` value | Description |
|---|---|---|
| Zero-shot (k=0) | `zero_shot` | No demonstrations |
| Random-k | `random` | k demos sampled uniformly from pool |
| TS-only top-k | `ts_only` | Top-k by TS shape similarity (L2-normalized cosine) |
| Text-only top-k | `text_only` | Top-k by sentence embedding cosine similarity |
| Fusion (α sweep) | `fusion` | Score = α·TS_sim + (1−α)·text_sim |
| Same-category oracle | `oracle` | Upper bound — pool restricted to same category |

Use `--retrieval_strategies all` to run all six.

---

## Models

| `--method` | Type | Hardware |
|---|---|---|
| `random_baseline` | Random | CPU |
| `Qwen/Qwen3-4B-Instruct-2507` | Text LLM | RTX 4090 |
| `Qwen/Qwen3.6-27B` | Large text LLM | 2× RTX 6000 (vLLM) |
| `Qwen/Qwen3-VL-8B-Instruct` | Vision LLM (TS→image) | RTX 4090 |
| `bytedance-research/ChatTS-8B` | TS-native LLM | RTX 4090 |
| `anthropic` / `openai` / `gemini` | API models | — |

---

## Directory Structure

```
DR-TSR/
├── run_retrieval.py          # Entry point — Experiment 1 orchestrator
├── qa_dataset.json           # TimeSeriesExam dataset (763 instances, 98 templates)
├── research_proposal.md      # Full research proposal
│
├── retrieval/                # Core retrieval module
│   ├── dataset.py            # TSEItem, load_tse_items(), cross_template_split()
│   ├── indices.py            # TSIndex (shape similarity), TextIndex (sentence embeddings)
│   ├── prompt.py             # build_retrieval_prompt() — ICL prompt builder
│   └── strategies.py        # retrieve() — routes to zero_shot/random/ts_only/text_only/fusion/oracle
│
├── evaluations/
│   ├── tse_retrieval_eval.py # run_evaluation_retrieval() — eval loop for one (strategy, k)
│   └── icl_ucr_eval.py       # _extract_predicted_label() — shared label extraction
│
├── models/                   # Model wrappers (all implement BaseModelWrapper)
├── utils/                    # args.py (get_retrieval_parser), model.py (registry)
├── loggers/                  # W&B + print logging
├── scripts/                  # generate_tse_augmented.py
├── third_party/TimeSeriesExam/  # TSE generation code (git submodule)
└── retrieval_results/        # Output JSONs (gitignored except smoke-test references)
```

---

## W&B

```bash
python run_retrieval.py --use_wandb 1 --project aviramom-/DR-TSR --exp_id retrieval_baseline ...
```

Results per condition and k are logged with keys: `retrieval_strategy`, `fusion_alpha`, `k`, `split_idx`.
