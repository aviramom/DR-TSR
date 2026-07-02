# DR-TSR — Demonstration Retrieval for Time Series Reasoning

Research codebase for studying utility-aware in-context learning (ICL) for time series
reasoning benchmarks. The central question: given a fixed budget of *k* demonstrations,
which examples maximally improve a frozen LLM's accuracy on a new query?

See [`research_proposal.md`](research_proposal.md) for the full problem statement, related
work survey, and experiment plan.

---

## Setup

```bash
conda activate multits
pip install -r requirements.txt
```

For vLLM-backed large models (e.g. `Qwen/Qwen3.6-27B`):
```bash
pip install vllm>=0.8
```

For cloud API models:
```bash
pip install openai anthropic google-generativeai
```

---

## Quick Start

**Zero-shot random baseline (no GPU needed):**
```bash
python run_exp.py \
    --method random_baseline \
    --task_id TimeSeriesExam \
    --num_shots 0 \
    --num_samples 50
```

**Small text LLM, 1-shot, with W&B logging:**
```bash
python run_exp.py \
    --method Qwen/Qwen3-4B-Instruct-2507 \
    --task_id TimeSeriesExam \
    --num_shots 1 \
    --use_wandb 1 \
    --exp_id exp1_baseline
```

**Cloud API model:**
```bash
python run_exp.py \
    --method anthropic \
    --task_id TimeSeriesExam \
    --num_shots 0
```

Data paths are resolved automatically from [`configs/data_paths.yaml`](configs/data_paths.yaml) —
no `--data_path` argument needed.

---

## Available Models

| `--method` | Backend | Notes |
|------------|---------|-------|
| `random_baseline` | none | Uniform random label |
| `knn_baseline` | DTW | 1-NN by series shape |
| `dino_knn_clsa_baseline` | DINOv2-Large | 1-NN by plotted series image |
| `Qwen/Qwen3-4B-Instruct-2507` | HF | Also: `Qwen3-8B`, Llama, Mistral variants |
| `Qwen/Qwen3.6-27B` | vLLM | Also: `-FP8` |
| `Qwen/Qwen3-VL-8B-Instruct` | HF Vision | TS rendered as image |
| `bytedance-research/ChatTS-8B` | HF | Also: `-14B`, `-vllm` variants |
| `anton-hugging/TimeOmni-1-7B` | HF | Qwen2.5-7B base |
| `openai` / `anthropic` / `gemini` | REST API | Set API key in `.env` |

Full list and details in [`models/CLAUDE.md`](models/CLAUDE.md).

---

## Project Structure

```
DR-TSR/
├── run_exp.py              ← Experiment entry point
├── configs/
│   └── data_paths.yaml     ← task_id → dataset file path
├── data_provider/          ← Dataset wrappers
├── evaluations/            ← Eval loops + ICL prompt builder
├── models/                 ← All model wrappers and baselines
├── loggers/                ← W&B + tqdm logging
├── utils/                  ← CLI args + model registry
├── third_party/
│   └── TimeSeriesExam/     ← Upstream dataset generation code
└── qa_dataset.json         ← TimeSeriesExam dataset (746 questions)
```

Each directory contains a `CLAUDE.md` with detailed documentation.

---

## Adding a New Dataset

1. Add the data path to `configs/data_paths.yaml`.
2. Create `data_provider/<name>_data.py` following the batch contract in [`data_provider/CLAUDE.md`](data_provider/CLAUDE.md).
3. Add `evaluate_<name>` in `evaluations/<name>_eval.py`.
4. Wire both into `run_exp.py:_build_dataset` and `_get_eval_fn`.

## Adding a New Model

1. Create `models/<name>.py` subclassing `BaseModelWrapper`.
2. Register it in `utils/model.py:method_wrapper_dict`.

See [`models/CLAUDE.md`](models/CLAUDE.md) for the full wrapper contract.
