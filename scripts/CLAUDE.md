# scripts/

SLURM submission scripts for running DR-TSR experiments on the cluster. All runner scripts
are thin wrappers around `run_exp.py` — they set up the SLURM environment and pass `"$@"`
through unchanged.

---

## Scripts

| Script | Runner type | GPU | Conda env | Walltime | Use case |
|--------|------------|-----|-----------|----------|----------|
| `run_single_cpu.sh` | CPU | none | `multits` | 30 min | Baselines, API models |
| `run_single_gpu.sh` | GPU | 1× RTX 4090 | `multits` | 1 h | HF models up to ~14B |
| `run_single_gpu_vllm.sh` | GPU | 1× RTX 4090 | `multits_large` | 1 h | vLLM models on a single 4090 |
| `run_single_gpu_large.sh` | GPU | 2× RTX 6000 | `multits_large` | 2 h | 27B+ models (tensor parallel) |
| `submit_tse_exp.sh` | orchestrator | — | — | — | Sweeps all models × seeds × shots |

---

## Conda Environments

- **`multits`** — standard env for HF-based models and baselines.
- **`multits_large`** — newer transformers + CUDA 12.4 + vLLM 0.19.1; required for Qwen3.6-27B and vLLM runners.

> **Note**: `ChatTSVLLMWrapper` is broken on vLLM 0.19.1 (multits_large removed `VLLM_USE_V1`).
> Use `bytedance-research/ChatTS-8B` (`ChatTSHFWrapper`) with `run_single_gpu.sh` instead.

---

## Usage

Run a single job manually:
```bash
sbatch scripts/run_single_gpu.sh \
    --method Qwen/Qwen3-8B \
    --task_id TimeSeriesExam \
    --num_shots 0 \
    --exp_id baseline_comparison_v1 \
    --use_wandb 1
```

Submit the full experiment sweep:
```bash
bash scripts/submit_tse_exp.sh
```

Logs are written to `logs_terminal/<job-name>_<JOBID>.out`. Create this directory before submitting:
```bash
mkdir -p logs_terminal
```

---

## `submit_tse_exp.sh` Configuration

Edit these variables at the top of the file to control the sweep:

| Variable | Description |
|----------|-------------|
| `exp_id` | W&B experiment group label |
| `seeds` | Array of random seeds |
| `shots` | Array of k-shot values |
| `num_samples` | Cap dataset size for smoke tests (empty = full eval) |
| `batch_size` | Samples per `model.generate()` call |
| `cache_dir` | HuggingFace model cache directory |

Model routing is hardcoded in the script: `random_baseline` → CPU; everything else → `run_single_gpu.sh`.
