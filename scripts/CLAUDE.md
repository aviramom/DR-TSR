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
| `submit_tse_qwen3_text.sh` | orchestrator | — | — | — | Qwen3-8B only, via vLLM (`-vllm` method, `run_single_gpu_vllm.sh`) — 36 k-shot jobs |
| `submit_tse_rrf_dino.sh` | orchestrator | — | — | — | Backfills the `rrf` and `delay_dino` retrievers (added after the initial sweep) across all 3 non-baseline models — 90 k-shot jobs |
| `submit_tse_rrf_combos.sh` | orchestrator | — | — | — | Sweeps all RRF pair combos + the 3-way shape fusion (`rrf-<a>-<b>[-<c>]` specs; ts+text skipped — already logged as `rrf`) across all 3 non-baseline models — 270 k-shot jobs |
| `submit_tse_mrrf_full.sh` | orchestrator | — | — | — | Full 6-way MR-RRF fusion only (`rrf-text-ts-vision_ts-spectral-stats-vision_wavelet`, see `MRRF_METHOD.md`) across all 3 non-baseline models × shots {1,2,3,5,8} × 3 seeds — 45 k-shot jobs. Run this before the MR-RRF ablation grid |
| `submit_tse_textbge_mrrf.sh` | orchestrator | — | — | — | Two retrievers: `text_bge` (BGE-large single text) + the full 6-way MR-RRF spec, across all 3 non-baseline models — 90 k-shot jobs. The MR-RRF half duplicates `submit_tse_mrrf_full.sh`; drop it from the array if that already ran |

---

## Conda Environments

- **`multits`** — standard env for HF-based models and baselines.
- **`multits_large`** — newer transformers (5.8) + torch 2.10/cu128 + vLLM 0.19.1; required for Qwen3.6-27B and vLLM runners. Also has `sentence_transformers` + `momentfm` installed so the text/ts retrievers run here too (vLLM ships FlashAttention 2 as its attention backend — no separate flash-attn build needed).

Both envs have `PyWavelets` installed (needed by the `vision_wavelet` retriever's CWT scalograms).

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
| `seeds` | Array of random seeds (default: 2021 0 1 — 3 seeds for variance) |
| `shots_kshot` | k values for k-shot runs (0-shot is always submitted separately) |
| `retrievers` | Retriever IDs to sweep (`random text vision_ts ts`) |
| `gpu_models` | LLMs to run on GPU |
| `num_samples` | Cap dataset size for smoke tests (empty = full eval) |
| `batch_size` | Samples per `model.generate()` call |
| `cache_dir` | HuggingFace model cache directory |

**Job structure**: 0-shot jobs never pass `--retriever` (defaults to `none`). k-shot jobs
(shots 1, 2, 3) sweep all four retrievers. Random baseline → CPU; LLMs → GPU with
`--device cuda --retriever_device cuda` — retriever encoders index the pool on the GPU
*before* the LLM loads, then offload to CPU (leave-one-out queries reuse indexed
embeddings, so no encoder is needed during evaluation).

**Default sweep**: 156 jobs — 12 zero-shot + 144 k-shot (4 models × 4 retrievers × 3 shots × 3 seeds).
