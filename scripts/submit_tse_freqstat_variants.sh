#!/bin/bash

################################################################################################
# Compare the three handcrafted-feature retrievers on TimeSeriesExam.
#
# These three each capture a different, non-shape signal from the series (see
# retrievers/spectral_retriever.py, retrievers/stats_retriever.py,
# retrievers/wavelet_retriever.py):
#   spectral        — FFT magnitude spectrum (top-32 components, 128-bin grid); pure
#                      numpy, no encoder. Captures global periodicity/rhythm.
#   stats           — 8 interpretable statistical features (mean, std, trend slope,
#                      noise, seasonality strength, kurtosis, lag-1 autocorr,
#                      permutation entropy); pure numpy/scipy, no encoder.
#   vision_wavelet  — Morlet CWT scalogram rendered as an image, encoded with the
#                      same DINOv3 backbone as vision_ts. Captures time-frequency
#                      localization (when each frequency occurs, not just whether).
#
# Usage:
#   bash scripts/submit_tse_freqstat_variants.sh
#
# Job breakdown (45 total): 3 models × 3 retrievers × 5 shot values × 1 seed.
# No random baseline, no seed sweep — this is a head-to-head of the retrievers,
# and retrieval itself is deterministic.
#
# Runner routing:
#   Qwen3-8B-vllm              → run_single_gpu_vllm.sh   (RTX 4090, multits_large)
#   Qwen3-VL-8B, ChatTS-8B     → run_single_gpu.sh        (RTX 4090, multits)
#
# Device split: spectral/stats build their pool index on CPU-cheap numpy in
# seconds regardless of device; vision_wavelet's DINOv3 encoder indexes the pool
# on the GPU before the LLM loads (--retriever_device cuda), then offloads to
# CPU — safe alongside vLLM. vision_wavelet needs PyWavelets (already installed
# in multits + multits_large).
################################################################################################

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/DR-TSR"
exp_id="freqstat_variants_v1"
task_id="TimeSeriesExam"
batch_size=1
num_samples=""          # set to e.g. 50 for smoke tests; leave empty for full eval

seed=2021               # single seed — retrieval is deterministic per retriever
shots=(1 2 3 5 8)                # matches the shot sweep of submit_tse_ts_variants.sh
                        # so results are comparable across representation choices

retrievers=(spectral stats vision_wavelet)

# HF-served GPU models (run_single_gpu.sh, multits env).
gpu_models=(
    "Qwen/Qwen3-VL-8B-Instruct"
    "bytedance-research/ChatTS-8B"
)

# vLLM-served GPU models (run_single_gpu_vllm.sh, multits_large env).
vllm_models=(
    "Qwen/Qwen3-8B-vllm"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Args shared by every job.
common_args() {
    local method="$1" retriever="$2" shot="$3"
    local args=(
        --cache_dir        "$cache_dir"
        --method           "$method"
        --task_id          "$task_id"
        --exp_id           "$exp_id"
        --project          "$project"
        --batch_size       "$batch_size"
        --num_shots        "$shot"
        --random_seed      "$seed"
        --retriever        "$retriever"
        --device           cuda
        --retriever_device cuda
        --use_wandb        1
        --display_samples  3
    )
    [ -n "$num_samples" ] && args+=(--num_samples "$num_samples")
    echo "${args[@]}"
}

total=0

# ---------------------------------------------------------------------------
# One job per (model × retriever)
# ---------------------------------------------------------------------------

echo "--- freqstat-variant jobs ---"

# GPU: LLMs (HF runner)
for method in "${gpu_models[@]}"; do
    for retriever in "${retrievers[@]}"; do
        for shot in "${shots[@]}"; do
            sbatch "$SCRIPT_DIR/run_single_gpu.sh" \
                $(common_args "$method" "$retriever" "$shot")
            ((total++))
        done
    done
done

# GPU: LLMs (vLLM runner)
for method in "${vllm_models[@]}"; do
    for retriever in "${retrievers[@]}"; do
        for shot in "${shots[@]}"; do
            sbatch "$SCRIPT_DIR/run_single_gpu_vllm.sh" \
                $(common_args "$method" "$retriever" "$shot")
            ((total++))
        done
    done
done

# ---------------------------------------------------------------------------
echo ""
echo "Submitted $total jobs"
echo "  task=${task_id}  exp_id=${exp_id}"
echo "  seed: ${seed}  shots: ${shots[*]}  retrievers: ${retrievers[*]}"
echo "  gpu models (HF):   ${gpu_models[*]}"
echo "  gpu models (vLLM): ${vllm_models[*]}"
