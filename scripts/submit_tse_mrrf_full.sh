#!/bin/bash

################################################################################################
# Submit TimeSeriesExam jobs for the FULL MR-RRF retriever only (see MRRF_METHOD.md).
#
# This is the headline condition — the 6-way Reciprocal Rank Fusion over all
# representation signals:
#   text            question sentence embedding (MiniLM)
#   ts              MOMENT embedding of the raw sequence
#   vision_ts       DINOv3 embedding of the rendered line plot
#   spectral        FFT magnitude spectrum (pure numpy)
#   stats           interpretable statistical feature vector (numpy/scipy)
#   vision_wavelet  DINOv3 embedding of the Morlet CWT scalogram (needs pywt)
#
# Run this BEFORE the full ablation sweep (singletons / pairs / triples) to check that
# the 6-way fusion is worth the ablation grid. Same 3 models, shots, seeds and exp_id
# as submit_tse_exp.sh / submit_tse_rrf_combos.sh, so runs land in the same W&B group.
#
# Usage:
#   bash scripts/submit_tse_mrrf_full.sh
#
# Job breakdown (45 total at default settings):
#   3 methods × 1 retriever × 5 shots × 3 seeds = 45 jobs  (no 0-shot — already logged)
#
# Runner routing (same as submit_tse_rrf_combos.sh):
#   Qwen3-8B-vllm                → run_single_gpu_vllm.sh   (RTX 4090, multits_large, batch_size=4)
#   Qwen3-VL-8B, ChatTS-8B       → run_single_gpu.sh        (RTX 4090, multits, batch_size=1)
#
# Device split:
#   LLM                          → cuda  (--device cuda)
#   Retriever encoders           → cuda  (--retriever_device cuda)
#   spectral/stats are pure CPU; text/ts/vision_ts/vision_wavelet index the pool on the
#   GPU sequentially before the LLM loads, then offload to CPU.
#
# NOTE: vision_wavelet needs PyWavelets — `pip install PyWavelets` in BOTH conda envs
# (multits and multits_large) before submitting.
################################################################################################

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/DR-TSR"
exp_id="retriever_comparison_v1"
task_id="TimeSeriesExam"
num_samples=""          # set to e.g. 50 for smoke tests; leave empty for full eval

seeds=(2021 0 1)
shots_kshot=(1 2 3 5 8)
retriever="rrf-text-ts-vision_ts-spectral-stats-vision_wavelet"

# HF-served GPU models (run_single_gpu.sh, multits env, batch_size=1).
gpu_models=(
    "Qwen/Qwen3-VL-8B-Instruct"
    "bytedance-research/ChatTS-8B"
)

# vLLM-served GPU model (run_single_gpu_vllm.sh, multits_large env, batch_size=4).
vllm_model="Qwen/Qwen3-8B-vllm"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Args shared by every job.
common_args() {
    local method="$1" seed="$2" shot="$3" batch_size="$4"
    local args=(
        --cache_dir       "$cache_dir"
        --method          "$method"
        --task_id         "$task_id"
        --exp_id          "$exp_id"
        --project         "$project"
        --batch_size      "$batch_size"
        --num_shots       "$shot"
        --random_seed     "$seed"
        --use_wandb       1
        --display_samples 3
    )
    [ -n "$num_samples" ] && args+=(--num_samples "$num_samples")
    echo "${args[@]}"
}

total=0

echo "--- k-shot jobs (full 6-way MR-RRF) ---"

# GPU: HF-served LLMs  (encoders index on cuda before the LLM loads, then offload)
for method in "${gpu_models[@]}"; do
    for seed in "${seeds[@]}"; do
        for shot in "${shots_kshot[@]}"; do
            sbatch "$SCRIPT_DIR/run_single_gpu.sh" \
                $(common_args "$method" "$seed" "$shot" 1) \
                --retriever        "$retriever" \
                --device           cuda \
                --retriever_device cuda
            ((total++))
        done
    done
done

# GPU: vLLM-served LLM  (encoders index on cuda before vLLM preallocates VRAM)
for seed in "${seeds[@]}"; do
    for shot in "${shots_kshot[@]}"; do
        sbatch "$SCRIPT_DIR/run_single_gpu_vllm.sh" \
            $(common_args "$vllm_model" "$seed" "$shot" 4) \
            --retriever        "$retriever" \
            --device           cuda \
            --retriever_device cuda
        ((total++))
    done
done

# ---------------------------------------------------------------------------
echo ""
echo "Submitted $total jobs"
echo "  task=${task_id}  exp_id=${exp_id}"
echo "  seeds: ${seeds[*]}"
echo "  k-shot values: ${shots_kshot[*]}"
echo "  retriever: ${retriever}"
echo "  gpu models (HF):   ${gpu_models[*]}"
echo "  gpu model (vLLM):  ${vllm_model}"
