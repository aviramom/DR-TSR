#!/bin/bash

################################################################################################
# Add the missing two-stage (coarse-to-fine) retriever runs on TimeSeriesExam.
#
# Each two-stage retriever pulls a coarse candidate set (--stage1_candidates, default 50)
# by TIME-SERIES similarity, then re-ranks those candidates by TEXT similarity of the
# question and keeps the final --num_shots. This sweeps the two stage-1 TS encoders as an
# ablation pair:
#   twostage-delay_dino-text       DINOv3 delay-embedding image  → text (MiniLM)
#   twostage-vision_wavelet-text   DINOv3 CWT scalogram image     → text (MiniLM)
#
# Usage:
#   bash scripts/submit_tse_twostage.sh
#
# Job breakdown (30 total): 3 models × 2 retrievers × 5 shot values × 1 seed.
# No random baseline, no seed sweep — retrieval is deterministic per retriever.
#
# Runner routing:
#   Qwen3-8B-vllm              → run_single_gpu_vllm.sh   (RTX 4090, multits_large)
#   Qwen3-VL-8B, ChatTS-8B     → run_single_gpu.sh        (RTX 4090, multits)
#
# Device: stage-1 DINOv3 encoders index the pool on the GPU before the LLM loads, then
# offload to CPU (--retriever_device cuda). vision_wavelet needs PyWavelets — already
# installed in both conda envs (see scripts/CLAUDE.md). --stage1_candidates defaults to
# 50, so it is not passed explicitly.
################################################################################################

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/DR-TSR"
exp_id="twostage_v1"
task_id="TimeSeriesExam"
batch_size=1
num_samples=""          # set to e.g. 50 for smoke tests; leave empty for full eval

seed=2021               # single seed — retrieval is deterministic
shots=(1 2 3 5 8)

retrievers=(twostage-delay_dino-text twostage-vision_wavelet-text)

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
# One job per (model × retriever × shot)
# ---------------------------------------------------------------------------

echo "--- two-stage jobs ---"

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
