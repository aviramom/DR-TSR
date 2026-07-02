#!/bin/bash

################################################################################################
# Submit TimeSeriesExam evaluation jobs across models × seeds × shots.
#
# Usage:
#   bash scripts/submit_tse_exp.sh
#
# Runner routing:
#   random_baseline              → run_single_cpu.sh   (CPU, multits)
#   Qwen3-8B, Qwen3-VL-8B,
#   TimeOmni-1-7B, ChatTS-8B    → run_single_gpu.sh   (RTX 4090, multits)
#
# NOTE: ChatTS-8B-vllm is disabled — vLLM 0.19.1 (multits_large) removed VLLM_USE_V1
# and its V1 TransformersForCausalLM backend crashes on ChatTS weight names.
# Use bytedance-research/ChatTS-8B (HF wrapper) instead.
################################################################################################

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/DR-TSR"
exp_id="baseline_comparison_v1"
task_id="TimeSeriesExam"
batch_size=4
num_samples=""          # set to e.g. 50 for smoke tests; leave empty for full eval

seeds=(0)
shots=(0)

# ---------------------------------------------------------------------------
# Helper: build common args string
# ---------------------------------------------------------------------------
common_args() {
    local method="$1" seed="$2" shot="$3"
    local args=(
        --cache_dir   "$cache_dir"
        --method      "$method"
        --task_id     "$task_id"
        --exp_id      "$exp_id"
        --project     "$project"
        --batch_size  "$batch_size"
        --num_shots   "$shot"
        --random_seed "$seed"
        --use_wandb   1
        --display_samples 3
    )
    [ -n "$num_samples" ] && args+=(--num_samples "$num_samples")
    echo "${args[@]}"
}

total=0

# --- CPU: random baseline (0-shot only — no demonstrations used) ---
for seed in "${seeds[@]}"; do
    sbatch "$SCRIPT_DIR/run_single_cpu.sh" $(common_args "random_baseline" "$seed" 0)
    ((total++))
done

# --- RTX 4090 (multits): text + vision LLMs + ChatTS HF ---
for method in \
    "Qwen/Qwen3-8B-Instruct" \
    "Qwen/Qwen3-VL-8B-Instruct" \
    "anton-hugging/TimeOmni-1-7B" \
    "bytedance-research/ChatTS-8B"
do
    for seed in "${seeds[@]}"; do
        for shot in "${shots[@]}"; do
            sbatch "$SCRIPT_DIR/run_single_gpu.sh" $(common_args "$method" "$seed" "$shot")
            ((total++))
        done
    done
done

echo "Submitted $total jobs — task=${task_id}, exp_id=${exp_id}"
echo "  seeds: ${seeds[*]}   shots: ${shots[*]}"
