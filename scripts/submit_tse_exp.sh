#!/bin/bash

################################################################################################
# Submit TimeSeriesExam evaluation jobs across models × seeds × shots.
#
# Usage:
#   bash scripts/submit_tse_exp.sh
#
# 27B models → run_single_gpu_large.sh (2× RTX 6000)
# all others → run_single_gpu.sh       (1× RTX 4090)
# baselines   → run_single_cpu.sh      (CPU)
################################################################################################

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/DR-TSR"
exp_id="baseline_comparison_v1"
task_id="TimeSeriesExam"
batch_size=4
num_samples=""          # set to e.g. 50 for smoke tests; leave empty for full eval

seeds=(0 1 2)
shots=(0 1 2)

# --- Model lists by resource tier ---
baseline_methods=(
    "random_baseline"
)

gpu_methods=(
    "Qwen/Qwen3-4B-Instruct-2507"
    "Qwen/Qwen3-8B-Instruct-2507"
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "bytedance-research/ChatTS-8B"
    "Qwen/Qwen3-VL-8B-Instruct"
    "anton-hugging/TimeOmni-1-7B"
)

gpu_large_methods=(
    "Qwen/Qwen3.6-27B"
    "Qwen/Qwen3.6-27B-FP8"
    "Qwen/Qwen3.6-27B-image-ts"
)

# ---------------------------------------------------------------------------
# Helper: build common args string
# ---------------------------------------------------------------------------
common_args() {
    local method="$1" seed="$2" shot="$3"
    local args=(
        --cache_dir "$cache_dir"
        --method    "$method"
        --task_id   "$task_id"
        --exp_id    "$exp_id"
        --project   "$project"
        --batch_size "$batch_size"
        --num_shots  "$shot"
        --random_seed "$seed"
        --use_wandb  1
        --display_samples 3
    )
    [ -n "$num_samples" ] && args+=(--num_samples "$num_samples")
    echo "${args[@]}"
}

total=0

# --- CPU baselines ---
for method in "${baseline_methods[@]}"; do
    for seed in "${seeds[@]}"; do
        for shot in "${shots[@]}"; do
            # Random baseline is deterministic in shot count — skip shots > 0 (no demos used)
            [ "$method" = "random_baseline" ] && [ "$shot" -gt 0 ] && continue
            sbatch "$SCRIPT_DIR/run_single_cpu.sh" $(common_args "$method" "$seed" "$shot")
            ((total++))
        done
    done
done

# --- GPU (4090) models ---
for method in "${gpu_methods[@]}"; do
    for seed in "${seeds[@]}"; do
        for shot in "${shots[@]}"; do
            sbatch "$SCRIPT_DIR/run_single_gpu.sh" $(common_args "$method" "$seed" "$shot")
            ((total++))
        done
    done
done

# --- GPU large (27B) models ---
for method in "${gpu_large_methods[@]}"; do
    for seed in "${seeds[@]}"; do
        for shot in "${shots[@]}"; do
            sbatch "$SCRIPT_DIR/run_single_gpu_large.sh" $(common_args "$method" "$seed" "$shot")
            ((total++))
        done
    done
done

echo "Submitted $total jobs — task=${task_id}, exp_id=${exp_id}"
echo "  seeds: ${seeds[*]}"
echo "  shots: ${shots[*]}"
echo "  baseline methods: ${baseline_methods[*]}"
echo "  gpu methods:      ${gpu_methods[*]}"
echo "  gpu-large methods: ${gpu_large_methods[*]}"
