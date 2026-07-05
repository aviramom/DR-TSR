#!/bin/bash

################################################################################################
# Submit TimeSeriesExam evaluation jobs across models × retrievers × shots × seeds.
#
# Usage:
#   bash scripts/submit_tse_exp.sh
#
# Job breakdown (156 total at default settings):
#   4 methods = random_baseline + 2 HF GPU models + 1 vLLM GPU model
#   0-shot  : 4 methods × 3 seeds              =  12 jobs  (no retriever)
#   k-shot  : 4 methods × 4 retrievers × 3 shots × 3 seeds = 144 jobs
#   Total   : 156
#
# Runner routing:
#   random_baseline              → run_single_cpu.sh        (CPU, multits)
#   Qwen3-8B-vllm               → run_single_gpu_vllm.sh   (RTX 4090, multits_large)
#   Qwen3-VL-8B, ChatTS-8B       → run_single_gpu.sh        (RTX 4090, multits)
#
# Device split:
#   LLM                         → cuda  (--device cuda)
#   Retriever encoders           → cpu   (--retriever_device cpu)
#   This keeps VRAM free for the LLM; encoder indexing runs once per job.
#
# NOTE: ChatTS-8B-vllm is disabled — vLLM 0.19.1 (multits_large) removed VLLM_USE_V1
# and its V1 TransformersForCausalLM backend crashes on ChatTS weight names.
# Use bytedance-research/ChatTS-8B (HF wrapper) instead.
################################################################################################

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/DR-TSR"
exp_id="retriever_comparison_v1"
task_id="TimeSeriesExam"
batch_size=1
num_samples=""          # set to e.g. 50 for smoke tests; leave empty for full eval

# 3 seeds gives enough variance for error bars without tripling compute.
seeds=(2021 0 1)

shots_kshot=(5 8)     # 0-shot is handled separately (no retriever)
retrievers=(random text vision_ts ts)

# HF-served GPU models (run_single_gpu.sh, multits env).
gpu_models=(
    "Qwen/Qwen3-VL-8B-Instruct"
    "bytedance-research/ChatTS-8B"
)

# vLLM-served GPU models (run_single_gpu_vllm.sh, multits_large env).
# Qwen3-8B is text-only; vLLM's paged KV cache fits the long k-shot prompts
# that OOM'd the HF pipeline. Uses the "-vllm" method suffix → LargeInstructModel.
vllm_models=(
    "Qwen/Qwen3-8B-vllm"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Args shared by every job.
common_args() {
    local method="$1" seed="$2" shot="$3"
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

# ---------------------------------------------------------------------------
# 0-shot (no retriever — retriever arg omitted, defaults to "none")
# ---------------------------------------------------------------------------

# echo "--- 0-shot jobs ---"

# # CPU: random baseline
# for seed in "${seeds[@]}"; do
#     sbatch "$SCRIPT_DIR/run_single_cpu.sh" \
#         $(common_args "random_baseline" "$seed" 0)
#     ((total++))
# done

# # GPU: LLMs (HF runner)
# for method in "${gpu_models[@]}"; do
#     for seed in "${seeds[@]}"; do
#         sbatch "$SCRIPT_DIR/run_single_gpu.sh" \
#             $(common_args "$method" "$seed" 0) \
#             --device cuda \
#             --retriever_device cpu
#         ((total++))
#     done
# done

# # GPU: LLMs (vLLM runner)
# for method in "${vllm_models[@]}"; do
#     for seed in "${seeds[@]}"; do
#         sbatch "$SCRIPT_DIR/run_single_gpu_vllm.sh" \
#             $(common_args "$method" "$seed" 0) \
#             --device cuda \
#             --retriever_device cpu
#         ((total++))
#     done
# done

# ---------------------------------------------------------------------------
# k-shot (shots 1, 2, 3 × all retrievers)
# ---------------------------------------------------------------------------

echo "--- k-shot jobs ---"

# CPU: random baseline
for seed in "${seeds[@]}"; do
    for shot in "${shots_kshot[@]}"; do
        for retriever in "${retrievers[@]}"; do
            sbatch "$SCRIPT_DIR/run_single_cpu.sh" \
                $(common_args "random_baseline" "$seed" "$shot") \
                --retriever "$retriever"
            ((total++))
        done
    done
done

# GPU: LLMs  (LLM on cuda, retriever encoders on cpu)
for method in "${gpu_models[@]}"; do
    for seed in "${seeds[@]}"; do
        for shot in "${shots_kshot[@]}"; do
            for retriever in "${retrievers[@]}"; do
                sbatch "$SCRIPT_DIR/run_single_gpu.sh" \
                    $(common_args "$method" "$seed" "$shot") \
                    --retriever        "$retriever" \
                    --device           cuda \
                    --retriever_device cpu
                ((total++))
            done
        done
    done
done

# GPU: vLLM LLMs  (LLM on cuda, retriever encoders on cpu)
for method in "${vllm_models[@]}"; do
    for seed in "${seeds[@]}"; do
        for shot in "${shots_kshot[@]}"; do
            for retriever in "${retrievers[@]}"; do
                sbatch "$SCRIPT_DIR/run_single_gpu_vllm.sh" \
                    $(common_args "$method" "$seed" "$shot") \
                    --retriever        "$retriever" \
                    --device           cuda \
                    --retriever_device cpu
                ((total++))
            done
        done
    done
done

# ---------------------------------------------------------------------------
echo ""
echo "Submitted $total jobs"
echo "  task=${task_id}  exp_id=${exp_id}"
echo "  seeds: ${seeds[*]}"
echo "  k-shot values: ${shots_kshot[*]}  retrievers: ${retrievers[*]}"
echo "  gpu models (HF):   ${gpu_models[*]}"
echo "  gpu models (vLLM): ${vllm_models[*]}"
