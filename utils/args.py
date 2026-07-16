import argparse
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Kept as a static list (rather than imported from utils.retriever) so that
# CLI validation stays import-light — utils.retriever pulls in torch.
_FUSABLE_RETRIEVERS = ["text", "text_bge", "ts", "ts_compress", "ts_multivec",
                       "ts_windowagg", "vision_ts", "delay_dino",
                       "spectral", "stats", "vision_wavelet", "dtw",
                       "siglip_plot", "siglip_delay"]
_PLAIN_RETRIEVERS = ["none", "random", "rrf", *_FUSABLE_RETRIEVERS]


def _retriever_name(value: str) -> str:
    """Validate --retriever: a plain name, an rrf-<a>-<b>[-<c>...] fusion spec,
    or a twostage-<a>-<b> coarse-to-fine spec."""
    if value in _PLAIN_RETRIEVERS:
        return value
    if value.startswith("rrf-"):
        components = value[len("rrf-"):].split("-")
        bad = [c for c in components if c not in _FUSABLE_RETRIEVERS]
        if len(components) >= 2 and not bad and len(set(components)) == len(components):
            return value
        raise argparse.ArgumentTypeError(
            f"invalid RRF spec '{value}': need >= 2 distinct components "
            f"from {_FUSABLE_RETRIEVERS}"
        )
    if value.startswith("twostage-"):
        components = value[len("twostage-"):].split("-")
        bad = [c for c in components if c not in _FUSABLE_RETRIEVERS]
        if len(components) == 2 and not bad:
            return value
        raise argparse.ArgumentTypeError(
            f"invalid two-stage spec '{value}': need exactly two components "
            f"from {_FUSABLE_RETRIEVERS}, e.g. twostage-delay_dino-text"
        )
    raise argparse.ArgumentTypeError(
        f"unknown retriever '{value}': choose from {_PLAIN_RETRIEVERS}, "
        f"an rrf-<a>-<b> fusion spec, or a twostage-<a>-<b> spec"
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ICL time series classification evaluation.")

    # Experiment
    parser.add_argument("--exp_id", type=str, default="1")
    parser.add_argument("--random_seed", type=int, default=2021)

    # Logging
    parser.add_argument("--project", type=str, default="aviramom-/DR-TSR")
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--override_run", type=int, default=0)
    parser.add_argument("--keys_to_match", type=list,
                        default=["exp_id", "random_seed", "task_id", "method", "retriever",
                                 "num_shots", "rrf_k", "stage1_candidates"])

    # Model
    parser.add_argument("--method", type=str, default="bytedance-research/ChatTS-8B")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--quantization", type=str, choices=["none", "4bit", "8bit"],
                        default="none")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--retriever_device", type=str, default="cpu",
                        help="Device for retriever encoder models (MOMENT, DINOv3). "
                             "cuda is safe alongside the LLM: indexing runs before "
                             "the LLM loads and encoders offload to CPU afterwards. "
                             "Default stays cpu for CPU-only nodes.")
    parser.add_argument("--low_cpu_mem_usage", action="store_true", default=True)

    # Data
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Max test samples per run (None = all)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to a single TSE category, e.g. 'Similarity Analysis'")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to write per-run prediction JSON files")

    # Task
    parser.add_argument("--task_id", type=str, default="TimeSeriesExam",
                        help="Task to evaluate. Data path is resolved from configs/data_paths.yaml.")

    # ICL / few-shot
    parser.add_argument("--num_shots", type=int, default=1)
    parser.add_argument(
        "--retriever",
        type=_retriever_name,
        default="none",
        help="Demonstration retriever for k-shot ICL. 'none' = zero-shot. "
             "Plain: none / random / text / text_bge / ts / ts_compress / "
             "ts_multivec / ts_windowagg / vision_ts / "
             "delay_dino / spectral / stats / vision_wavelet / dtw. "
             "Fused: rrf-<a>-<b>[-<c>...] combines any 2+ plain (non-random) "
             "retrievers via Reciprocal Rank Fusion, e.g. rrf-ts-delay_dino or "
             "the full MR-RRF 6-way "
             "rrf-text-ts-vision_ts-spectral-stats-vision_wavelet. "
             "'rrf' = legacy alias for rrf-ts-text. "
             "Two-stage: twostage-<a>-<b> pulls candidates by <a> then re-ranks "
             "by <b>, e.g. twostage-delay_dino-text.",
    )
    parser.add_argument(
        "--rrf_k", type=int, default=60,
        help="RRF smoothing constant (the 'k' in score = sum 1/(k + rank)) used "
             "when --retriever is an rrf-... fusion spec. Ignored otherwise. "
             "Default 60 is the standard value from Cormack et al. 2009; lower "
             "values weight top ranks more heavily, higher values flatten the fusion.",
    )
    parser.add_argument(
        "--stage1_candidates", type=int, default=50,
        help="Coarse candidate-set size pulled by the stage-1 retriever, used "
             "when --retriever is a twostage-... spec. Ignored otherwise. "
             "Stage 2 re-ranks these candidates to the final --num_shots.",
    )


    # Inference
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--display_samples", type=int, default=3)

    return parser

def create_parser(notebook: bool = False):
    parser = get_parser()
    parsed = parser.parse_args("") if notebook else parser.parse_args()

    if hasattr(parsed, "quantization") and parsed.quantization == "none":
        parsed.quantization = None
    if hasattr(parsed, "cache_dir") and parsed.cache_dir == "":
        parsed.cache_dir = None

    return parsed
