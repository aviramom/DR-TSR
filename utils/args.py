import argparse
import numpy as np
from dotenv import load_dotenv
load_dotenv()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ICL time series classification evaluation.")

    # Experiment
    parser.add_argument("--exp_id", type=str, default="1")
    parser.add_argument("--random_seed", type=int, default=2021)

    # Logging
    parser.add_argument("--project", type=str, default="aviramom-/DR-TSR")
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--override_run", type=int, default=1)
    parser.add_argument("--keys_to_match", type=list,
                        default=["exp_id", "random_seed", "task_id", "method", "prompt_format"])

    # Model
    parser.add_argument("--method", type=str, default="random_baseline")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--quantization", type=str, choices=["none", "4bit", "8bit"],
                        default="none")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--low_cpu_mem_usage", action="store_true", default=True)

    # Data
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Max test samples per run (None = all)")

    # Task
    parser.add_argument("--task_id", type=str, default="TimeSeriesExam",
                        help="Task to evaluate. Data path is resolved from configs/data_paths.yaml.")

    # ICL / few-shot
   
    parser.add_argument("--num_shots", type=int, default=1)


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
