#!/usr/bin/env python3
"""
Generate augmented TimeSeriesExam data for ICL experiments.

For each question template (tid), generates `num_variants` TS samples per option
(class). The output JSON is the input to TimeSeriesExamDataset.

Usage:
    python scripts/generate_tse_augmented.py \
        --tse_repo third_party/TimeSeriesExam \
        --output qa_dataset_augmented.json \
        --num_variants 10 \
        --ts_length 256 \
        --seed 42
"""

import sys
import os
import json
import re
import argparse
import traceback


def clean_option_name(name: str) -> str:
    """Convert template option names to human-readable labels."""
    m = re.match(r'\[(\w+):NumericalParameterSampler\(([^,]+),\s*([^)]+)\)\]', name)
    if m:
        param, lo, hi = m.group(1), m.group(2).strip(), m.group(3).strip()
        try:
            lo_f = float(lo)
            hi_f = float(hi)
            if lo_f == int(lo_f) and hi_f == int(hi_f):
                lo, hi = str(int(lo_f)), str(int(hi_f))
        except ValueError:
            pass
        return f"{param} {lo}-{hi}"
    return name


def _enforce_length(arr, n):
    """Truncate or zero-pad arr to exactly n elements."""
    arr = list(arr)
    if len(arr) >= n:
        return arr[:n]
    return arr + [0.0] * (n - len(arr))


def _generate_single_ts(option, template_noise, ts_length):
    """Generate one single-TS sample for an Option."""
    from utils.utils import get_ts_obj_from_option, calculate_noise_level
    from utils.error_classes import AnomalyOnePeakError

    ts_obj = get_ts_obj_from_option(option)
    ts = ts_obj.generate(ts_length)
    if option.noise_snr > 0.0:
        noise_level = calculate_noise_level(option.noise_snr, ts_obj, ts_length)
        noise = template_noise(noise_level=noise_level)
        ts = ts + noise.generate(ts_length)
    if hasattr(ts_obj, "transformations"):
        for t in ts_obj.transformations:
            ts = t.transform(ts)
    return _enforce_length(ts, ts_length)


def _generate_two_ts(option, template_noise, ts_length):
    """Generate one two-TS sample for a TwoTSOption or PairTSOption."""
    from utils.utils import (get_ts_obj_from_two_options, get_pair_ts_obj_from_option,
                              calculate_noise_level, TwoTSOption, PairTSOption)
    from utils.error_classes import AnomalyOnePeakError

    if isinstance(option, TwoTSOption):
        ts_obj1, ts_obj2 = get_ts_obj_from_two_options(option)
        ts1 = ts_obj1.generate(ts_length)
        ts2 = ts_obj2.generate(ts_length)
        if option.noise_snr1 > 0.0:
            nl = calculate_noise_level(option.noise_snr1, ts_obj1, ts_length)
            ts1 = ts1 + template_noise(noise_level=nl).generate(ts_length)
        if option.noise_snr2 > 0.0:
            nl = calculate_noise_level(option.noise_snr2, ts_obj2, ts_length)
            ts2 = ts2 + template_noise(noise_level=nl).generate(ts_length)
        if hasattr(ts_obj1, "transformations"):
            for t in ts_obj1.transformations:
                ts1 = t.transform(ts1)
        if hasattr(ts_obj2, "transformations"):
            for t in ts_obj2.transformations:
                ts2 = t.transform(ts2)
    else:  # PairTSOption
        ts_obj = get_pair_ts_obj_from_option(option)
        ts1, ts2 = ts_obj.generate(ts_length)
        if option.noise_snr > 0.0:
            nl = calculate_noise_level(option.noise_snr, ts_obj, ts_length)
            noise = template_noise(noise_level=nl).generate(ts_length)
            ts1, ts2 = ts1 + noise, ts2 + noise

    return [_enforce_length(ts1, ts_length), _enforce_length(ts2, ts_length)]


def generate_variants_for_template(template, num_variants, ts_length, base_seed):
    """
    Generate num_variants TS samples per option for a template.

    Returns a dict:
        {
            'tid': int,
            'question': str,
            'option_names': [str, ...],   # cleaned display names (A->name0, B->name1, ...)
            'category': str,
            'subcategory': str,
            'difficulty': str,
            'is_two_series': bool,
            'ts_variants': {              # option_name -> list of variants
                'name0': [[256 floats], ...],
                'name1': [[ts1_256, ts2_256], ...],  # two-series
            }
        }
    """
    from utils.utils import seed_everything, Option, TwoTSOption, PairTSOption
    from utils.error_classes import AnomalyOnePeakError

    tid = template["tid"]
    is_two_series = any(
        isinstance(opt, (TwoTSOption, PairTSOption)) for opt in template["options"]
    )

    option_names = [clean_option_name(opt.option_name) for opt in template["options"]]
    # Deduplicate names (rare, but some templates reuse the same label text)
    seen = {}
    for idx, name in enumerate(option_names):
        if name in seen:
            option_names[idx] = f"{name}_{idx}"
        seen[name] = idx

    ts_variants = {name: [] for name in option_names}

    for opt_idx, (option, opt_name) in enumerate(zip(template["options"], option_names)):
        max_attempts = num_variants * 5
        collected = 0
        attempt = 0

        while collected < num_variants and attempt < max_attempts:
            seed_everything(base_seed + opt_idx * 10000 + attempt)
            try:
                if isinstance(option, Option):
                    variant = _generate_single_ts(option, template["noise"], ts_length)
                else:
                    variant = _generate_two_ts(option, template["noise"], ts_length)
                ts_variants[opt_name].append(variant)
                collected += 1
            except AnomalyOnePeakError:
                pass
            except Exception as e:
                pass
            attempt += 1

        if collected < num_variants:
            print(f"  WARNING: tid={tid} option '{opt_name}' only got {collected}/{num_variants} variants")

    return {
        "tid": tid,
        "question": template["question"],
        "option_names": option_names,
        "difficulty": template["difficulty"],
        "is_two_series": is_two_series,
        "ts_variants": ts_variants,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate augmented TSE dataset for ICL")
    parser.add_argument("--tse_repo", type=str, default="third_party/TimeSeriesExam")
    parser.add_argument("--output", type=str, default="qa_dataset_augmented.json")
    parser.add_argument("--num_variants", type=int, default=10,
                        help="Number of TS variants to generate per option per template")
    parser.add_argument("--ts_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tids", type=str, default=None,
                        help="Comma-separated list of tids to generate (default: all)")
    args = parser.parse_args()

    tse_repo = os.path.abspath(args.tse_repo)
    if not os.path.isdir(tse_repo):
        raise FileNotFoundError(f"TSE repo not found at {tse_repo}. Clone it first.")
    sys.path.insert(0, tse_repo)

    from question_template import TEMPLATE

    filter_tids = None
    if args.tids:
        filter_tids = set(int(t) for t in args.tids.split(","))

    augmented = []
    skipped = []

    for category, category_dict in TEMPLATE.items():
        for subcategory, subcategory_dict in category_dict.items():
            for template_name, template in subcategory_dict.items():
                tid = template["tid"]
                if filter_tids is not None and tid not in filter_tids:
                    continue

                print(f"Generating tid={tid} ({category} / {subcategory})...")
                try:
                    entry = generate_variants_for_template(
                        template, args.num_variants, args.ts_length, args.seed
                    )
                    entry["category"] = category
                    entry["subcategory"] = subcategory
                    augmented.append(entry)
                except Exception as e:
                    print(f"  ERROR: tid={tid}: {e}")
                    traceback.print_exc()
                    skipped.append(tid)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(augmented, f, indent=2)

    print(f"\nDone. {len(augmented)} templates saved to {args.output}")
    if skipped:
        print(f"Skipped tids: {skipped}")


if __name__ == "__main__":
    main()
