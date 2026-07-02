# third_party/

Vendored external codebases. Do not modify these files — they are upstream sources
kept here for reference and reproducibility.

---

## `TimeSeriesExam/`

Original dataset generation codebase for the **TimeSeriesExam** benchmark
(Cai et al., NeurIPS 2024 Workshop on Time Series in the Age of Large Models).
Source: [AutonLab/TimeSeriesExam](https://huggingface.co/datasets/AutonLab/TimeSeriesExam1),
paper: arXiv 2410.14752, license: MIT.

**This code is not used at inference time.** The pre-generated dataset is
`qa_dataset.json` at the project root (746 questions). This folder is kept so
the generation pipeline can be re-run if needed (e.g. to produce a larger or
differently-parameterised dataset for the demonstration pool).

### Structure

| Path | Purpose |
|------|---------|
| `main.py` | Entry point — samples each template N times to produce the JSON dataset |
| `run.sh` | Example invocation with default hyperparameters |
| `question_template.py` | All 104 expert-curated question templates |
| `timeseries_curation/timeseries_object.py` | Baseline TS primitives (linear, cyclic, etc.) — each has a `generate(length)` method |
| `timeseries_curation/transformations.py` | Transformations applied on top of generated series |
| `timeseries_curation/composer.py` | Composition modules that combine multiple primitives |
| `timeseries_curation/inject_anomalies.py` | Anomaly injection for anomaly-detection templates |
| `utils/utils.py` | Option types: `SingleTSOption`, `TwoTSOption`, `PairedTSOption` |

### Generation hyperparameters

| Param | Description |
|-------|-------------|
| `num_questions_per_option` | Samples drawn per option within each template |
| `ts_length` | Length of generated series (default 128) |
| `output_file` | Path for the output JSON |

### Re-generating the dataset

```bash
cd third_party/TimeSeriesExam
conda activate ts_exam   # Python 3.11, see requirements.txt
sh run.sh
```

Output is a JSON array; copy it to the project root as `qa_dataset.json` and
update `configs/data_paths.yaml` if the path changes.
