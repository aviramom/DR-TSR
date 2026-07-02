# data_provider/

Unified dataset wrappers for the DR-TSR benchmark framework. Each file takes
a raw dataset on disk and returns samples in the single batch format that all
`BaseModelWrapper` subclasses in `models/` consume.

---

## Batch Contract

Every item returned by a dataset is a dict with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `input_text` | `str` | Full prompt. In **combined** mode the TS values are serialized as numeric text (`[v1, v2, ...]`). In **separate** mode each series is replaced by a `<ts><ts/>` placeholder. |
| `input_ts` | `list[list[float]]` | Raw float arrays, one per series in the sample, in the same order as the placeholders / serialized blocks in `input_text`. Always populated regardless of mode so models that read raw arrays (e.g. `KNNBaseline`, `ChatTSHFWrapper`) work in both modes. |
| `output_text` | `str` | Correct option letter (`'A'`, `'B'`, …). |
| `task_id` | `str` | Dataset identifier (e.g. `'TimeSeriesExam'`). |
| `options` | `list[str]` | Valid option letters for this sample (e.g. `['A', 'B', 'C']`). |

Additional metadata fields (category, tid, etc.) are dataset-specific and
documented in the per-file sections below.

### input_mode

`input_mode` is passed to the dataset constructor:

- **`"combined"`** — used by `InstructModel`, `LargeInstructModel`,
  `APIModelWrapper`, `RandomBaseline`, `KNNBaseline`.
  `input_text` carries the full numeric TS; `input_ts` is also populated for
  models that need raw arrays (e.g. `KNNBaseline` for DTW).

- **`"separate"`** — used by `ImageInstructModel`, `QwenVLImageModel`,
  `ChatTSHFWrapper`, `ChatTSVLLMWrapper`, `DinoKNNCLSABaseline`.
  `input_text` has one `<ts><ts/>` per series. Image models render each
  array as a matplotlib plot; ChatTS models pass the arrays to their patch
  processor. The number of `<ts><ts/>` tokens must equal `len(input_ts[i])`.

---

## Datasets

### `TimeSeriesExamDataset` — `timeseriesexam_data.py`

Wraps `qa_dataset.json` (746 items from the TimeSeriesExam benchmark).

**Constructor**:
```python
TimeSeriesExamDataset(
    data_path: str = "qa_dataset.json",
    input_mode: str = "combined",   # "combined" | "separate"
    ts_precision: int = 4,          # decimal places for combined serialization
    num_samples: int | None = None, # cap for quick smoke tests
)
```

**Extra metadata fields per item**:

| Field | Description |
|-------|-------------|
| `answer_text` | Full text of the correct answer (original dataset value). |
| `question` | Raw question text (without prompt scaffolding). |
| `category` | TSE category: `Pattern Recognition`, `Noise Understanding`, `Anolmaly Detection`, `Similarity Analysis`, `Causality Analysis`. |
| `subcategory` | Finer-grained subcategory (13 total). |
| `tid` | Template ID (1–104) — used for cross-template splits in Experiment 1. |
| `id` | Unique item ID. |
| `difficulty` | `'easy'` / `'medium'` / `'hard'`. |

**Series layout**:

- Single-series items: `ts` is set; `ts1`/`ts2` are `None`.
  → `input_ts = [[v, ...]]`  (one array)
  → one TS block / placeholder in `input_text`

- Two-series items (similarity / causality questions): `ts1` and `ts2` are
  set; `ts` is `None`.
  → `input_ts = [[v, ...], [u, ...]]`  (two arrays in order)
  → two TS blocks / placeholders in `input_text`

**API**:
```python
ds = TimeSeriesExamDataset("qa_dataset.json", input_mode="combined")
item = ds[0]           # single item dict
batch = ds.as_batch()  # full dataset collated for model.generate()
batch = ds.as_batch(indices=[0, 3, 7])  # subset
tids  = ds.get_field("tid")             # one field across all items
```

**Prompt format** (combined, single-series):
```
Time Series: [1.0680, -0.3076, ...]

Question: What is the type of the trend of the given time series?
A) Exponential
B) Linear
C) No Trend

Return ONLY the label as one of: [A, B, C]
```

---

## Adding a New Dataset

1. Create `data_provider/<name>_data.py` with a class that:
   - Accepts `input_mode` as a constructor argument.
   - Returns items matching the batch contract above.
   - Sets `TASK_ID` to a unique string.
2. Export the class from `data_provider/__init__.py`.
3. Add an entry in this file under **Datasets**.
