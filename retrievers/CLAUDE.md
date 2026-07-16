# retrievers/

Demonstration retrieval components for DR-TSR. Every retriever subclasses
`BaseRetriever` and exposes a two-method contract: build an index from a pool
of candidate demonstrations, then retrieve the best k for any given query.

---

## Interface contract

```python
class BaseRetriever(ABC):
    def index(self, pool: List[Dict[str, Any]]) -> None: ...
    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]: ...
```

**`index(pool)`**  
Build whatever internal data structure (embedding matrix, FAISS index, DTW cache,
etc.) the retriever needs. Called once per experiment run before evaluation starts.
`pool` must be disjoint from the test set — callers are responsible for the split.

**`retrieve(query, k)`**  
Return up to `k` items from the indexed pool, ranked best-first. Must never return
the query item itself (match on `item["id"]` if available), and must never return
items from the same question template (match on `item["tid"]` if available).

**Leave-one-out lookup + encoder offload**  
`run_exp.py` indexes the *full* dataset (leave-one-out), so every query's embedding
already exists in the pool matrix. Encoder retrievers therefore:

1. Build an `id → pool row` map at index time (`BaseRetriever._build_id_map`).
2. In `retrieve()`, look up the query's precomputed embedding by `id`; only
   queries absent from the pool are re-encoded (fallback path).
3. Offload the encoder to CPU at the end of `index()`
   (`BaseRetriever._offload_encoder`), freeing all encoder VRAM for the LLM.

This makes `--retriever_device cuda` safe alongside the LLM: `run_exp.py` indexes
the pool on the GPU *before* the LLM loads (critical for vLLM, which preallocates
most of VRAM), and no encoder stays resident during evaluation.

---

## Batch-contract dict fields

Every item passed to `index()` or returned from `retrieve()` uses the same format
as dataset items (see `data_provider/CLAUDE.md`):

| Field | Type | Description |
|-------|------|-------------|
| `input_text` | `str` | Full prompt (TS serialized, or with `<ts><ts/>` placeholders) |
| `input_ts` | `list[list[float]]` | Raw float arrays, one per series |
| `output_text` | `str` | Gold option letter (`'A'`, `'B'`, …) |
| `task_id` | `str` | Dataset identifier |
| `options` | `list[str]` | Valid option letters |

Metadata fields available on TimeSeriesExam items: `id`, `category`,
`subcategory`, `tid` (template id), `difficulty`, `question`, `answer_text`.

---

## Where retrievers are used

1. **`run_exp.py:177`** — `retriever = None` placeholder; replace with an
   instantiated retriever to activate k-shot retrieval.
2. **`evaluations/timeseriesexam_eval.py`** — `evaluate_tse()` calls
   `retriever.retrieve(item, k=num_shots)` then passes the result to
   `build_icl_prompt()`.

Typical experiment setup:

```python
retriever = SomeRetriever(...)
retriever.index(pool_items)          # build index once
# run_exp.py then passes retriever → evaluate_tse → retrieve per query
```

---

## Template exclusion and diversity

TimeSeriesExam questions are generated from a fixed set of templates (`tid` field).
Two invariants are enforced by all retrievers:

1. **Same-template exclusion** — no demonstration may share `tid` with the query.
   Same-template items have near-identical question text, so a text retriever would
   otherwise retrieve k copies of the same question type, teaching the model nothing
   new about the query.

2. **Greedy template diversity** — among the k returned demonstrations, each has a
   distinct `tid`. After ranking by relevance, candidates are selected greedily:
   the first unseen template wins, duplicates are skipped.

Both invariants are implemented in `BaseRetriever._cosine_top_k` (via the
`exclude_tid` parameter) and in `RandomRetriever.retrieve()`.  If `tid` is absent
from the item dicts (e.g. a different dataset), the code degrades gracefully to the
original behaviour.

---

## Retrievers

| Class | File | Signal | Notes |
|-------|------|--------|-------|
| `RandomRetriever` | `random_retriever.py` | none | Random baseline |
| `TextRetriever` | `text_retriever.py` | question text | SentenceTransformer cosine kNN; spec name `text_bge` = same class with `BAAI/bge-large-en-v1.5` |
| `TSRetriever` | `ts_retriever.py` | time series shape | MOMENT embedding cosine kNN. **Truncates to MOMENT's 512-step window** — the three `ts_*` variants below each keep the full series a different way (shared plumbing in `moment_base.py`) |
| `TSCompressRetriever` | `ts_compress_retriever.py` | full-series shape (downsampled) | Spec name `ts_compress`. Strategy A: linearly downsample the whole series to 512 steps, embed once. Keeps global shape/trend, smooths away local spikes |
| `TSMultiVecRetriever` | `ts_multivec_retriever.py` | full-series shape (all windows) | Spec name `ts_multivec`. Strategy B: embed every non-overlapping 512-step window, index all vectors (flat matrix + owner map); query-time MaxSim late interaction (per query window, take each item's best window; average). Nothing lost at index time |
| `TSWindowAggRetriever` | `ts_windowagg_retriever.py` | full-series shape (window average) | Spec name `ts_windowagg`. Strategy C: embed every window, length-weighted average into one vector. Full resolution seen, one vector per item, cross-window blur |
| `VisionTSRetriever` | `vision_ts_retriever.py` | TS rendered as image | DINOv3 CLS embedding cosine kNN |
| `DelayDINORetriever` | `delay_dino_retriever.py` | TS as delay-embedding image | DINOv3 CLS embedding cosine kNN; pool-level min/max normalization, sliding-window delay embedding → 2D image |
| `SpectralRetriever` | `spectral_retriever.py` | FFT magnitude spectrum | Pure numpy, no encoder. rFFT magnitudes (DC dropped) interpolated to a fixed 128-bin grid, top-32 components kept, power-normalized → cosine kNN. Captures global periodicity/rhythm |
| `StatsRetriever` | `stats_retriever.py` | statistical features | Pure numpy/scipy, no encoder. 8 interpretable features (mean, std, trend slope, noise, seasonality strength, kurtosis, lag-1 autocorr, permutation entropy); pool-level z-score per dimension → L2 → cosine kNN |
| `WaveletRetriever` | `wavelet_retriever.py` | TS as CWT scalogram image | Spec name `vision_wavelet`. Morlet CWT (pywt) → magnitude scalogram heatmap → same DINOv3 encoder as `vision_ts`, forming a clean ablation pair that isolates time-frequency localization. Multivariate: per-channel scalograms tiled vertically. Needs `PyWavelets` |
| `SigLIPRetriever` | `siglip_retriever.py` | fused TS-image + question-text | Spec names `siglip_plot` / `siglip_delay`. SigLIP2 (`google/siglip2-base-patch16-224`) maps images and text into one shared space; each item is one fused vector = l2(mean(l2(image emb), l2(text emb))). `siglip_plot` reuses the `vision_ts` line-plot rendering, `siglip_delay` the `delay_dino` delay-embedding rendering (pool min/max, per-series mean) — mirroring the DINOv3 pair so the encoder + text fusion is the isolated variable. Text tower truncates the question to 64 tokens |
| `DTWRetriever` | `dtw_retriever.py` | TS shape via DTW | Spec name `dtw`. The canonical shape-similarity baseline. Pure `tslearn`, no encoder. Each item → z-scored, fixed-length (256) signature (channels concatenated); Sakoe-Chiba-banded DTW distance. Precomputes the full N×N distance matrix once at index time (leave-one-out → O(1) per query); ranks via `_top_k_from_scores` on the negated distances |
| `RRFRetriever` | `rrf_retriever.py` | N fused sub-retrievers | Reciprocal Rank Fusion over an arbitrary list of sub-retrievers (`retrievers=[...]`, N >= 2). This is the MR-RRF fusion from `MRRF_METHOD.md` — no separate class needed |
| `TwoStageRetriever` | `two_stage_retriever.py` | coarse TS → fine text | Spec name `twostage-<a>-<b>`. Stage `<a>` pulls `n_candidates` (default 50, `--stage1_candidates`) by its signal; stage `<b>` re-ranks them by cosine on its own indexed pool vectors (`BaseRetriever.pool_similarity`) and keeps the top k. Stage `<b>` must be a cosine-family retriever. Used as `twostage-delay_dino-text` / `twostage-vision_wavelet-text` |

Fusion combos are selected from the CLI via a **composite spec**, parsed by
`utils/retriever.py:build_retriever`:

```
--retriever rrf-<a>-<b>[-<c>...]
```

where each component is one of `text`, `text_bge`, `ts`, `ts_compress`,
`ts_multivec`, `ts_windowagg`, `vision_ts`, `delay_dino`, `spectral`, `stats`,
`vision_wavelet`, `dtw`, `siglip_plot`, `siglip_delay` (any 2+, no duplicates).
Component names never contain `-`, so the separator is unambiguous and the
spec doubles as the run's filename/W&B tag. Examples: `rrf-ts-delay_dino`,
`rrf-text-vision_ts`, and the full 6-way MR-RRF fusion
`rrf-text-ts-vision_ts-spectral-stats-vision_wavelet` (see `MRRF_METHOD.md`
and `scripts/submit_tse_mrrf_full.sh`).

A second composite form selects the two-stage (coarse-to-fine) retriever:

```
--retriever twostage-<a>-<b>
```

Stage `<a>` (any component above) pulls `--stage1_candidates` (default 50) items
by its signal; stage `<b>` re-ranks them and returns the top k. Stage `<b>` must
be a cosine-family retriever (it re-ranks via `pool_similarity`, which reads its
indexed pool vectors). Examples: `twostage-delay_dino-text`,
`twostage-vision_wavelet-text` (see `scripts/submit_tse_twostage.sh`).

`--retriever rrf` (bare) is a **legacy alias for `rrf-ts-text`** — the combo the
original `retriever_comparison_v1` runs were logged under. `random` cannot be
fused (a random ranking adds nothing but noise to RRF).

The RRF smoothing constant (`k` in `score = sum 1/(k + rank)`) is set via
`--rrf_k` (default 60, ignored unless `--retriever` is an `rrf-...` spec) and
threaded through to `RRFRetriever(k_rrf=...)` by `build_retriever`. It's part
of `--keys_to_match`, so sweeping it under the same `--retriever` spec needs
either distinct `--exp_id`s per value (the convention used by
`scripts/submit_tse_top3_mrrf_variants.sh`, which sweeps the top-3 retrievers
from the head-to-head sweeps — `vision_ts` + `delay_dino` + `vision_wavelet` —
over `rrf_k` in `{10, 60, 100}`) or accepting that W&B dedup will otherwise key
on it too.

See [RRF.md](RRF.md) for a detailed explanation of RRF — what the formula means,
why `k=60`, how it compares to weighted cosine similarity, and why the `ts`+`text`
combo underperforms while shape-only combos are expected to do better.

---

## Adding a new retriever

1. Create `retrievers/<name>.py` subclassing `BaseRetriever`.
2. Implement `index()` and `retrieve()`.
3. Import it in `retrievers/__init__.py` and add it to `__all__`.
4. Wire it into `run_exp.py` (step 7, currently `retriever = None`).

A `utils/retriever.py` registry (mirroring `utils/model.py`) can be added once
there are multiple concrete retrievers to register.
