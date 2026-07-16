# Multi-Representation RRF (MR-RRF) — Method Description

## Core Idea

A time series can be represented in multiple fundamentally different ways — a rendered plot image, a raw numeric sequence, a frequency-domain spectrum, and a set of statistical features. Each representation captures genuinely different aspects of the same underlying signal. No other modality (text, image) has this property.

Current retrieval methods for time series pick one representation and retrieve by similarity in that space. This work proposes **MR-RRF**: Reciprocal Rank Fusion over multiple complementary representations of the same time series, combined with the text (question) representation. The claim is that fusing ranked lists from all representations outperforms any single representation or pairwise fusion, and that different representations are differentially informative for different reasoning categories (periodicity questions → spectral retrieval; noise questions → statistical retrieval; pattern questions → visual retrieval; causality/skill questions → text retrieval).

---

## The Five Retrieval Signals

For each item in the pool (a time series + question + correct answer), compute and store five embeddings:

### 1. Visual TS embedding (`vision_ts`)
- Render the time series as a plot image (same rendering pipeline already in use)
- Embed with a frozen vision encoder (DINOv3 or similar)
- Similarity: cosine

### 2. Numeric TS embedding (`ts`)
- Encode the raw numeric sequence with a frozen time series encoder (MOMENT or similar)
- Similarity: cosine or Euclidean

### 3. Spectral embedding (`spectral`)
- Apply FFT to the series (or each channel independently for multivariate)
- Take the magnitude spectrum (discard phase)
- Optionally: keep only the top-K frequency components (e.g. top 32) to reduce dimensionality and noise
- Normalize by total power so scale differences don't dominate
- Similarity: cosine in the magnitude spectrum space
- **Why**: captures periodicity, dominant frequencies, and rhythmic structure — things neither DINOv3 nor MOMENT reliably encode
- **No model needed**: pure numpy, no pretrained encoder required

### 4. Statistical feature embedding (`stats`)
- Compute a small vector of interpretable time series features per series:
  - Trend slope (linear regression coefficient, normalized)
  - Noise level (std of first-order differences, normalized)
  - Seasonality strength (ratio of seasonal component variance to total variance, e.g. via STL or simple autocorrelation at lag=period)
  - Mean and std (after instance normalization, these are 0 and 1 — use pre-normalization values)
  - Kurtosis (heavy tails / spikiness)
  - Autocorrelation at lag 1 (short-term memory)
  - Approximate entropy or permutation entropy (complexity/regularity)
- Stack into a single feature vector, L2-normalize
- Similarity: cosine
- **Why**: captures statistical character of the series — noise level, trend, regularity — which is directly tested in several TimeSeriesExam categories
- **No model needed**: scipy/statsmodels, no pretrained encoder required

### 5. Text embedding (`text`)
- Embed the question + options as a single string with a frozen sentence embedder
- Use a strong embedder: `BAAI/bge-large-en-v1.5` or `intfloat/e5-base-v2` (not MiniLM)
- Similarity: cosine
- **Why**: captures the reasoning skill being tested, independent of series content

### 6. Wavelet embedding (`vision_wavelet`)
- Compute the Continuous Wavelet Transform (CWT) of the series using the Morlet wavelet (`pywt.cwt` with `wavelet='morl'`)
- Use a log-spaced range of scales covering the relevant frequency range for the series length (e.g. `scales = np.geomspace(1, len(series)//2, num=64)`)
- Take the magnitude of the complex CWT coefficients to produce a 2D scalogram (scales × time)
- Normalize the scalogram to [0, 1] for consistent rendering
- Render as a heatmap image (same dimensions as the plot images used for `vision_ts`)
- Embed with the same frozen vision encoder as `vision_ts` (DINOv3 or similar)
- Similarity: cosine
- **Why**: FFT gives global frequency content but loses temporal localization — it cannot distinguish "periodicity throughout" from "periodicity only in the first half." The wavelet scalogram captures *when* each frequency is present, making it sensitive to localized anomalies, regime changes, and burst patterns that FFT is blind to. Using DINOv3 on the scalogram image reuses the existing vision pipeline with no new encoder downloads.
- **Implementation note**: for multivariate series, either (a) compute the scalogram of the mean series, (b) compute per-channel scalograms and tile them into a single image, or (c) use the channel with highest variance. Option (b) is most informative but produces a wider image — ensure the vision encoder's patch size handles it gracefully.
- **Relationship to `vision_ts`**: same encoder, different input. This creates a clean ablation pair — any difference in retrieval quality between `vision_ts` and `vision_wavelet` isolates what time-frequency localization adds over raw visual appearance.
- **Dependencies**: `pywt` (PyWavelets) — lightweight, no GPU required

---

## Fusion: Reciprocal Rank Fusion

Given a query, retrieve a ranked list of candidates from each of the six signals independently, then fuse using standard RRF:

```
RRF_score(candidate) = sum over all retrievers r of: 1 / (k + rank_r(candidate))
```

where `k = 60` (standard smoothing constant from Cormack et al. 2009), and `rank_r(candidate)` is the candidate's rank in retriever r's list (1-indexed; candidates not in a retriever's list contribute 0).

Re-rank all candidates by RRF score descending. Apply the standard exclusion rules (no same `tid` as query, no same `id` as query). Return top-k.

### Key implementation notes
- Each sub-retriever should expose its **full ranked pool** to the fusion step (not just top-k), for maximum recall. If pool size makes this slow, expose top-M where M >> k (e.g. M = 4 × pool_size, or the full pool).
- All five sub-retrievers are frozen — no training, no fine-tuning, no gradient steps anywhere in this pipeline.
- The RRF weight is equal across all five signals (uniform). A natural extension is to weight by per-category historical accuracy (see Section below), but start with equal weights.

---

## Optional Extension: Representation-Aware Re-ranking

After MR-RRF produces a shortlist of top-M candidates, apply a lightweight re-ranking step:

1. Classify the query's category from its text embedding (nearest centroid of each category's pool embeddings — no training, just a lookup from your per-category accuracy table).
2. Look up which sub-retriever has historically performed best for that category (from the per-category ablation results).
3. Boost candidates that ranked highly in that sub-retriever's individual list: add a small bonus `β / (k + rank_best_retriever(candidate))` to each candidate's RRF score, where β is a scalar (tune on a held-out validation slice, typical range 0.5–2.0).
4. Re-sort and return top-k from the boosted scores.

This remains training-free — the boost weights come directly from your per-category accuracy tables, not from any learned model. It is the bridge between pure RRF and a trained modality router.

---

## What to Vary in Experiments

- **Ablation over retriever subsets**: test all singletons + all pairs + key triples + the full 6-way fusion to identify which representations contribute and which are redundant. Key comparisons: `vision_ts` vs `vision_wavelet` (isolates time-frequency localization), `spectral` vs `vision_wavelet` (two frequency-domain signals, different representations), full 6-way vs best 5-way (does each signal add marginal value).
- **k values**: {1, 2, 3, 5, 8} shots — same as existing experiments.
- **Per-category breakdown**: report accuracy per TimeSeriesExam category for each condition. The hypothesis is that the gain from spectral retrieval is concentrated in noise/periodicity questions, and the gain from statistical retrieval is concentrated in pattern/anomaly questions.
- **Pool size for sub-retrievers**: test full pool vs. top-M shortlist per sub-retriever to check if truncation hurts.
- **Text embedder quality**: compare MiniLM (current) vs. BGE-large or E5 within the MR-RRF framework.

---

## What This Does NOT Require

- No LLM calls during retrieval
- No gradient updates or fine-tuning of any encoder
- No labeled utility data
- No additional model downloads beyond what is already running (spectral and statistical signals are pure numpy/scipy; wavelet scalogram uses the existing vision encoder)
- Only new dependency: `pywt` (PyWavelets) for the CWT computation

---

## Expected Outcome and Paper Claim

The method tests the hypothesis: **for time series reasoning tasks, the multiple non-equivalent representations of a time series are complementary retrieval signals, and fusing all of them outperforms any single representation or pairwise combination.**

Supporting evidence would be:
1. MR-RRF (6-way) > best individual retriever across models
2. MR-RRF (6-way) > best pairwise RRF (e.g. vision_ts + text)
3. Per-category analysis shows different representations dominate for different question types
4. The spectral, statistical, and wavelet signals add value beyond visual and numeric encoders
5. `vision_wavelet` outperforms `vision_ts` on anomaly detection and pattern questions with localized changes — isolating the value of time-frequency localization

If (1) and (2) hold, the method is a clean, training-free contribution. If (3) holds as well, the analysis section alone is a publishable finding about which aspects of a time series are informative for which reasoning skills — independent of whether the accuracy gains are large.

---

## Implementation Checklist for Claude Code

- [ ] Implement `SpectralRetriever`: FFT → magnitude spectrum → top-K components → normalize by total power → cosine kNN
- [ ] Implement `StatisticalRetriever`: feature vector (trend, noise, seasonality, kurtosis, autocorr, entropy) → L2-normalize → cosine kNN
- [ ] Implement `WaveletRetriever`: CWT with Morlet wavelet (`pywt.cwt`) → magnitude scalogram → normalize to [0,1] → render as heatmap image → embed with same frozen vision encoder as `vision_ts` → cosine kNN
- [ ] Upgrade `TextRetriever` to use `BAAI/bge-large-en-v1.5` (or make embedder configurable)
- [ ] Implement `MRRRFRetriever`: takes a list of sub-retrievers, runs all, applies RRF formula (`k=60`), returns top-k after exclusion rules
- [ ] Extend evaluation loop to support configurable list of retrievers (singletons, pairs, and full 6-way fusion at minimum)
- [ ] Add per-category accuracy breakdown to the evaluation output
- [ ] (Optional) Implement `RepresentationAwareReranker` as a post-processing step on top of MR-RRF
- [ ] Cache all embeddings to disk per retriever type — label cache files by retriever name and embedder version for reproducibility
- [ ] Enforce exclusion rules (no same `tid`, no same `id`) after fusion, not inside each sub-retriever independently
- [ ] For `WaveletRetriever` on multivariate series: default to per-channel scalograms tiled into a single image; fall back to mean-series scalogram if tiling produces an image size the vision encoder cannot handle
