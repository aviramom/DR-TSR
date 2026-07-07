# Reciprocal Rank Fusion (RRF)

## What is it?

RRF is a method for combining multiple ranked lists into a single ranking.
Instead of combining raw scores (which requires the scores to be on the same scale),
it combines **rank positions** — which are always comparable, regardless of the
underlying similarity metric.

It was introduced in:
> Cormack, Clarke, Buettcher (2009). *Reciprocal Rank Fusion outperforms Condorcet and
> individual Rank Learning Methods.* SIGIR.

---

## The formula

Given N ranked lists over the same set of candidates:

```
RRF_score(d) = sum_i 1 / (k + rank_i(d))
```

- `rank_i(d)` is the position of candidate `d` in list `i` (1-indexed; contributes
  nothing if `d` is absent from that list)
- `k = 60` is a smoothing constant (the value from the original paper)

Items present in only one list receive a partial score from that list only.
Items present in more lists accumulate more contributions — so consistent
agreement between retrievers boosts a candidate above what any one alone would rank it.
The classic two-retriever case is just N=2.

---

## Why `k = 60`? Why not just `1 / rank`?

Using `1 / rank` gives rank-1 a weight of 1.0 and rank-2 a weight of 0.5 — a 2×
difference for a single position gap. That's too steep; it makes the top result
dominate even when the evidence is marginal.

Adding `k` flattens the curve near the top:

| rank | 1/rank | 1/(60+rank) |
|------|--------|-------------|
| 1    | 1.000  | 0.0164      |
| 2    | 0.500  | 0.0161      |
| 5    | 0.200  | 0.0154      |
| 10   | 0.100  | 0.0143      |
| 60   | 0.017  | 0.0083      |

With `k = 60`, ranks 1–10 are treated as nearly equivalent, meaning "ranked in the
top 10 by both retrievers" beats "ranked #1 by only one." This is exactly the
behavior you want for fusion: **consensus beats dominance**.

The value `k = 60` is empirically robust and rarely needs tuning.

---

## Why not use weighted cosine similarity instead?

Weighted cosine (`α · sim_ts + (1-α) · sim_text`) operates on raw scores.
The problem: different embedding spaces have different score distributions.
MOMENT TS embeddings might cluster in [0.85, 0.99] while sentence embeddings
spread across [0.2, 0.8]. A naive weighted sum is dominated by whichever space
has higher variance — not because it is more informative, but because the numbers
are bigger.

You could normalize each distribution per query, but that adds complexity and
the normalization itself is sensitive to pool size and outliers.

RRF sidesteps all of this: ranks are always integers starting from 1, regardless
of what the underlying model scores look like.

---

## How `RRFRetriever` works in this codebase

```python
from retrievers import RRFRetriever, TSRetriever, TextRetriever, VisionTSRetriever, DelayDINORetriever

retriever = RRFRetriever(
    retrievers=[TSRetriever(device="cpu"), DelayDINORetriever(device="cpu")],
    k_rrf=60,          # smoothing constant
    n_candidates=None, # None = use full pool (most correct); set e.g. 4*k for speed
)
retriever.index(pool)
demos = retriever.retrieve(query, k=3)
```

Any number of sub-retrievers (>= 2) can be fused — pass a longer list, e.g.
`[TSRetriever(), VisionTSRetriever(), DelayDINORetriever()]` for a 3-way fusion.

Internally, `retrieve(query, k)`:
1. Calls `retriever.retrieve(query, n)` on every sub-retriever.
2. Assigns RRF scores to every unique candidate across all lists.
3. Sorts all candidates by RRF score descending.
4. Applies the standard template-diversity and same-query exclusion rules
   (same as every other retriever: no same-`tid`, no same-`id` as the query).
5. Returns the top-k.

The `n_candidates` parameter controls how many candidates each sub-retriever
exposes to the fusion step. Using the full pool size is the most correct choice;
lowering it trades recall for speed.

---

## Fusing the wrong signals hurts more than it helps

The original `rrf` combo (`ts` + `text`) fuses TS-shape similarity with question-text
similarity — two largely uncorrelated signals for a task that's fundamentally about
time-series shape reasoning. Empirically it underperformed *both* of its inputs across
every model (see `analysis/retriever_comparison.ipynb`): fusing a strong signal with a
mismatched one dilutes rather than reinforces, since RRF assumes the lists being fused
are all plausibly relevant to the same latent relevance signal.

The three shape-based retrievers (`ts`, `vision_ts`, `delay_dino`) were consistently the
top individual performers, meaning they already partially agree on good neighbors —
exactly the precondition RRF needs to help (agreement compounds, disagreement doesn't
hurt any single ranking).

Fusion combos are selected from the CLI via `--retriever rrf-<a>-<b>[-<c>...]`
(any 2+ distinct components from `{text, ts, vision_ts, delay_dino}`; parsed by
`utils/retriever.py:build_retriever`). Bare `rrf` is a legacy alias for
`rrf-ts-text`. The sweep in `scripts/submit_tse_rrf_combos.sh` covers all pairs
plus the 3-way shape fusion:

| Spec | Fuses | Rationale |
|---|---|---|
| `rrf` (= `rrf-ts-text`) | MOMENT + question text | Original combo — underperformed both inputs; kept for backward compatibility with existing results |
| `rrf-ts-delay_dino` | MOMENT + delay-embedding DINO | Most representationally distinct pair among the strong performers (raw numeric embedding vs. delay-embedding image, different backbones) |
| `rrf-ts-vision_ts` | MOMENT + line-plot DINO | Rendered line-plot embedding vs. raw numeric embedding — also good diversity |
| `rrf-vision_ts-delay_dino` | line-plot + delay-embedding DINO | Both vision-backbone embeddings of a rendered series — most likely redundant, lowest expected lift |
| `rrf-text-vision_ts` | question text + line-plot DINO | Does text pair better with a shape signal it disagrees with less? Control for the ts+text failure |
| `rrf-text-delay_dino` | question text + delay-embedding DINO | Same control with the other image representation |
| `rrf-ts-vision_ts-delay_dino` | all three shape signals | Tests whether more agreement compounds further |
