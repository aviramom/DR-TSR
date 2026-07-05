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

Given two ranked lists A and B over the same set of candidates:

```
RRF_score(d) = 1 / (k + rank_A(d)) + 1 / (k + rank_B(d))
```

- `rank_A(d)` is the position of candidate `d` in list A (1-indexed)
- `rank_B(d)` is its position in list B (0 if absent — it contributes nothing)
- `k = 60` is a smoothing constant (the value from the original paper)

Items present in only one list receive a partial score from that list only.
Items present in both lists accumulate contributions from both — so consistent
agreement between retrievers boosts a candidate above what either alone would rank it.

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
from retrievers import RRFRetriever, TSRetriever, TextRetriever

retriever = RRFRetriever(
    retriever_a=TSRetriever(device="cpu"),
    retriever_b=TextRetriever(),
    k_rrf=60,          # smoothing constant
    n_candidates=None, # None = use full pool (most correct); set e.g. 4*k for speed
)
retriever.index(pool)
demos = retriever.retrieve(query, k=3)
```

Internally, `retrieve(query, k)`:
1. Calls `retriever_a.retrieve(query, n)` and `retriever_b.retrieve(query, n)`.
2. Assigns RRF scores to every unique candidate across both lists.
3. Sorts all candidates by RRF score descending.
4. Applies the standard template-diversity and same-query exclusion rules
   (same as every other retriever: no same-`tid`, no same-`id` as the query).
5. Returns the top-k.

The `n_candidates` parameter controls how many candidates each sub-retriever
exposes to the fusion step. Using the full pool size is the most correct choice;
lowering it trades recall for speed.
