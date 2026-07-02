# Utility-Aware Demonstration Retrieval for In-Context Time Series Reasoning

*Working draft — research proposal*

---

## 1. Problem Statement

Recent benchmarks have revealed a fundamental gap in time series understanding. TSRBench (Yu et al., ICML 2026), a large-scale evaluation covering 4,125 problems across 14 real-world domains and 15 task types — spanning perception, reasoning, prediction, and decision-making — evaluated over 30 models including state-of-the-art LLMs, vision-language models, and specialized time series LLMs. The results are sobering: even the strongest models fail to reliably reason about time series, and crucially, combining textual and visual representations of the same series does not produce the expected gains — current multimodal models fail to effectively fuse these complementary signals. The conclusion is not that better pretraining alone will close this gap, but that the bottleneck lies in *how models are guided to reason* about temporal data at inference time.

This motivates a different question: rather than asking *how do we train a better model*, can we ask *how do we get more out of a fixed, frozen model* by choosing more carefully what we show it at inference time? In-context learning (ICL) — where the model is given a small number of worked (input, output) demonstrations in its prompt without any gradient update — has proven to substantially improve LLM performance across a wide range of tasks and domains. Whether this benefit extends reliably to time series reasoning is an open question this work aims to address, but the strong ICL gains observed elsewhere make it a well-motivated starting point. The leverage point shifts from the model itself to the *selection of demonstrations*: what examples, shown in what combination, maximally activate the reasoning capability already present in a frozen LLM.

However, context length is a scarce and costly resource. The number of demonstrations a model can be shown is bounded both by context-window limits and by the cost of longer prompts. This raises a concrete, underexplored question:

> **Given a fixed budget of k demonstrations, how should we select which k examples from a known pool of (time series, question, answer) triples to show the model, in order to maximize accuracy on a new, unseen query?**

The default approach — used throughout the existing time-series-RAG literature — is to retrieve the k most *similar* examples to the query, by some shape-based or embedding-based distance. We hypothesize that similarity is a weak and indirect proxy for what actually matters: whether an example, once placed in context, changes the model's probability of producing the correct answer. This gap between "similar" and "useful" is well documented in the NLP in-context-learning literature, but has not yet been systematically studied for time series reasoning tasks, particularly in a setting where the query consists of *both* a time series and a natural-language question about it.

### 1.1 Why this matters

- **Practical**: efficient demonstration selection reduces the number of tokens needed in context to reach a given accuracy, lowering inference cost and latency — important for any production system where time-series-literate LLMs are used as a reasoning layer over structured data (finance, healthcare, energy, industrial monitoring).
- **Data scarcity**: time series reasoning benchmarks of this kind are typically small (hundreds to low thousands of labeled items, not the millions needed to train a deep learning model for the task from scratch). In such low-sample domains, training a dedicated supervised model end-to-end is not a realistic option. The practical alternative is to rely on a general-purpose pretrained model (an LLM that already has broad reasoning ability) and improve its task performance through smart demonstration/shot selection rather than through additional training. This reframes the research problem from "how do we train a better model" to "how do we get more out of a fixed, frozen model with a limited number of well-chosen examples" — which is exactly the retrieval/selection question this proposal addresses.
- **More is not always better**: empirical evidence across text and multimodal ICL consistently shows that increasing the number of demonstrations does not monotonically improve performance — and can actively hurt it. Lu et al. (2021) showed that the same set of examples in a different order can swing accuracy dramatically, implying quality and arrangement matter more than quantity. Studies of multimodal ICL find that performance often degrades with more demonstrations under distribution shift, as models begin copying surface answer patterns rather than reasoning from the examples. This means the context budget is not merely a cost constraint — it is also a *quality* constraint. Filling it with the wrong examples is worse than using fewer. Demonstration *selection* is therefore not an optional optimization but a necessary component of any ICL system: the question is not how many examples to show, but which ones.
- **Scientific**: it is currently unknown whether LLM time-series reasoning is better supported by demonstrations that are similar in *series shape*, similar in *task/skill type*, or some combination — and whether the relative importance of these two signals is stable across reasoning categories (pattern recognition vs. causality, for instance). Establishing this is a prerequisite for any principled retrieval design in this space.

---

## 2. Related Work

### 2.1 Retrieval-augmented time series forecasting (numeric RAG)

A recent cluster of papers studies retrieval augmentation for *point/probabilistic forecasting*, where retrieved historical series segments are fused numerically into a time series foundation model (TSFM):

- **RAF** (Tire et al., 2024) — first paper to study RAG inside TSFMs; retrieves nearest-neighbor series by Chronos-encoder embedding distance and concatenates the retrieved context+future with the query series. Shows retrieval gains emerge with model scale.
- **RAFT** (Han, Lee, Cha, Arik, Yoon; ICML 2025) — retrieves by raw time-domain similarity (e.g. Pearson correlation) directly from the training corpus; reports an 86% multivariate / 80% univariate win ratio over baselines.
- **TS-RAG** (Ning et al.; NeurIPS 2025) — freezes a TSFM backbone (Chronos-Bolt) and adds a learnable Adaptive Retrieval Mixer (attention + MoE-style gating) over top-k retrieved future horizons. Strongest reported numeric-RAG result (≈3.5% average MSE improvement, up to 6.84% peak) at very low retrieval latency via FAISS indexing.
- **TimeRAF** (Zhang et al.) — end-to-end learnable retriever with channel-wise fusion ("Channel Prompting") for zero-shot forecasting.
- **RATD** (Liu et al.; NeurIPS 2024) — retrieval-guided diffusion model for probabilistic forecasting; retrieved series guide the denoising process rather than being concatenated into a deterministic input.

**Limitation relevant to our work**: in all of the above, retrieval is *similarity-based* (embedding distance, DTW, or correlation) and the downstream task is *forecasting*, not text-mediated reasoning. None of them optimize the retriever against a measured downstream-utility signal, and none operate in a setting where the query includes free-form natural language.

### 2.2 LLM-prompt-based retrieval for time series

- **TimeRAG** (Yang et al.) — builds a knowledge base via clustering, retrieves by DTW, serializes retrieved series into a text prompt for an LLM forecaster.
- **FinSrag / FinSeer** (Xiao et al.) — domain-specific (financial) retriever trained to align queries with *influential* historical sequences, filtering noise; the first work in this space to move past pure shape similarity toward a learned, somewhat utility-flavored retrieval signal — though trained via similarity-of-influence rather than direct downstream feedback.

### 2.3 Multimodal time-series/text retrieval

- **TRACE** (Chen, Zhao, Nurbek, et al.; NeurIPS 2025) — learns a joint embedding space aligning multivariate time series with paired textual *descriptions* (e.g. weather event narratives) via channel-level contrastive learning and hard-negative mining. Demonstrates that learned multimodal alignment dramatically outperforms naive similarity retrieval (≈90% vs. 55–65% top-1 label matching against frozen TSFM embeddings; Euclidean/DTW baselines at 0.38–0.55 P@1 vs. 0.90 for TRACE on TS-to-TS retrieval).
  - **Key relevance**: TRACE is strong evidence that (a) encoder-only TS architectures (e.g. MOMENT) retrieve better than encoder-decoder ones (e.g. Chronos), and (b) hard-negative mining against "misleadingly similar" distractors meaningfully improves retrieval quality.
  - **Key limitation relevant to our work**: TRACE's supervision signal is *content alignment* (does this text describe this series), not *task utility* (does this example help answer a different query). The text in their setting is a description; the text in ours is a question about a different aspect of the series than mere description. This is a different learning problem and their method cannot be applied unmodified.

### 2.4 In-context example selection (NLP literature)

A separate, more mature literature studies exactly our research question — which k examples to retrieve for ICL — for text tasks:

- **KATE** (Liu et al., 2021) — first to show kNN similarity retrieval (over random selection) improves GPT-3 ICL accuracy. Similarity-only baseline.
- **EPR** (Rubin et al., 2022) — trains a retriever using a signal derived from output-label similarity, not just input similarity — a first step toward utility-aware retrieval.
- **LLM-R** (Wang et al., 2023) — trains a reward model from actual LLM feedback on candidate demonstrations, then distills it into a fast dense retriever via knowledge distillation. The closest existing template to what we propose; demonstrated consistent gains across 30 tasks and generalization to unseen tasks.
- **RetICL** — frames example selection as a sequential decision process (a Markov decision process solved via reinforcement learning), explicitly modeling that the value of an example depends on what's already been selected (non-independence of the demonstration set).
- **Vote-k / selective annotation** (Su et al., 2022) — diversity-aware selection that avoids redundant demonstrations, optimizing coverage of the candidate space rather than pure closeness to the query.
- **Lu et al. (2021)** — demonstrates strong sensitivity of ICL accuracy to demonstration *order*, independent of which examples are chosen.

**Gap we address**: none of this literature has been applied to time series reasoning, where the input has a structurally distinct modality (numeric sequences) alongside text, and where "similarity" can be computed in (at least) two genuinely different and only partially correlated spaces — series shape and task/skill semantics.

### 2.5 Retrieval-augmented ICL in vision-language models

A close cross-domain parallel exists in multimodal (vision-language) large models, confirming that the retrieval-augmented ICL paradigm transfers across modalities and giving a useful contrast point for our own design:

- **RAICL** (2025) — explicitly named "Retrieval-Augmented In-Context Learning," applied to disease classification with multimodal LLMs (Qwen, Llava, Gemma) on pathology (TCGA) and chest X-ray data. Retrieves demonstrations using embeddings from multiple encoders per modality (ResNet for images; BERT/BioBERT/ClinicalBERT for text) and compares similarity metrics, finding Euclidean distance better for accuracy and cosine similarity better for macro-F1 — a useful empirical data point for our own metric choice. Reports accuracy gains of 0.7854→0.8368 (TCGA) and 0.7924→0.8658 (chest X-ray), in a similar range to the time-series-RAG results in Section 2.1.
  - **Key limitation relevant to our work**: like RAF, RAFT, and TimeRAG on the time series side, RAICL selects demonstrations purely by similarity — there is no learned utility scorer and no signal derived from measured downstream effect on model output. It establishes that retrieval-augmented ICL helps in multimodal classification, but stops at the same similarity-only baseline our work aims to move beyond.
- **DRUM** (Learning Demonstration Retriever for Large MUlti-modal Models, 2024) — the closest existing methodological cousin to our learned-scorer approach in the vision-language space: rather than using an off-the-shelf similarity embedding (e.g. CLIP), DRUM explicitly fine-tunes a vision-language embedding model to learn which demonstrations retrieve well, mirroring the move from pure similarity retrieval to learned, trained retrieval that we propose for the time-series setting.
- **Advancing Multimodal ICL with Task-aware Demonstrations** — directly builds on LLM-R (Wang et al., 2023; Section 2.4), adapting its learned-retrieval approach from text-only LLMs to vision-language models. This is precedent that LLM-R's approach has already been successfully ported across a modality boundary once, supporting the validity of porting it to time series as we propose.

**Cautionary finding worth noting**: several studies in this space (e.g. work on "true" multimodal ICL and on what VLMs attend to in context) report that vision-language models frequently underutilize the actual visual content of demonstrations, leaning instead on textual/label patterns and largely ignoring the image itself, even when given multimodal demonstrations. This is a direct parallel to a risk in our own setting: an LLM given a time-series demonstration may lean on the textual pattern of the question far more than on the series itself. This reinforces the motivation for keeping the TS and text retrieval signals separate and measurable rather than assuming both modalities contribute equally by default.

### 2.6 Time series understanding benchmarks (evaluation substrate)

- **TimeSeriesExam** (Cai et al., 2024) — 763 multiple-choice questions generated from 104 expert-curated templates across five categories (pattern recognition, noise understanding, anomaly detection, similarity/comparative analysis, causality analysis), with subcategories within each. Each instance is paired with a series (length 128) and, in some templates, an additional shorter example series. Designed for *evaluation*, not training, and was not released with a train/test split.

- **TSQA / Time-MQA** (Kong et al., 2025) — a large-scale dataset of approximately 200,000 question-answer pairs spanning 12 real-world domains (healthcare, environment, energy, finance, transport, IoT, nature, human activities, AIOps, web, and others) and five task types. Within the open-ended reasoning subset, the dataset includes 11,281 multiple-choice questions — the portion relevant to this work. Built from real-world time series rather than synthetic templates, making it substantially more diverse in both series content and question phrasing than TimeSeriesExam. Fully open-sourced on HuggingFace (`Time-MQA/TSQA`). We propose using the MCQ subset of TSQA as the primary source for the demonstration pool and scorer training data.

- **TSRBench** (Yu et al., ICML 2026) — a large-scale evaluation benchmark covering 4,125 problems across 14 real-world domains and 15 task types organized into four dimensions: perception, reasoning, prediction, and decision-making. Uses multiple-choice, true/false, and open-ended formats across tasks; the MCQ-format subset is the portion directly applicable to this work. Notably, 58.1% of problems involve multivariate series. A key finding from the benchmark's own evaluation is that current multimodal models fail to effectively fuse textual and visual representations of the same series — directly motivating the modality-isolation design of this work. We propose using the MCQ subset of TSRBench as the primary held-out evaluation benchmark, kept entirely separate from pool construction and scorer training.

---

## 3. Proposed Approach

We propose a two-stage pipeline that separates *retrieval* from *selection*, keeping the two modalities independent throughout so their individual contributions can be measured.

**Stage 1 — Two independent retrieval indices.** Every item in the demonstration pool is embedded into two separate vector spaces: one by a time series encoder (capturing series shape), one by a sentence embedder (capturing question and skill semantics). These two indices are kept strictly separate rather than fused into a single multimodal space, so that the contribution of each modality can be isolated and compared before any combination is attempted. At inference, both indices are queried in parallel to produce a shortlist of candidate demonstrations.

**Stage 2 — Utility-aware scoring and selection.** A lightweight scorer is trained to predict how useful each shortlisted candidate will be for a given query — not how similar it looks, but how much inserting it as a demonstration changes the model's probability of producing the correct answer. The training signal for this scorer is derived directly from the LLM itself: for each (query, candidate) pair in a held-aside labeling set, we measure the change in log-probability assigned to the correct option when the candidate is inserted as a single demonstration (the "utility" or "benefit" of that candidate). A scorer is then trained to predict this utility from the similarity features of the (query, candidate) pair. At inference, shortlisted candidates are ranked by predicted utility and the top-k are selected for the final prompt. A diversity penalty (e.g. Maximal Marginal Relevance) is applied at selection time to avoid redundant demonstrations consuming the context budget.


---

## 4. Open Questions and Challenges

The proposed approach raises several non-trivial research questions that require careful empirical investigation. We organize them by component.

### 4.1 The Scorer

**Handling variable-length inputs.** Both modalities vary in length across benchmarks. Time series differ in the number of time steps, and questions differ in length depending on how much context, how many options, and how much descriptive text they contain. Whether utility is length-sensitive in either modality — and how to normalize or handle length mismatch between a candidate and a query — is an open design question. A candidate with a very short series or a brief question may not transfer usefully to a query with a long, complex series and a verbose, multi-part question, even if the content is superficially similar. Embedding models handle this implicitly through pooling or truncation, but it is not guaranteed that the resulting fixed-length vectors preserve length-relevant similarity faithfully.

**Is utility additive?** The scorer is trained on single-candidate utility (one demonstration at a time), implicitly assuming that the utility of a set of k demonstrations approximates the sum of individual utilities. This is almost certainly false in general: two demonstrations with low individual utility may together cover complementary aspects of a query's reasoning requirement and produce high combined utility, while two high-utility demonstrations that are near-duplicates may together add little more than one alone. How large this gap is in practice — and whether greedy diversity-penalized selection (MMR) is sufficient to close it — is an open empirical question.

**Scorer transferability across LLMs.** The utility labels used to train the scorer are derived from a specific LLM's log-probabilities. Whether a scorer trained on one model's feedback generalizes to a different LLM is unclear. If utility is largely a property of the demonstration's *task relevance* (which should transfer), the scorer may generalize well. If it captures model-specific preferences (formatting, token patterns, instruction style), it may not. This question has implications for how broadly the method can be deployed without retraining the scorer per model.

### 4.2 The Data

**What data to use for the pool, scorer training, and evaluation.** These three roles require genuinely disjoint data sources to avoid leakage. The pool (retrieval candidates) and scorer training queries should come from one or more source benchmarks (e.g. TSQA, TimeSeriesExam); the final evaluation should come from an independent benchmark built by a different team (e.g. TSRBench). Choosing the right sources for each role — and verifying that their task coverage and domain distribution are compatible — is a non-trivial design decision.

**Template duplication and shortcut learning.** Many time series QA benchmarks are generated from a fixed set of question templates, meaning multiple items share near-identical phrasing. A retriever trained or evaluated on such data risks learning to match templates rather than task-relevant reasoning patterns — a form of shortcut learning. Mitigation strategies include cross-template splits (holding out entire templates from the retrieval pool), explicit diversity constraints at selection time, and paraphrase augmentation to increase textual variety within each template family.

**Diversity of demonstrations in the pool.** Even with a good scorer, the quality of selected demonstrations is bounded by the diversity of what is available in the pool. If the pool is narrow in domain, task type, or phrasing, even the best scorer cannot compensate. Characterizing what pool composition leads to the best downstream gains — and whether a heterogeneous multi-source pool outperforms a single-domain one — is an open question.

### 4.3 The Retriever

**Which time series encoder to use.** The time series embedding space used for initial retrieval is a critical design choice. Options range from classical time series encoders (trained on numeric forecasting/classification objectives) to vision encoders applied to plotted series images (treating the series as a visual object). These two approaches capture fundamentally different representations — one numeric and structural, one perceptual and visual — and it is not obvious which better predicts whether a demonstration will be useful for a reasoning task.

**Which modality drives utility more — series shape or question semantics?** A central empirical question of this work is whether the text side (question + options) or the time series side contributes more to predicting which demonstrations are useful. The NLP-ICL literature suggests task/skill similarity is often the stronger signal; the time-series-RAG literature assumes shape similarity is primary. Which holds for time series *reasoning* tasks — where the question defines what reasoning skill is needed, and the series shape determines whether that skill is testable — is unknown and is one of the core things Experiment 1 (Section 4) is designed to answer.


---

## 5. First Experiment

**Goal**: establish whether series-shape similarity or question/skill similarity (or some combination) better predicts which demonstrations help, before any learned/trained retrieval component is introduced.

**Dataset**: TimeSeriesExam, repurposed into pool/test splits since it was designed for evaluation only.

**Split**: a cross-template split — hold out a fraction of the 104 templates entirely for test, stratified by category, so that retrieval cannot trivially succeed by finding another instance of the same template. Given the modest sample size (763 questions total), use multiple random splits and report mean ± std rather than a single split.

**Conditions to compare** (at each demonstration budget k ∈ {0,1,2,3,5,8}):
1. Zero-shot (k=0)
2. Random-k
3. TS-only top-k (via the TS index)
4. Text-only top-k (via the text index)
5. Weighted score-fusion of both (sweep over fusion weight α)
6. Same-category oracle (upper-bound reference, using ground-truth category metadata)

**Metric**: MCQ accuracy per condition per k, averaged across splits.

**What this first experiment tells us**: whether series-shape or question/skill similarity matters more for this task, whether they're complementary (fusion beats both alone) or redundant, and what a naive similarity-based budget–accuracy curve looks like — establishing the baseline that any later learned/utility-aware retrieval method would need to beat.
