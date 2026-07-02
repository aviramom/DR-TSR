# models/

All model wrappers. Each class implements `BaseModelWrapper` and is registered
in `utils/model.py:method_wrapper_dict` under one or more `--method` IDs.

---

## Wrapper Overview

| File | Class | Method IDs | Backend | Batch fields used |
|------|-------|-----------|---------|-------------------|
| `instruct_model.py` | `InstructModel` | `Qwen/Qwen3-4B-Instruct-2507`, Llama/Mistral variants | HF pipeline | `input_text` |
| `instruct_model.py` | `LargeInstructModel` | `Qwen/Qwen3.6-27B`, `-FP8` | vLLM (falls back to HF) | `input_text` |
| `image_instruct_model.py` | `ImageInstructModel` | `Qwen/Qwen3.6-27B-image-ts` | vLLM multimodal | `input_text`, `input_ts` |
| `qwen_vl_image_model.py` | `QwenVLImageModel` | `Qwen/Qwen3-VL-8B-Instruct` | HF Vision2Seq | `input_text`, `input_ts` |
| `qwen_vl_thinking_model.py` | `QwenVLThinkingModel` | `Qwen/Qwen3-VL-8B-Thinking` | HF Vision2Seq + thinking | `input_text`, `input_ts` |
| `qwen_vl_thinking_vllm_model.py` | `QwenVLThinkingVLLMModel` | `Qwen/Qwen3-VL-8B-Thinking-vllm` | vLLM + thinking | `input_text`, `input_ts` |
| `chatts_model.py` | `ChatTSHFWrapper` | `bytedance-research/ChatTS-8B`, `-14B` | HF | `input_text`, `input_ts` |
| `vllm_chatts_model.py` | `ChatTSVLLMWrapper` | `bytedance-research/ChatTS-8B-vllm`, `-14B-vllm` | vLLM | `input_text`, `input_ts` |
| `api_model.py` | `APIModelWrapper` | `openai`, `anthropic`, `gemini`, `deepseek_v3`, `ollama` | REST API | `input_text` |
| `baselines.py` | `RandomBaseline` | `random_baseline` | none | `input_text` (parses options) |
| `baselines.py` | `KNNBaseline` | `knn_baseline` | DTW | `input_text`, `input_ts` |
| `baselines.py` | `DinoKNNCLSABaseline` | `dino_knn_clsa_baseline` | DINOv2-Large | `input_text`, `input_ts` |
| `baselines.py` | `ZeroedTSBaseline` | `zeroed_ts_baseline` | HF (Qwen3-4B) | `input_text` (zeroes example TS) |
| `baselines.py` | `EmptyTSBaseline` | `empty_ts_baseline` | HF (Qwen3-4B) | `input_text` (empties example TS) |
| `baselines.py` | `EmptyAllTSBaseline` | `empty_all_ts_baseline` | HF (Qwen3-4B) | `input_text` (empties all TS) |
| `baselines.py` | `EmptyAllTSChatTSBaseline` | `empty_all_ts_chatts_baseline` | HF ChatTS | `input_ts` (zeroed) |

---

## Batch Contract

`ICLUCRDataset` produces batches with these keys:

```
input_text   str   Full prompt with TS already embedded as "[v1, v2, ...]"
input_ts     list  [support_ts_0, ..., support_ts_k-1, query_ts]  (raw floats)
output_text  str   Gold label (int as string)
task_id      str   e.g. "icl_ucr_GunPoint"
options      list  Valid class labels parsed from prompt
mean/std     list  Per-series stats (metadata only, not used by models)
```

**Text models** (`InstructModel`, `LargeInstructModel`, `APIModelWrapper`, baselines):
consume `input_text` only. TS values are already in the text as numeric arrays.

**Image models** (`ImageInstructModel`, `QwenVLImageModel`):
split `input_text` on `<ts><ts/>` placeholders to know where to insert matplotlib
plots. They also consume `input_ts` for the raw values to plot.

**ChatTS models** (`ChatTSHFWrapper`, `ChatTSVLLMWrapper`):
strip the embedded numeric arrays from `input_text` (via `ts_parser`), rebuild
placeholders, then pass raw arrays from `input_ts` to the processor.

---

## Special Behaviours

- **`LargeInstructModel`**: prefers vLLM, falls back to HF `device_map="auto"` if vLLM is absent.
- **`ImageInstructModel`**: strips `-image-ts` suffix from method ID to get the real checkpoint path.
- **`ChatTSVLLMWrapper`**: sets `VLLM_USE_V1=0` to avoid a weight-tying check in vLLM V1.
- **`EmptyAllTSBaseline`**: needs `--quantization 8bit` to fit Qwen3-4B on a single RTX 4090.
- **`DinoKNNCLSABaseline`**: normalises TS to [0, 1] using support-set min/max (CLSA protocol) before embedding.
- All LLM wrappers strip `<think>...</think>` from Qwen3/3.6 reasoning-model outputs.
- **Thinking models** (`QwenVLThinkingModel`, `QwenVLThinkingVLLMModel`): use `skip_special_tokens=False` so `</think>` survives decoding, then split on it to return only the answer section. `thinking_budget=2048` (soft hint via `apply_chat_template`). **Do not reduce below 2048** — smaller budgets cause the model to close `</think>` prematurely and restart reasoning in the answer section, producing INVALID_PREDICTION.
- `-vllm` suffix on the method ID is stripped internally; the same `Qwen/Qwen3-VL-8B-Instruct` checkpoint is loaded for both `Qwen/Qwen3-VL-8B-Thinking` and `Qwen/Qwen3-VL-8B-Thinking-vllm`.

---

## Log-Probability Extraction (Future — Utility Scoring)

Later phases of the retrieval project require measuring how much a candidate demonstration
changes the model's probability of producing the correct answer. This needs a new method
on `BaseModelWrapper`:

```python
def get_log_prob_correct(self, prompt: str, correct_option: str) -> float:
    """Return the log-probability assigned to correct_option given prompt."""
```

HF and vLLM generation both support token-level log-probs (`output_scores` / `logprobs`).
For Experiment 1 (baseline study), models are used for inference only (`generate`) — no
log-prob extraction is needed yet.

---

## Known Issues

### `DinoKNNBaseline` (class in `baselines.py`) uses missing `ts_encoders` package
The class imports `DinoEncoder` from `models.ts_encoders.dino.dino_enc` which does
not exist. It has been **removed from `method_wrapper_dict`** — use
`dino_knn_clsa_baseline` instead, which has the full delay-embedding + DINOv2-Large
pipeline and no missing dependencies.

### `get_args_dict()` must not redeclare global CLI args
`base_model.get_relevant_args()` raises `ValueError` if a key in `get_args_dict()` already
exists in the global argparse namespace (defined in `utils/args.py`). The global args include
`device`, `method`, `cache_dir`, `quantization`, `batch_size`, and others — do not repeat
them in any model's `get_args_dict()`. Only add model-specific keys such as `model_type`,
`input_mode`, `max_seq_length`, `max_new_tokens`, `format`.
