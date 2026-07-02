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
| `chatts_model.py` | `ChatTSHFWrapper` | `bytedance-research/ChatTS-8B`, `-14B` | HF | `input_text`, `input_ts` |
| `vllm_chatts_model.py` | `ChatTSVLLMWrapper` | `bytedance-research/ChatTS-8B-vllm`, `-14B-vllm` | vLLM | `input_text`, `input_ts` |
| `time_omni_model.py` | `TimeOmniHFWrapper` | `anton-hugging/TimeOmni-1-7B` | HF (Qwen2.5-7B base) | `input_text` |
| `api_model.py` | `APIModelWrapper` | `openai`, `openai_o1`, `anthropic`, `gemini`, `deepseek_v3`, `ollama` | REST API | `input_text` |
| `baselines.py` | `RandomBaseline` | `random_baseline` | none | `input_text` (parses options) |
| `baselines.py` | `KNNBaseline` | `knn_baseline` | DTW | `input_text`, `input_ts` |
| `baselines.py` | `DinoKNNCLSABaseline` | `dino_knn_clsa_baseline` | DINOv2-Large | `input_text`, `input_ts` |
| `baselines.py` | `ZeroedTSBaseline` | `zeroed_ts_baseline` | HF (Qwen3-4B) | `input_text` (zeroes example TS) |
| `baselines.py` | `EmptyTSBaseline` | `empty_ts_baseline` | HF (Qwen3-4B) | `input_text` (empties example TS) |
| `baselines.py` | `EmptyAllTSBaseline` | `empty_all_ts_baseline` | HF (Qwen3-4B) | `input_text` (empties all TS) |
| `baselines.py` | `EmptyAllTSChatTSBaseline` | `empty_all_ts_chatts_baseline` | HF ChatTS | `input_ts` (zeroed) |

---

## Batch Contract

All batch fields are always present. What each model reads depends on its `input_mode`.

```
input_text   str   combined mode: full prompt with TS as "[v1, v2, ...]" numeric text
                   separate mode: prompt with <ts><ts/> placeholders for each series
input_ts     list  [demo_ts_0, ..., demo_ts_k-1, query_ts]  (raw float arrays)
output_text  str   Gold label
task_id      str   e.g. "retrieval"
options      list  Valid option letters for this sample
```

The `input_mode` field in each model's `get_args_dict()` tells the eval loop how to deliver TS:

**`input_mode = "combined"`** (`InstructModel`, `LargeInstructModel`, `TimeOmniHFWrapper`, `APIModelWrapper`, text baselines):
`input_text` already contains TS as numeric arrays; `input_ts` is ignored.

**`input_mode = "separate"`** (`ImageInstructModel`, `QwenVLImageModel`, `ChatTSHFWrapper`, `ChatTSVLLMWrapper`, `DinoKNNCLSABaseline`):
`input_text` contains `<ts><ts/>` placeholders; raw arrays are in `input_ts`.
- Image models replace each placeholder with a matplotlib plot.
- ChatTS models pass `input_ts` arrays to the patch-embedding processor.
- `DinoKNNCLSABaseline` reads `input_ts` directly (ignores `input_text` for TS).

`KNNBaseline` declares `input_mode = "combined"` but also reads `input_ts` for DTW distance — both fields must be valid.

---

## Special Behaviours

- **`LargeInstructModel`**: prefers vLLM, falls back to HF `device_map="auto"` if vLLM is absent.
- **`ImageInstructModel`**: strips `-image-ts` suffix from method ID to get the real checkpoint path.
- **`ChatTSVLLMWrapper`**: sets `VLLM_USE_V1=0` to avoid a weight-tying check in vLLM V1.
- **`EmptyAllTSBaseline`**: needs `--quantization 8bit` to fit Qwen3-4B on a single RTX 4090.
- **`DinoKNNCLSABaseline`**: normalises TS to [0, 1] using support-set min/max (CLSA protocol) before embedding.
- All LLM wrappers strip `<think>...</think>` from Qwen3/3.6 reasoning-model outputs.

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

### `ChatTSVLLMWrapper` broken on vLLM 0.19.1 — use `ChatTSHFWrapper` instead
vLLM 0.19.1 (`multits_large` env) removed the `VLLM_USE_V1` env var. The workaround in
`vllm_chatts_model.py` that sets `os.environ["VLLM_USE_V1"] = "0"` is therefore silently
ignored, V1 runs, and its `TransformersForCausalLM` weight loader crashes because ChatTS
weights are named `model.model.layers.*` but V1 looks for `layers.*` at the top level.
Use `bytedance-research/ChatTS-8B` (`ChatTSHFWrapper`, `run_single_gpu.sh`) instead.

### `get_args_dict()` must not redeclare global CLI args
`base_model.get_relevant_args()` raises `ValueError` if a key in `get_args_dict()` already
exists in the global argparse namespace (defined in `utils/args.py`). The global args include
`device`, `method`, `cache_dir`, `quantization`, `batch_size`, and others — do not repeat
them in any model's `get_args_dict()`. Only add model-specific keys such as `model_type`,
`input_mode`, `max_seq_length`, `max_new_tokens`, `format`.
