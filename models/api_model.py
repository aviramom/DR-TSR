import os
from typing import Any, Dict, List, Optional
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

try:
    import requests
except Exception:
    requests = None

# Correct base wrapper import
from models.base_model import BaseModelWrapper

# Optional SDK imports (handled gracefully if missing)
try:
    # Newer Google SDK (preferred): `from google import genai`
    from google import genai as google_genai
except Exception:
    google_genai = None

try:
    # Older Google SDK: `import google.generativeai as genai`
    import google.generativeai as google_generativeai
except Exception:
    google_generativeai = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import anthropic
except Exception:
    anthropic = None

# API keys (read both conventional and project-specific names when relevant)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)
# Support both GOOGLE_API_KEY and GOOGLE_GEMINI_API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_GEMINI_API", None))
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", None)
OLLAMA_API = os.environ.get("OLLAMA_API", None)


# (OpenRouter support removed) Optional OpenRouter key is no longer used


class APIModelWrapper(BaseModelWrapper):
    """
    Wraps multiple hosted LLM APIs behind the BaseModelWrapper interface.

    Supported methods (set via args.method):
      - "openai": OpenAI Chat Completions (e.g., gpt-4)
      - "openai_o1": OpenAI o1 model (chat.completions)
      - "anthropic": Anthropic Claude (messages API)
      - "gemini": Google Gemini (google-genai client preferred; falls back to google-generativeai)
      - "deepseek_v3": DeepSeek V3 via OpenAI-compatible API and base_url
      - "deepseek_r1": DeepSeek R1 via OpenAI-compatible API and base_url
            - "ollama": Ollama-compatible /api/chat endpoint via requests
      - "llama": Local/hosted LLaMA via a provided transformers pipeline (pass pipeline in generate())

    Batch item contract (assumption):
      Each dict in `batch` must include a "content" key with the user prompt string.
      For method=="llama", pass a `pipeline` callable via generate(pipeline=...).
    """

    def __init__(self, args, **kwargs) -> None:
        self.args = args
        self.client: Any = None
        self.method: str = getattr(args, "method", "openai")
        # For DeepSeek (SiliconFlow OpenAI-compatible endpoint)
        self.base_url: Optional[str] = getattr(args, "base_url", "https://api.siliconflow.com/v1")

        # Common model identifier used across wrappers
        self.llm_id: str = str(getattr(args, "llm_id", ""))

        # Ollama native endpoint (NOT OpenAI-compatible /v1). Allow override via args/env.
        self.ollama_url: str = str(
            getattr(args, "ollama_url", None)
            or OLLAMA_API
            or "https://cis-ollama/api/chat"
        )

        self.model = self._default_model(self.method)
        self.temperature: float = float(getattr(args, "temperature", 0.7))
        self.max_new_tokens: int = int(getattr(args, "max_new_tokens", 512))
        self.timeout: int = int(getattr(args, "timeout", 30))
        self.load_model()

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        return {
           # "temperature": 0.7,
            "max_new_tokens": 512,
            "timeout": 30,
            "base_url": "https://api.siliconflow.com/v1",  # used for DeepSeek (SiliconFlow),
            "model_type": "api",
            "input_mode": "combined",
            "llm_id": "qwen3:235b",
            "ollama_url": "https://cis-ollama/api/chat",
            "think": True,
        }

    @staticmethod
    def _default_model(method: str) -> str:
        defaults = {
            "openai": "gpt-4",
            "openai_o1": "o1",
            "anthropic": "claude-3-5-sonnet-20241022",
            "gemini": "gemini-2.0-flash",
            "deepseek_v3": "deepseek-ai/DeepSeek-V3",
            "deepseek_r1": "deepseek-ai/DeepSeek-R1",
            "ollama": "ollama",
            "llama": "llama",

        }
        return defaults[method]



    def load_model(self, *args, **kwargs):
        method = self.method

        if method in ("openai", "openai_o1"):
            if OpenAI is None:
                raise ImportError("openai package not installed. Install with: pip install openai")
            if not OPENAI_API_KEY:
                raise EnvironmentError("OPENAI_API_KEY is not set.")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            return self.client

        if method == "anthropic":
            if anthropic is None:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
            if not ANTHROPIC_API_KEY:
                raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            return self.client

        if method == "gemini":
            if GOOGLE_API_KEY is None:
                raise EnvironmentError("GOOGLE_API_KEY (or GOOGLE_GEMINI_API) is not set.")
            # Prefer google-genai client API when available
            if google_genai is not None:
                self.client = google_genai.Client(api_key=GOOGLE_API_KEY)
                return self.client
            # Fallback to older google-generativeai SDK
            if google_generativeai is not None:
                google_generativeai.configure(api_key=GOOGLE_API_KEY)
                # Model constructed per-call in generate
                self.client = google_generativeai
                return self.client
            raise ImportError(
                "Neither google-genai nor google-generativeai is installed. Install one of: \n"
                "  pip install google-genai  # preferred\n  or\n  pip install google-generativeai"
            )

        if method in ("deepseek_v3", "deepseek_r1"):
            if OpenAI is None:
                raise ImportError("openai package not installed. Install with: pip install openai")
            # Use SiliconFlow OpenAI-compatible endpoint for DeepSeek models
            if not SILICONFLOW_API_KEY:
                raise EnvironmentError("SILICONFLOW_API_KEY is not set.")
            self.client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=self.base_url)
            if not getattr(self, "model", None):
                self.model = "deepseek-ai/DeepSeek-R1" if method == "deepseek_r1" else "deepseek-ai/DeepSeek-V3"
            return self.client

        if method == "ollama":
            if requests is None:
                raise ImportError("requests package not installed. Install with: pip install requests")
            # Use requests.Session so connections are reused across calls.
            session = requests.Session()
            # Match the user's working setup: ignore TLS verification.
            session.verify = False
            self.client = session

            # For Ollama, actual model name comes from llm_id.
            if self.llm_id:
                if self.llm_id in ["llama3.2:latest", "meta-llama/Llama-3.2-1B"]:
                    self.model = "llama3.2:latest"
                elif self.llm_id in ["qwen3:235b","Qwen/Qwen3-235B-A22B"]:
                    self.model = "qwen3:235b"
                else:
                    # model not supported  eror
                    raise ValueError(f"Unsupported llm_id for Ollama: {self.llm_id}. Supported: 'llama3.2:latest', 'qwen3:235b'")

                self.model = self.llm_id
            return self.client

        raise ValueError(f"Unsupported method: {method}")

    def generate(self, batch: List[Dict[str, Any]], max_new_tokens: int = 50, **generate_kwargs) -> List[str]:

        outputs: List[str] = []
        # Preserve existing behavior across the codebase:
        # - default to args.max_new_tokens (self.max_new_tokens)
        # - allow explicit overrides via generate_kwargs or an explicitly-passed max_new_tokens
        if "max_new_tokens" in generate_kwargs:
            max_tokens = int(generate_kwargs["max_new_tokens"])
        elif max_new_tokens != 50:
            max_tokens = int(max_new_tokens)
        else:
            max_tokens = int(self.max_new_tokens)
        model = self.model
        temperature = self.temperature

        if self.client is None:
            raise ValueError("Client not initialized. Call load_model() before generate().")

        timeout = int(generate_kwargs.get("timeout", self.timeout))

        # Accept either the common dict batch {"input_text": [...]} or a raw list[str]
        if isinstance(batch, dict):
            batch_content = batch.get("input_text") or batch.get("full_text") or []
        else:
            batch_content = batch

        for content in batch_content:

            if content is None:
                raise ValueError("Each prompt must be a non-empty string.")

            if self.method == "openai":
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens,
                    temperature= temperature,
                    timeout=timeout,
                )
                outputs.append(getattr(resp.choices[0].message, "content", str(resp.choices[0].message)))

            elif self.method == "openai_o1":
                resp = self.client.chat.completions.create(
                    model=model,  # typically "o1"
                    messages=[{"role": "user", "content": content}],
                    timeout=timeout,
                )
                outputs.append(getattr(resp.choices[0].message, "content", str(resp.choices[0].message)))

            elif self.method == "anthropic":
                message = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens if max_tokens is not None else 4096,
                    messages=[{"role": "user", "content": content}],
                )
                # message.content is a list of content blocks
                try:
                    outputs.append(message.content[0].text)
                except Exception:
                    # Fallback to string repr
                    outputs.append(str(message))

            elif self.method == "gemini":
                # Prefer google-genai Client
                if google_genai is not None and isinstance(self.client, google_genai.Client):
                    response = self.client.models.generate_content(
                        model=model,
                        contents=content,
                    )
                    text = self._extract_gemini_text(response)
                    outputs.append(text)
                else:
                    # Fallback to google-generativeai
                    model_obj = self.client.GenerativeModel(model)
                    response = model_obj.generate_content(content)
                    text = getattr(response, "text", None)
                    if not text:
                        text = self._extract_gemini_text(response)
                    outputs.append(text)

            elif self.method == "deepseek_v3":
                resp = self.client.chat.completions.create(
                    model=model,  # deepseek-ai/DeepSeek-V3
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens if max_tokens is not None else 4096,
                    stream=False,
                )
                outputs.append(resp.choices[0].message.content)

            elif self.method == "deepseek_r1":
                resp = self.client.chat.completions.create(
                    model=model,  # deepseek-ai/DeepSeek-R1
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens if max_tokens is not None else 4096,
                    stream=False,
                )
                outputs.append(resp.choices[0].message.content)

            elif self.method == "ollama":
                if requests is None:
                    raise ImportError("requests package not installed. Install with: pip install requests")
                # Ollama native chat endpoint: POST {host}/api/chat
                # Use llm_id as the actual server-side model identifier.
                ollama_model = self.llm_id or model

                think = bool(generate_kwargs.get("think", getattr(self.args, "think", True)))


                payload = {
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": content}],
                    "stream": False,
                    "options": {
                        "think": think,
                        "num_predict": self.args.max_new_tokens
                    },
                }

                resp = self.client.post(self.ollama_url, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                # Ollama /api/chat returns: {"message": {"content": "..."}, ...}
                msg = data.get("message") if isinstance(data, dict) else None

                if isinstance(msg, dict) and "content" in msg:
                    outputs.append(str(msg.get("content") or ""))
                elif isinstance(data, dict) and isinstance(data.get("response"), str):
                    outputs.append(data["response"])
                else:
                    outputs.append(str(data))

            else:
                raise ValueError(f"Unsupported method: {self.method}")

        return outputs

    @staticmethod
    def _extract_gemini_text(response: Any) -> str:
        """Best-effort extraction of text from different Gemini SDK response shapes."""
        # google-genai may provide .text or .output_text
        for attr in ("text", "output_text"):
            val = getattr(response, attr, None)
            if isinstance(val, str) and val:
                return val
        # Older candidates structure
        try:
            return response.candidates[0].content.parts[0].text
        except Exception:
            pass
        # Fallback to string repr
        return str(response)