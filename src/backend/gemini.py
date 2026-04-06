"""
Google Gemini / Gemma native backend.

Uses the google-genai SDK for models that are not reachable via the
OpenAI-compat endpoint (e.g. Gemma 4 family).

Flash / Flash-Lite models continue to use openai_compat — this backend
is only needed when the OpenAI-compat path returns 404 or malformed JSON.
"""

import time
import random
from typing import Optional

from backend.base import BackendConfig, LLMBackend, LLMBackendError

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

# Retry settings (mirrors openai_compat)
_MAX_RETRIES = 4
_INITIAL_BACKOFF_SECONDS = 5.0
_BACKOFF_MULTIPLIER = 2.0
_MAX_BACKOFF_SECONDS = 60.0

# Hard timeout per request — prevents a stalled Gemini call from holding a
# pool semaphore slot for minutes.  90s is generous for longest synthesis pages.
_REQUEST_TIMEOUT_SECONDS = 90


class GeminiBackend(LLMBackend):
    """
    Backend for Google Gemini and Gemma models via the google-genai SDK.

    Supports:
    - system_prompt via system_instruction
    - json_schema embedded as prompt guidance (Gemini does not yet support
      OpenAI-style response_format with arbitrary schemas on all models)
    - Automatic retry with exponential backoff on rate-limit (429) errors
    """

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)

        if genai is None:
            raise LLMBackendError(
                "The 'google-genai' Python package is required for the Gemini "
                "backend.  Install it with:  pip install google-genai"
            )

        if not config.api_key:
            raise LLMBackendError(
                "Gemini backend requires an API key.  "
                "Set the GEMINI_API_KEY environment variable."
            )

        self._client = genai.Client(api_key=config.api_key)
        self._retry_count = 0
        self._fallback_hops = 0

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        json_schema: Optional[dict] = None,
        context: str = "",
        system_prompt: Optional[str] = None,
    ) -> str:
        tokens = max_tokens if max_tokens is not None else self._config.max_tokens

        # Embed JSON schema guidance in the prompt (same approach as Anthropic backend)
        actual_prompt = prompt
        if json_schema is not None:
            import json as _json
            schema_hint = _json.dumps(json_schema.get("schema", json_schema), indent=2)
            actual_prompt = (
                f"{prompt}\n\nRespond with ONLY valid JSON matching this schema:\n{schema_hint}"
            )

        config_kwargs: dict = {
            "max_output_tokens": tokens,
            "temperature": self._config.temperature,
        }
        if system_prompt is not None:
            config_kwargs["system_instruction"] = system_prompt

        config_kwargs["http_options"] = genai_types.HttpOptions(timeout=_REQUEST_TIMEOUT_SECONDS)
        generate_config = genai_types.GenerateContentConfig(**config_kwargs)

        backoff = _INITIAL_BACKOFF_SECONDS
        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self._config.model,
                    contents=actual_prompt,
                    config=generate_config,
                )
                text = response.text
                if not text or not text.strip():
                    raise LLMBackendError(
                        f"[{self.label}] Empty response from {self._config.model}"
                    )
                return text.strip()

            except LLMBackendError:
                raise
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc).lower()
                is_rate_limit = "429" in str(exc) or "quota" in exc_str or "rate" in exc_str
                is_retryable = is_rate_limit or "500" in str(exc) or "503" in str(exc)

                if not is_retryable or attempt + 1 >= _MAX_RETRIES:
                    break

                self._retry_count += 1
                jitter = random.uniform(0, backoff * 0.3)
                sleep_time = min(backoff + jitter, _MAX_BACKOFF_SECONDS)
                ctx = f" [{context}]" if context else ""
                print(
                    f"  [{self.label}{ctx}] {'Rate limited' if is_rate_limit else 'Error'} "
                    f"on {self._config.model} (attempt {attempt + 1}/{_MAX_RETRIES}), "
                    f"retrying in {sleep_time:.0f}s..."
                )
                time.sleep(sleep_time)
                backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF_SECONDS)

        raise LLMBackendError(
            f"[{self.label}] Gemini request failed after {_MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    def metrics(self) -> dict:
        return {
            "retry_count": self._retry_count,
            "fallback_hops": self._fallback_hops,
        }
