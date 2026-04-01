"""
OpenAI-compatible LLM backend.

Covers any service that speaks the OpenAI chat completions protocol:
Ollama, OpenRouter, Runpod, LM Studio, vLLM, Together AI, etc.

Includes:
- Automatic retry with exponential backoff for rate-limited (429) requests
- OpenRouter-specific features (structured outputs, model fallbacks,
  response healing) activated automatically when the base URL is OpenRouter
"""

import time
from typing import Any, Optional

from backend.base import BackendConfig, LLMBackend, LLMBackendError

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]


# Retry settings for rate-limited requests
_MAX_RETRIES = 3
_INITIAL_BACKOFF_SECONDS = 5.0
_BACKOFF_MULTIPLIER = 2.0
_MAX_BACKOFF_SECONDS = 60.0

# OpenRouter detection
_OPENROUTER_HOSTS = {"openrouter.ai"}


def _is_openrouter(base_url: str) -> bool:
    """Check if the base URL points to OpenRouter."""
    return any(host in base_url.lower() for host in _OPENROUTER_HOSTS)


class OpenAICompatBackend(LLMBackend):
    """
    Backend for any OpenAI-compatible API endpoint.

    The openai Python package is used as the HTTP client.  If it's not
    installed, instantiation raises a clear error.

    Rate-limited (429) responses are automatically retried with
    exponential backoff up to _MAX_RETRIES times.

    When pointed at OpenRouter, the backend automatically:
    - Enables the response-healing plugin (fixes malformed JSON)
    - Uses fallback models if configured via PASS2_FALLBACK_MODELS
    - Supports structured output schemas via the json_schema parameter
    """

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)

        if openai is None:
            raise LLMBackendError(
                "The 'openai' Python package is required for OpenAI-compatible "
                "backends.  Install it with:  pip install openai"
            )

        api_key = config.api_key if config.api_key else "not-needed"

        self._client = openai.OpenAI(
            base_url=config.base_url,
            api_key=api_key,
        )
        self._is_openrouter = _is_openrouter(config.base_url)
        self._fallback_models: list[str] = []
        # Local endpoints (Ollama, LM Studio) rarely hit 429s and recover quickly.
        # Use a shorter initial backoff for them; keep the longer default for OpenRouter.
        self._initial_backoff = _INITIAL_BACKOFF_SECONDS if self._is_openrouter else 1.0

    @property
    def is_openrouter(self) -> bool:
        """True when this backend is pointed at OpenRouter."""
        return self._is_openrouter

    def set_fallback_models(self, models: list[str]) -> None:
        """Set fallback model IDs for OpenRouter's model fallback feature."""
        self._fallback_models = list(models)

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        json_schema: Optional[dict] = None,
    ) -> str:
        """
        Send a prompt and return the model's text response.

        Args:
            prompt:      The user prompt string.
            max_tokens:  Override the default max_tokens if provided.
            json_schema: Optional JSON schema to enforce structured output.
                         When provided on OpenRouter, uses their structured
                         outputs feature.  On other providers, the schema
                         is included in the prompt as guidance (best-effort).

        Returns:
            The model's response as a plain string.
        """
        tokens = max_tokens if max_tokens is not None else self._config.max_tokens

        # Build extra_body for OpenRouter-specific features
        extra_body: dict[str, Any] = {}

        if self._is_openrouter:
            # Model fallbacks
            if self._fallback_models:
                extra_body["models"] = self._fallback_models

            # Response healing plugin (fixes malformed JSON)
            extra_body["plugins"] = [{"id": "response-healing"}]

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._config.temperature,
            "max_tokens": tokens,
        }

        # Structured output support
        if json_schema is not None and self._is_openrouter:
            # OpenRouter enforces the schema server-side and handles
            # response healing.  For non-OpenRouter providers (Ollama,
            # LM Studio, etc.), we skip schema injection entirely —
            # the prompts already contain JSON format instructions and
            # the _parse_json_array fallback parser handles any preamble.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("name", "response"),
                    "strict": True,
                    "schema": json_schema.get("schema", json_schema),
                },
            }

        if extra_body:
            kwargs["extra_body"] = extra_body

        # Retry loop with backoff for rate limits
        last_exc: Optional[Exception] = None
        backoff = self._initial_backoff

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)

                content = response.choices[0].message.content if response.choices else None
                if content is None:
                    raise LLMBackendError(
                        f"[{self.label}] Empty response from {self._config.model}"
                    )
                return content

            except Exception as exc:
                last_exc = exc

                is_rate_limit = (
                    "429" in str(exc)
                    or "rate limit" in str(exc).lower()
                    or "rate_limit" in str(exc).lower()
                )

                if not is_rate_limit or attempt >= _MAX_RETRIES:
                    raise LLMBackendError(
                        f"[{self.label}] Request to {self._config.base_url} failed: {exc}"
                    ) from exc

                wait_time = min(backoff, _MAX_BACKOFF_SECONDS)
                print(f"  [{self.label}] Rate limited (attempt {attempt + 1}/{_MAX_RETRIES + 1}), "
                      f"retrying in {wait_time:.0f}s...")
                time.sleep(wait_time)
                backoff *= _BACKOFF_MULTIPLIER

        raise LLMBackendError(
            f"[{self.label}] Request failed after {_MAX_RETRIES + 1} attempts: {last_exc}"
        )