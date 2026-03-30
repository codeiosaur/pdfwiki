"""
OpenAI-compatible LLM backend.

Covers any service that speaks the OpenAI chat completions protocol:
Ollama, OpenRouter, Runpod, LM Studio, vLLM, Together AI, etc.

Includes automatic retry with exponential backoff for rate-limited (429)
requests, which is essential for free-tier API endpoints.
"""

import time
from typing import Optional

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


class OpenAICompatBackend(LLMBackend):
    """
    Backend for any OpenAI-compatible API endpoint.

    The openai Python package is used as the HTTP client.  If it's not
    installed, instantiation raises a clear error.

    Rate-limited (429) responses are automatically retried with
    exponential backoff up to _MAX_RETRIES times.
    """

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)

        if openai is None:
            raise LLMBackendError(
                "The 'openai' Python package is required for OpenAI-compatible "
                "backends.  Install it with:  pip install openai"
            )

        # For local endpoints (Ollama, LM Studio) that don't need a key,
        # pass a dummy value — the openai client requires *something*.
        api_key = config.api_key if config.api_key else "not-needed"

        self._client = openai.OpenAI(
            base_url=config.base_url,
            api_key=api_key,
        )

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        tokens = max_tokens if max_tokens is not None else self._config.max_tokens

        last_exc: Optional[Exception] = None
        backoff = _INITIAL_BACKOFF_SECONDS

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._config.temperature,
                    max_tokens=tokens,
                )

                content = response.choices[0].message.content if response.choices else None
                if content is None:
                    raise LLMBackendError(
                        f"[{self.label}] Empty response from {self._config.model}"
                    )
                return content

            except Exception as exc:
                last_exc = exc

                # Check if this is a rate limit error (429)
                is_rate_limit = (
                    "429" in str(exc)
                    or "rate limit" in str(exc).lower()
                    or "rate_limit" in str(exc).lower()
                )

                if not is_rate_limit or attempt >= _MAX_RETRIES:
                    # Not a rate limit, or we've exhausted retries — give up
                    raise LLMBackendError(
                        f"[{self.label}] Request to {self._config.base_url} failed: {exc}"
                    ) from exc

                # Rate limited — wait and retry
                wait_time = min(backoff, _MAX_BACKOFF_SECONDS)
                print(f"  [{self.label}] Rate limited (attempt {attempt + 1}/{_MAX_RETRIES + 1}), "
                      f"retrying in {wait_time:.0f}s...")
                time.sleep(wait_time)
                backoff *= _BACKOFF_MULTIPLIER

        # Should never reach here, but just in case
        raise LLMBackendError(
            f"[{self.label}] Request failed after {_MAX_RETRIES + 1} attempts: {last_exc}"
        )