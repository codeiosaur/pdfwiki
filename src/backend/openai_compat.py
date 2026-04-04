"""
OpenAI-compatible LLM backend.

Covers any service that speaks the OpenAI chat completions protocol:
Ollama, OpenRouter, Runpod, LM Studio, vLLM, Together AI, etc.

Includes:
- Automatic retry with exponential backoff for rate-limited (429) requests
- OpenRouter-specific features (structured outputs, model fallbacks,
  response healing) activated automatically when the base URL is OpenRouter
"""

import random
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

# Ollama default port — used to enable structured output enforcement
_OLLAMA_DEFAULT_PORT = "11434"


def _is_openrouter(base_url: str) -> bool:
    """Check if the base URL points to OpenRouter."""
    return any(host in base_url.lower() for host in _OPENROUTER_HOSTS)


def _is_ollama(base_url: str) -> bool:
    """Check if the base URL points to a local Ollama instance (default port)."""
    return _OLLAMA_DEFAULT_PORT in base_url


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
        self._is_ollama = _is_ollama(config.base_url)
        self._fallback_models: list[str] = []
        # Local endpoints (Ollama, LM Studio) rarely hit 429s and recover quickly.
        # Use a shorter initial backoff for them; keep the longer default for OpenRouter.
        self._initial_backoff = _INITIAL_BACKOFF_SECONDS if self._is_openrouter else 1.0

        # Observability counters — cumulative since this instance was created.
        self._total_requests: int = 0
        self._retry_count: int = 0
        self._fallback_hops: int = 0

    @property
    def is_openrouter(self) -> bool:
        """True when this backend is pointed at OpenRouter."""
        return self._is_openrouter

    def set_fallback_models(self, models: list[str]) -> None:
        """Set fallback model IDs for OpenRouter's model fallback feature."""
        self._fallback_models = list(models)

    def metrics(self) -> dict:
        """Return cumulative observability counters for this backend instance."""
        return {
            "total_requests": self._total_requests,
            "retry_count": self._retry_count,
            "fallback_hops": self._fallback_hops,
        }

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        json_schema: Optional[dict] = None,
        context: str = "",
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

            # Zero Data Retention is a request-level OpenRouter parameter.
            if self._config.openrouter_zdr:
                extra_body["zdr"] = True

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._config.temperature,
            "max_tokens": tokens,
        }

        # Structured output support
        if json_schema is not None and (self._is_openrouter or self._is_ollama):
            # OpenRouter and Ollama (0.5+) both enforce the schema server-side
            # via response_format json_schema, which constrains token generation
            # to match the array structure.  This prevents fence-wrapped output
            # and object-instead-of-array failures common in smaller models.
            # Other providers fall through: prompts contain format instructions
            # and _parse_json_array handles any preamble or fences.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("name", "response"),
                    "strict": True,
                    "schema": json_schema.get("schema", json_schema),
                },
            }

        # Client-side model rotation: try the primary model, then each fallback in order.
        # Each model gets its own retry budget with exponential backoff + jitter.
        # Server-side OpenRouter fallbacks (extra_body["models"]) are NOT used —
        # we rotate explicitly so failures are visible and identified per-model.
        models_to_try = [self._config.model]
        if self._is_openrouter and self._fallback_models:
            models_to_try += self._fallback_models

        # extra_body without the "models" key (we rotate client-side instead)
        base_extra_body = {k: v for k, v in extra_body.items() if k != "models"}

        tag = f"{self.label}" + (f" | {context}" if context else "")

        last_exc: Optional[Exception] = None
        self._total_requests += 1

        for model_idx, model in enumerate(models_to_try):
            kwargs["model"] = model
            if base_extra_body:
                kwargs["extra_body"] = base_extra_body
            else:
                kwargs.pop("extra_body", None)

            backoff = self._initial_backoff

            for attempt in range(_MAX_RETRIES + 1):
                try:
                    response = self._client.chat.completions.create(**kwargs)

                    message = response.choices[0].message if response.choices else None
                    content = message.content if message else None

                    # Some OpenRouter models satisfy json_schema constraints via a
                    # tool-call under the hood, leaving content=None and putting the
                    # JSON in tool_calls[0].function.arguments.
                    if not content and message is not None:
                        tool_calls = getattr(message, "tool_calls", None)
                        if tool_calls:
                            content = tool_calls[0].function.arguments

                    if not content:
                        # Build a diagnostic hint to help identify why the response was empty.
                        # Reasoning models (e.g. Nemotron, DeepSeek-R1) spend tokens on
                        # chain-of-thought that is counted in usage but not returned as content.
                        hints: list[str] = []
                        if message is not None:
                            if getattr(message, "refusal", None):
                                hints.append("model refused")
                            reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
                            if reasoning:
                                hints.append("model produced reasoning tokens but no output — try a non-reasoning model")
                        usage = getattr(response, "usage", None)
                        if usage:
                            completion_tokens = getattr(usage, "completion_tokens", None)
                            if completion_tokens:
                                hints.append(f"{completion_tokens} completion tokens counted but content empty")
                        hint_str = f" ({'; '.join(hints)})" if hints else ""
                        raise LLMBackendError(
                            f"[{self.label}] Empty response from {model}{hint_str}"
                        )
                    return content

                except Exception as exc:
                    last_exc = exc

                    is_rate_limit = (
                        "429" in str(exc)
                        or "rate limit" in str(exc).lower()
                        or "rate_limit" in str(exc).lower()
                    )
                    is_empty_response = "Empty response" in str(exc)

                    if not (is_rate_limit or is_empty_response) or attempt >= _MAX_RETRIES:
                        break  # Stop retrying this model, try the next one

                    wait_time = min(backoff, _MAX_BACKOFF_SECONDS) + random.uniform(0, 2.0)
                    reason = "Rate limited" if is_rate_limit else "Empty response"
                    print(f"  [{tag}] {reason} on {model} (attempt {attempt + 1}/{_MAX_RETRIES + 1}), "
                          f"retrying in {round(wait_time)}s...")
                    self._retry_count += 1
                    time.sleep(wait_time)
                    backoff *= _BACKOFF_MULTIPLIER

            # This model exhausted its retries — move to the next
            if model_idx < len(models_to_try) - 1:
                next_model = models_to_try[model_idx + 1]
                self._fallback_hops += 1
                print(f"  [{tag}] {model} failed, trying {next_model}...")

        tried = ", ".join(models_to_try)
        raise LLMBackendError(
            f"[{self.label}] All models failed ({tried}). Last error: {last_exc}"
        )