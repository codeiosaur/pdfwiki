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
import re
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


def _extract_retry_after(error_str: str) -> Optional[float]:
    """
    Extract Retry-After value from OpenAI-compatible error response.
    Handles multiple formats:
    - Numeric seconds: 'Retry-After: 5' -> 5.0
    - Duration string (Groq): 'Please try again in 15m39.8592s' -> 939.8592
    - ISO duration: '1m', '30s', etc.
    """
    try:
        # Format 1: Numeric Retry-After header/field (case-insensitive)
        match = re.search(r"Retry-After['\"]?\s*:\s*['\"]?([0-9.]+)", error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))

        # Format 2: Groq-style duration in message (e.g., "15m39.8592s")
        # Looks for patterns like "15m39.8592s" or "15m 39s" or "39.8592s"
        match = re.search(r"in\s+(\d+)m([\d.]+)s", error_str)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            return minutes * 60 + seconds

        # Format 3: Just seconds with 's' suffix (e.g., "39.8592s")
        match = re.search(r"in\s+([\d.]+)s\b", error_str)
        if match:
            return float(match.group(1))
    except (ValueError, AttributeError):
        pass
    return None


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
        self._structured_output = config.structured_output or self._is_openrouter or self._is_ollama
        self._json_mode = config.json_mode and not self._structured_output
        self._wrap_array_schema = config.wrap_array_schema and self._structured_output
        self._ollama_num_ctx: Optional[int] = config.ollama_num_ctx if self._is_ollama else None
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
        system_prompt: Optional[str] = None,
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

        # Build extra_body for provider-specific features
        extra_body: dict[str, Any] = {}

        if self._is_ollama and self._ollama_num_ctx is not None:
            extra_body["options"] = {"num_ctx": self._ollama_num_ctx}

        if self._is_openrouter:
            # Model fallbacks
            if self._fallback_models:
                extra_body["models"] = self._fallback_models

            # Response healing plugin (fixes malformed JSON)
            extra_body["plugins"] = [{"id": "response-healing"}]

            # Zero Data Retention is a request-level OpenRouter parameter.
            if self._config.openrouter_zdr:
                extra_body["zdr"] = True

        # Build messages — include system prompt if provided.
        # On OpenRouter, wrap the system message content in a block with
        # cache_control so providers that support prompt caching (Anthropic,
        # OpenAI, DeepSeek, Gemini 2.5) can serve repeated prefixes from cache.
        if system_prompt is not None:
            if self._is_openrouter:
                system_message: dict[str, Any] = {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            else:
                system_message = {"role": "system", "content": system_prompt}
            messages: list[dict[str, Any]] = [
                system_message,
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": tokens,
        }

        # Structured output support
        if json_schema is not None and self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        elif json_schema is not None and self._structured_output:
            # OpenRouter and Ollama (0.5+) both enforce the schema server-side
            # via response_format json_schema, which constrains token generation
            # to match the array structure.  This prevents fence-wrapped output
            # and object-instead-of-array failures common in smaller models.
            # Other providers fall through: prompts contain format instructions
            # and _parse_json_array handles any preamble or fences.
            schema = json_schema.get("schema", json_schema)
            if self._wrap_array_schema and schema.get("type") == "array":
                # Some providers (e.g. Cerebras) support json_schema on objects
                # but reject array roots.  Wrap the array in an object; the parser
                # unwraps it via the max-length list heuristic.
                schema = {
                    "type": "object",
                    "properties": {"items": schema},
                    "required": ["items"],
                    "additionalProperties": False,
                }
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("name", "response"),
                    "strict": True,
                    "schema": schema,
                },
            }

        # Client-side model rotation: try the primary model, then each fallback in order.
        # Primary model + fallback queue.
        # Primary model is ALWAYS preferred if not on cooldown or permanently failed.
        # Fallbacks are rotated (not retried individually) when primary is unavailable.
        from collections import deque

        primary_model = self._config.model
        fallback_models = deque(self._fallback_models if self._is_openrouter and self._fallback_models else [])

        primary_cooldown_until: Optional[float] = None
        permanently_failed: set[str] = set()
        last_exc: Optional[Exception] = None

        # extra_body without the "models" key (we rotate client-side instead)
        base_extra_body = {k: v for k, v in extra_body.items() if k != "models"}
        tag = f"{self.label}" + (f" | {context}" if context else "")
        self._total_requests += 1

        while True:
            # Determine which model to use (primary if available, else next fallback)
            if primary_model not in permanently_failed:
                if primary_cooldown_until is None or time.time() >= primary_cooldown_until:
                    model = primary_model
                    primary_cooldown_until = None
                elif fallback_models:
                    model = fallback_models[0]
                else:
                    # Primary on cooldown, no fallbacks left — wait for primary to reset
                    wait_until_reset = primary_cooldown_until - time.time()
                    print(f"  [{tag}] All fallbacks exhausted, {primary_model} resumes in {round(wait_until_reset)}s...")
                    time.sleep(wait_until_reset + 0.1)
                    continue
            elif fallback_models:
                model = fallback_models[0]
            else:
                # All models failed permanently, no fallbacks
                tried = primary_model + (f", {', '.join(self._fallback_models)}" if self._fallback_models else "")
                raise LLMBackendError(
                    f"[{self.label}] All models failed permanently ({tried}). Last error: {last_exc}"
                )

            kwargs["model"] = model
            if base_extra_body:
                kwargs["extra_body"] = base_extra_body
            else:
                kwargs.pop("extra_body", None)

            backoff = self._initial_backoff
            _schema_dropped = False

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
                    hints: list[str] = []
                    _reasoning_detected = False
                    if message is not None:
                        if getattr(message, "refusal", None):
                            hints.append("model refused")
                        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
                        if reasoning:
                            _reasoning_detected = True
                            hints.append("model produced reasoning tokens but no output — try a non-reasoning model")
                    usage = getattr(response, "usage", None)
                    if usage:
                        completion_tokens = getattr(usage, "completion_tokens", None)
                        if completion_tokens:
                            hints.append(f"{completion_tokens} completion tokens counted but content empty")
                    hint_str = f" ({'; '.join(hints)})" if hints else ""
                    exc = LLMBackendError(
                        f"[{self.label}] Empty response from {model}{hint_str}"
                    )
                    # Thinking model + json_schema: drop the schema constraint and retry
                    if _reasoning_detected and not _schema_dropped and "response_format" in kwargs:
                        kwargs.pop("response_format")
                        _schema_dropped = True
                        print(f"  [{tag}] Thinking model produced no structured output on {model}, "
                              f"retrying without schema constraint...")
                        self._retry_count += 1
                        continue
                    raise exc
                return content

            except Exception as exc:
                last_exc = exc

                is_rate_limit = (
                    "429" in str(exc)
                    or "rate limit" in str(exc).lower()
                    or "rate_limit" in str(exc).lower()
                )
                is_empty_response = "Empty response" in str(exc)
                is_permanent_error = (
                    "400" in str(exc)
                    or "401" in str(exc)
                    or "403" in str(exc)
                    or "model_not_found" in str(exc).lower()
                    or "not available" in str(exc).lower()
                )

                # Handle based on error type
                if is_rate_limit:
                    server_delay = _extract_retry_after(str(exc))
                    if server_delay is None:
                        server_delay = _INITIAL_BACKOFF_SECONDS
                    server_delay += 0.5  # Small buffer

                    if model == primary_model:
                        # Primary hit rate limit — put it on cooldown
                        primary_cooldown_until = time.time() + server_delay
                        print(f"  [{tag}] {model} rate limited, back online in {round(server_delay)}s...")
                        self._retry_count += 1
                        # Fall through to model selection logic
                        continue
                    else:
                        # Fallback hit rate limit — rotate to next fallback, don't wait
                        fallback_models.rotate(-1)
                        print(f"  [{tag}] {model} rate limited, trying next fallback...")
                        self._retry_count += 1
                        continue

                elif is_permanent_error:
                    # Don't retry permanent errors
                    permanently_failed.add(model)
                    if model == primary_model:
                        print(f"  [{tag}] {model} failed permanently (not retryable), trying fallbacks...")
                    else:
                        fallback_models.rotate(-1)
                        print(f"  [{tag}] {model} failed permanently, trying next fallback...")
                    # Continue to model selection logic
                    continue

                elif is_empty_response:
                    # Retry with backoff
                    wait_time = min(backoff, _MAX_BACKOFF_SECONDS) + random.uniform(0, 2.0)
                    print(f"  [{tag}] Empty response from {model}, retrying in {round(wait_time)}s...")
                    self._retry_count += 1
                    time.sleep(wait_time)
                    backoff *= _BACKOFF_MULTIPLIER
                    continue

                else:
                    # Other error: mark as permanent and move on
                    permanently_failed.add(model)
                    print(f"  [{tag}] {model} failed: {exc}")
                    continue