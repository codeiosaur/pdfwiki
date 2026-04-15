"""
Google Gemini / Gemma native backend.

Uses the google-genai SDK for models that are not reachable via the
OpenAI-compat endpoint (e.g. Gemma 4 family).

Flash / Flash-Lite models continue to use openai_compat — this backend
is only needed when the OpenAI-compat path returns 404 or malformed JSON.
"""

import threading
import time
import random
import re
from typing import Optional

from backend.base import BackendConfig, LLMBackend, LLMBackendError, RetryableError

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

# Hard timeout per request via SDK-level HttpOptions — avoids zombie threads
# from the thread-wrapper approach.
_REQUEST_TIMEOUT_SECONDS = 150


def _extract_retry_delay(error_str: str) -> Optional[float]:
    """
    Extract retryDelay from Gemini error response.
    Example: 'retryDelay': '2.786362344s' -> returns 2.786362344
    """
    try:
        # Look for retryDelay field with quoted numeric value + 's'
        match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?([0-9.]+)s?['\"]?", error_str)
        if match:
            return float(match.group(1))
    except (ValueError, AttributeError):
        pass
    return None


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

        self._api_key = config.api_key
        self._retry_count = 0
        self._fallback_hops = 0
        # Build list of models to try: primary + fallbacks
        self._models_to_try: list[str] = [self._config.model]
        if hasattr(self._config, 'fallback_models') and self._config.fallback_models:
            self._models_to_try.extend(self._config.fallback_models)

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

        generate_config = genai_types.GenerateContentConfig(**config_kwargs)

        # Try each model in the fallback chain
        for model_idx, model in enumerate(self._models_to_try):
            backoff = _INITIAL_BACKOFF_SECONDS
            last_exc: Optional[Exception] = None

            for attempt in range(_MAX_RETRIES):
                # Only set system_instruction for Gemini models, not Gemma models
                model_config_kwargs = config_kwargs.copy()
                if system_prompt is not None and "gemini" in model.lower():
                    model_config_kwargs["system_instruction"] = system_prompt

                model_generate_config = genai_types.GenerateContentConfig(**model_config_kwargs)

                try:
                    _result: list = [None]
                    _error: list = [None]

                    def _api_call():
                        try:
                            client = genai.Client(api_key=self._api_key)
                            _result[0] = client.models.generate_content(
                                model=model,
                                contents=actual_prompt,
                                config=model_generate_config,
                            )
                        except Exception as _e:
                            _error[0] = _e

                    _t = threading.Thread(target=_api_call, daemon=True)
                    _t.start()
                    _t.join(timeout=_REQUEST_TIMEOUT_SECONDS)
                    if _t.is_alive():
                        raise LLMBackendError(
                            f"[{self.label}] Request timed out after {_REQUEST_TIMEOUT_SECONDS}s"
                        )
                    if _error[0] is not None:
                        raise _error[0]
                    response = _result[0]
                    if not response.candidates:
                        reason = getattr(response, "prompt_feedback", "no candidates")
                        raise LLMBackendError(
                            f"[{self.label}] Response blocked or empty from {model}: {reason}"
                        )
                    text = response.text
                    if not text or not text.strip():
                        raise LLMBackendError(
                            f"[{self.label}] Empty response from {model}"
                        )
                    return text.strip()

                except LLMBackendError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    exc_str = str(exc).lower()
                    is_rate_limit = "429" in str(exc) or "quota" in exc_str or "rate" in exc_str
                    is_timeout = "timeout" in exc_str or "deadline" in exc_str or "timed out" in exc_str
                    is_retryable = is_rate_limit or is_timeout or "500" in str(exc) or "503" in str(exc)

                    # Rate limit on this model: try next fallback model
                    if is_rate_limit and model_idx < len(self._models_to_try) - 1:
                        ctx = f" [{context}]" if context else ""
                        print(
                            f"  [{self.label}{ctx}] Rate limited on {model}, "
                            f"trying fallback model..."
                        )
                        self._fallback_hops += 1
                        break  # Break inner loop to try next model

                    # Not retryable or exhausted retries: fail or try next model
                    if not is_retryable or attempt + 1 >= _MAX_RETRIES:
                        ctx = f" [{context}]" if context else ""
                        # If there are more models to try, don't print error yet
                        if model_idx < len(self._models_to_try) - 1:
                            print(
                                f"  [{self.label}{ctx}] Error on {model} "
                                f"(attempt {attempt + 1}/{_MAX_RETRIES}), trying fallback model..."
                            )
                            self._fallback_hops += 1
                            break  # Break inner loop to try next model
                        else:
                            # Last model exhausted
                            print(f"  [{self.label}{ctx}] Error on {model}: {exc}")
                            break

                    self._retry_count += 1

                    # Try to extract server-provided retryDelay from rate limit errors
                    sleep_time = backoff
                    if is_rate_limit:
                        server_delay = _extract_retry_delay(str(exc))
                        if server_delay is not None:
                            sleep_time = server_delay + 0.5  # Small buffer after server's suggested delay
                        else:
                            jitter = random.uniform(0, backoff * 0.3)
                            sleep_time = min(backoff + jitter, _MAX_BACKOFF_SECONDS)
                    else:
                        # Timeout or 5xx: use exponential backoff
                        jitter = random.uniform(0, backoff * 0.3)
                        sleep_time = min(backoff + jitter, _MAX_BACKOFF_SECONDS)

                    ctx = f" [{context}]" if context else ""
                    label = "Rate limited" if is_rate_limit else "Timed out" if is_timeout else "Error"
                    print(
                        f"  [{self.label}{ctx}] {label} on {model} "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES}), retrying in {sleep_time:.1f}s..."
                    )
                    raise RetryableError(sleep_time, f"{label} on {model}")
                    backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF_SECONDS)

        # All models exhausted
        raise LLMBackendError(
            f"[{self.label}] All models failed after {len(self._models_to_try)} model(s) and {_MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    def metrics(self) -> dict:
        return {
            "retry_count": self._retry_count,
            "fallback_hops": self._fallback_hops,
        }
