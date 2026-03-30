"""
OpenAI-compatible LLM backend.

Covers any service that speaks the OpenAI chat completions protocol:
Ollama, OpenRouter, Runpod, LM Studio, vLLM, Together AI, etc.
"""

from typing import Optional

from backend.base import BackendConfig, LLMBackend, LLMBackendError

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]


class OpenAICompatBackend(LLMBackend):
    """
    Backend for any OpenAI-compatible API endpoint.

    The openai Python package is used as the HTTP client.  If it's not
    installed, instantiation raises a clear error.
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

        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._config.temperature,
                max_tokens=tokens,
            )
        except Exception as exc:
            raise LLMBackendError(
                f"[{self.label}] Request to {self._config.base_url} failed: {exc}"
            ) from exc

        content = response.choices[0].message.content if response.choices else None
        if content is None:
            raise LLMBackendError(
                f"[{self.label}] Empty response from {self._config.model}"
            )

        return content