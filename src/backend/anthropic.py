"""
Anthropic API backend.

Uses the Anthropic Python SDK for the Claude model family.
"""

from typing import Optional

from backend.base import BackendConfig, LLMBackend, LLMBackendError

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]


class AnthropicBackend(LLMBackend):
    """
    Backend for the Anthropic Messages API.

    Requires the 'anthropic' Python package and a valid API key
    set via the ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)

        if anthropic is None:
            raise LLMBackendError(
                "The 'anthropic' Python package is required for the Anthropic "
                "backend.  Install it with:  pip install anthropic"
            )

        if not config.api_key:
            raise LLMBackendError(
                "Anthropic backend requires an API key.  "
                "Set the ANTHROPIC_API_KEY environment variable."
            )

        self._client = anthropic.Anthropic(api_key=config.api_key)

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        tokens = max_tokens if max_tokens is not None else self._config.max_tokens

        try:
            response = self._client.messages.create(
                model=self._config.model,
                max_tokens=tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise LLMBackendError(
                f"[{self.label}] Anthropic API request failed: {exc}"
            ) from exc

        # Extract text from the response content blocks.
        text_parts = [
            block.text
            for block in response.content
            if hasattr(block, "text")
        ]

        if not text_parts:
            raise LLMBackendError(
                f"[{self.label}] Empty response from {self._config.model}"
            )

        return "\n".join(text_parts)