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

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        json_schema: Optional[dict] = None,
        context: str = "",
    ) -> str:
        tokens = max_tokens if max_tokens is not None else self._config.max_tokens

        # If a JSON schema is provided, add it as guidance in the prompt
        actual_prompt = prompt
        if json_schema is not None:
            import json as _json
            schema_hint = _json.dumps(json_schema.get("schema", json_schema), indent=2)
            actual_prompt = f"{prompt}\n\nRespond with ONLY valid JSON matching this schema:\n{schema_hint}"

        try:
            response = self._client.messages.create(
                model=self._config.model,
                max_tokens=tokens,
                messages=[{"role": "user", "content": actual_prompt}],
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