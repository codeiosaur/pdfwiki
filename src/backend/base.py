"""
Abstract LLM backend interface.

All LLM providers implement this interface so the rest of the pipeline
can call backend.generate(prompt) without knowing what's behind it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class BackendConfig:
    """
    Configuration for an LLM backend.

    Attributes:
        provider:   One of 'openai_compat', 'anthropic'.
                    'openai_compat' covers Ollama, OpenRouter, Runpod,
                    LM Studio, vLLM, Together AI, and any other service
                    that speaks the OpenAI chat completions protocol.
        base_url:   The API base URL (e.g. http://localhost:11434/v1).
        model:      Model identifier (e.g. 'llama3.1:8b', 'claude-haiku-4-5-20251001').
        api_key:    Optional API key.  Loaded from environment variables —
                    NEVER hardcoded or committed to version control.
        temperature: Sampling temperature.  0 = deterministic.
        max_tokens:  Default max tokens for completions.
        label:      Human-readable name for logging (e.g. 'local', 'api').
    """
    provider: str
    base_url: str
    model: str
    api_key: Optional[str] = field(default=None, repr=False)  # repr=False prevents printing
    temperature: float = 0.0
    max_tokens: int = 800
    label: str = ""

    def __post_init__(self) -> None:
        if self.provider not in ("openai_compat", "anthropic"):
            raise ValueError(
                f"Unknown provider '{self.provider}'. "
                f"Supported: 'openai_compat', 'anthropic'."
            )


class LLMBackend(ABC):
    """Abstract interface that all LLM backends implement."""

    def __init__(self, config: BackendConfig) -> None:
        self._config = config

    @property
    def label(self) -> str:
        return self._config.label or self._config.provider

    @property
    def model(self) -> str:
        return self._config.model

    @abstractmethod
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
            json_schema: Optional JSON schema dict to enforce structured output.
                         Provider support varies — OpenRouter enforces it
                         server-side, others use it as prompt guidance.
            context:     Optional label for retry log messages (e.g. "batch 3").

        Returns:
            The model's response as a plain string.

        Raises:
            LLMBackendError: If the request fails.
        """
        ...


class LLMBackendError(Exception):
    """Raised when an LLM backend call fails."""
    pass