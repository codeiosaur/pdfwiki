"""
LLM backend abstraction layer.

Provides a unified interface for calling language models from any provider:
local (Ollama, LM Studio), cloud API (Anthropic, OpenRouter), or
self-hosted (Runpod, vLLM).

Usage:
    from backend import create_backend, create_pass_backends

    # Single backend for all LLM calls
    backend = create_backend()
    response = backend.generate("Extract facts from this text...")

    # Hybrid: local extraction + API concept assignment
    pass1, pass2 = create_pass_backends()
    statements = extract_with(pass1, chunks)
    facts = assign_with(pass2, statements)
"""

from backend.base import BackendConfig, LLMBackend, LLMBackendError
from backend.factory import create_backend, create_pass_backends

__all__ = [
    "BackendConfig",
    "LLMBackend",
    "LLMBackendError",
    "create_backend",
    "create_pass_backends",
]