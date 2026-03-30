"""
Backend factory — builds LLM backend instances from environment variables.

Supports three modes controlled by environment variables:

  LOCAL ONLY (default):
    LLM_PROVIDER=openai_compat
    LLM_BASE_URL=http://localhost:11434/v1
    LLM_MODEL=llama3.1:8b

  API ONLY:
    LLM_PROVIDER=anthropic
    LLM_MODEL=claude-haiku-4-5-20251001
    ANTHROPIC_API_KEY=sk-ant-...

  HYBRID (local extraction, API for concept assignment):
    PASS1_PROVIDER=openai_compat
    PASS1_BASE_URL=http://localhost:11434/v1
    PASS1_MODEL=llama3.1:8b
    PASS2_PROVIDER=anthropic
    PASS2_MODEL=claude-haiku-4-5-20251001
    ANTHROPIC_API_KEY=sk-ant-...

When PASS1_*/PASS2_* variables are set, they override the global LLM_*
variables for that specific pass.  This lets users run extraction locally
and concept assignment via API.
"""

from typing import Optional

from backend.base import BackendConfig, LLMBackend, LLMBackendError
from backend.config import get_env, get_env_secret, log_backend_config


# ── Defaults ──────────────────────────────────────────────────────────

_DEFAULT_PROVIDER = "openai_compat"
_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL = "llama3.1:8b"
_DEFAULT_MAX_TOKENS = 800
_DEFAULT_TEMPERATURE = 0.0


# ── Provider registry ─────────────────────────────────────────────────

def _resolve_api_key(provider: str) -> Optional[str]:
    """
    Look up the API key for a provider from environment variables.

    Key lookup order:
      1. LLM_API_KEY          (generic, works for any provider)
      2. ANTHROPIC_API_KEY     (Anthropic-specific)
      3. OPENROUTER_API_KEY    (OpenRouter-specific)
      4. OPENAI_API_KEY        (OpenAI-compatible services that need a key)

    Returns None if no key is found (fine for local endpoints).
    """
    # Generic key takes priority — lets users set one key for everything
    generic = get_env_secret("LLM_API_KEY")
    if generic:
        return generic

    if provider == "anthropic":
        return get_env_secret("ANTHROPIC_API_KEY")

    # For openai_compat, check provider-specific keys
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        key = get_env_secret(var)
        if key:
            return key

    return None


def _build_config(
    provider: str,
    base_url: str,
    model: str,
    label: str,
    api_key: Optional[str] = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> BackendConfig:
    """Build a BackendConfig, resolving the API key if not provided."""
    if api_key is None:
        api_key = _resolve_api_key(provider)

    # Set default base_url for Anthropic if not explicitly provided
    if provider == "anthropic" and base_url == _DEFAULT_BASE_URL:
        base_url = "https://api.anthropic.com"

    return BackendConfig(
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        label=label,
    )


def _create_backend_from_config(config: BackendConfig) -> LLMBackend:
    """Instantiate the correct backend class from a config."""
    if config.provider == "openai_compat":
        from backend.openai_api import OpenAICompatBackend
        return OpenAICompatBackend(config)
    elif config.provider == "anthropic":
        from backend.anthropic import AnthropicBackend
        return AnthropicBackend(config)
    else:
        raise LLMBackendError(f"Unknown provider: {config.provider}")


# ── Public API ────────────────────────────────────────────────────────

def create_backend(label: str = "default") -> LLMBackend:
    """
    Create a single LLM backend from global LLM_* environment variables.

    Environment variables:
        LLM_PROVIDER   — 'openai_compat' or 'anthropic'  (default: openai_compat)
        LLM_BASE_URL   — API base URL  (default: http://localhost:11434/v1)
        LLM_MODEL      — Model identifier  (default: llama3.1:8b)
        LLM_API_KEY    — API key (optional for local endpoints)
    """
    provider = get_env("LLM_PROVIDER", _DEFAULT_PROVIDER)
    base_url = get_env("LLM_BASE_URL", _DEFAULT_BASE_URL)
    model = get_env("LLM_MODEL", _DEFAULT_MODEL)

    config = _build_config(provider, base_url, model, label=label)
    log_backend_config(label, provider, base_url, model, config.api_key)

    return _create_backend_from_config(config)


def create_pass_backends() -> tuple[LLMBackend, LLMBackend]:
    """
    Create separate backends for Pass 1 (extraction) and Pass 2 (concept assignment).

    If PASS1_* or PASS2_* env vars are set, those override the global LLM_* vars
    for that pass.  This enables hybrid mode (e.g., local extraction + API assignment).

    Environment variables (all optional — fall back to LLM_* globals):
        PASS1_PROVIDER, PASS1_BASE_URL, PASS1_MODEL
        PASS2_PROVIDER, PASS2_BASE_URL, PASS2_MODEL
    """
    global_provider = get_env("LLM_PROVIDER", _DEFAULT_PROVIDER)
    global_base_url = get_env("LLM_BASE_URL", _DEFAULT_BASE_URL)
    global_model = get_env("LLM_MODEL", _DEFAULT_MODEL)

    # Pass 1: extraction (defaults to global/local)
    p1_provider = get_env("PASS1_PROVIDER", global_provider)
    p1_base_url = get_env("PASS1_BASE_URL", global_base_url)
    p1_model = get_env("PASS1_MODEL", global_model)

    p1_config = _build_config(p1_provider, p1_base_url, p1_model, label="pass1-extract")
    log_backend_config("pass1-extract", p1_provider, p1_base_url, p1_model, p1_config.api_key)
    pass1 = _create_backend_from_config(p1_config)

    # Pass 2: concept assignment (can be a different provider/model)
    p2_provider = get_env("PASS2_PROVIDER", global_provider)
    p2_base_url = get_env("PASS2_BASE_URL", global_base_url)
    p2_model = get_env("PASS2_MODEL", global_model)

    p2_config = _build_config(p2_provider, p2_base_url, p2_model, label="pass2-assign")
    log_backend_config("pass2-assign", p2_provider, p2_base_url, p2_model, p2_config.api_key)
    pass2 = _create_backend_from_config(p2_config)

    return pass1, pass2