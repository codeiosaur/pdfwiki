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
_DEFAULT_TEMPERATURE = 0.0

# The old 900-token cap predated the batch_size=4 / 10-20-statements-per-chunk changes.
# At 4 chunks × 15 statements × ~30 tokens each the response easily exceeds 900 tokens
# and the JSON gets truncated.  Modern Ollama models handle 4096 tokens without issue.
_DEFAULT_MAX_TOKENS = 4096


def _default_max_tokens() -> int:
    """Return the default max_tokens for all endpoints."""
    return _DEFAULT_MAX_TOKENS


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag from common truthy/falsey strings."""
    raw = get_env(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


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
    max_tokens: Optional[int] = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    openrouter_zdr: bool = False,
) -> BackendConfig:
    """Build a BackendConfig, resolving the API key if not provided."""
    if api_key is None:
        api_key = _resolve_api_key(provider)
    if max_tokens is None:
        max_tokens = _default_max_tokens()

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
        openrouter_zdr=openrouter_zdr,
    )


def _create_backend_from_config(config: BackendConfig) -> LLMBackend:
    """Instantiate the correct backend class from a config."""
    if config.provider == "openai_compat":
        from backend.openai_compat import OpenAICompatBackend
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
    max_tokens_raw = get_env("LLM_MAX_TOKENS", "")
    max_tokens = int(max_tokens_raw) if max_tokens_raw.isdigit() else None
    openrouter_zdr = _env_flag("OPENROUTER_ZDR", default=False)

    config = _build_config(
        provider,
        base_url,
        model,
        label=label,
        max_tokens=max_tokens,
        openrouter_zdr=openrouter_zdr,
    )
    log_backend_config(label, provider, base_url, model, config.api_key)

    return _create_backend_from_config(config)


def create_pass_backends() -> tuple[LLMBackend, LLMBackend, LLMBackend]:
    """
    Create separate backends for Pass 1 (extraction), Pass 2 (concept assignment),
    and Pass 3 (synthesis).

    If PASS1_*/PASS2_*/PASS3_* env vars are set, those override the global LLM_*
    vars for that pass.  All three passes fall back to LLM_* globals when their
    PASS*_* vars are absent.

    Environment variables (all optional — fall back to LLM_* globals):
        PASS1_PROVIDER, PASS1_BASE_URL, PASS1_MODEL
        PASS2_PROVIDER, PASS2_BASE_URL, PASS2_MODEL
        PASS3_PROVIDER, PASS3_BASE_URL, PASS3_MODEL
    """
    global_provider = get_env("LLM_PROVIDER", _DEFAULT_PROVIDER)
    global_base_url = get_env("LLM_BASE_URL", _DEFAULT_BASE_URL)
    global_model = get_env("LLM_MODEL", _DEFAULT_MODEL)
    global_openrouter_zdr = _env_flag("OPENROUTER_ZDR", default=False)

    global_max_tokens_raw = get_env("LLM_MAX_TOKENS", "")
    global_max_tokens = int(global_max_tokens_raw) if global_max_tokens_raw.isdigit() else None

    # Pass 1: extraction (defaults to global/local)
    p1_provider = get_env("PASS1_PROVIDER", global_provider)
    p1_base_url = get_env("PASS1_BASE_URL", global_base_url)
    p1_model = get_env("PASS1_MODEL", global_model)
    p1_max_tokens_raw = get_env("PASS1_MAX_TOKENS", "")
    p1_max_tokens = int(p1_max_tokens_raw) if p1_max_tokens_raw.isdigit() else global_max_tokens
    p1_openrouter_zdr = _env_flag("PASS1_ZDR", default=global_openrouter_zdr)

    p1_config = _build_config(
        p1_provider,
        p1_base_url,
        p1_model,
        label="pass1-extract",
        max_tokens=p1_max_tokens,
        openrouter_zdr=p1_openrouter_zdr,
    )
    log_backend_config("pass1-extract", p1_provider, p1_base_url, p1_model, p1_config.api_key)
    pass1 = _create_backend_from_config(p1_config)

    # Configure fallback models for Pass 1 (OpenRouter-specific)
    p1_fallback_models_raw = get_env("PASS1_FALLBACK_MODELS", "")
    if p1_fallback_models_raw.strip() and getattr(pass1, "is_openrouter", False):
        p1_fallback_models = [m.strip() for m in p1_fallback_models_raw.split(",") if m.strip()]
        if p1_fallback_models:
            pass1.set_fallback_models(p1_fallback_models)
            print(f"  [pass1-extract] Fallback models: {p1_fallback_models}")

    # Pass 2: concept assignment (can be a different provider/model)
    p2_provider = get_env("PASS2_PROVIDER", global_provider)
    p2_base_url = get_env("PASS2_BASE_URL", global_base_url)
    p2_model = get_env("PASS2_MODEL", global_model)
    p2_max_tokens_raw = get_env("PASS2_MAX_TOKENS", "")
    p2_max_tokens = int(p2_max_tokens_raw) if p2_max_tokens_raw.isdigit() else global_max_tokens
    p2_openrouter_zdr = _env_flag("PASS2_ZDR", default=global_openrouter_zdr)

    p2_config = _build_config(
        p2_provider,
        p2_base_url,
        p2_model,
        label="pass2-assign",
        max_tokens=p2_max_tokens,
        openrouter_zdr=p2_openrouter_zdr,
    )
    log_backend_config("pass2-assign", p2_provider, p2_base_url, p2_model, p2_config.api_key)
    pass2 = _create_backend_from_config(p2_config)

    # Configure fallback models for Pass 2 (OpenRouter-specific)
    fallback_models_raw = get_env("PASS2_FALLBACK_MODELS", "")
    if fallback_models_raw.strip() and getattr(pass2, "is_openrouter", False):
        fallback_models = [m.strip() for m in fallback_models_raw.split(",") if m.strip()]
        if fallback_models:
            pass2.set_fallback_models(fallback_models)
            print(f"  [pass2-assign] Fallback models: {fallback_models}")

    # Pass 3: synthesis (falls back to LLM_* globals, same as Pass 1 and Pass 2)
    p3_provider = get_env("PASS3_PROVIDER", global_provider)
    p3_base_url = get_env("PASS3_BASE_URL", global_base_url)
    p3_model = get_env("PASS3_MODEL", global_model)
    p3_max_tokens_raw = get_env("PASS3_MAX_TOKENS", "")
    p3_max_tokens = int(p3_max_tokens_raw) if p3_max_tokens_raw.isdigit() else global_max_tokens
    p3_openrouter_zdr = _env_flag("PASS3_ZDR", default=global_openrouter_zdr)

    p3_config = _build_config(
        p3_provider,
        p3_base_url,
        p3_model,
        label="pass3-synthesize",
        max_tokens=p3_max_tokens,
        openrouter_zdr=p3_openrouter_zdr,
    )
    log_backend_config("pass3-synthesize", p3_provider, p3_base_url, p3_model, p3_config.api_key)
    pass3 = _create_backend_from_config(p3_config)

    # Configure fallback models for Pass 3 (OpenRouter-specific)
    p3_fallback_models_raw = get_env("PASS3_FALLBACK_MODELS", "")
    if p3_fallback_models_raw.strip() and getattr(pass3, "is_openrouter", False):
        p3_fallback_models = [m.strip() for m in p3_fallback_models_raw.split(",") if m.strip()]
        if p3_fallback_models:
            pass3.set_fallback_models(p3_fallback_models)
            print(f"  [pass3-synthesize] Fallback models: {p3_fallback_models}")

    return pass1, pass2, pass3