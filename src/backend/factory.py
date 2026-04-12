"""
Backend factory — builds LLM backend instances from environment variables or
a backends.json config file.

Supports two configuration modes:

  ENV VARS (legacy, still supported):
    PASS1_PROVIDER, PASS1_BASE_URL, PASS1_MODEL, ...
    PASS2_PROVIDER, PASS2_BASE_URL, PASS2_MODEL, ...
    PASS3_PROVIDER, PASS3_BASE_URL, PASS3_MODEL, ...

  backends.json (preferred — supports multiple backends per pass):
    See backends.json.example for format.  When backends.json exists in the
    working directory, it takes precedence over PASS*_* env vars.

When backends.json assigns multiple backends to a pass, calls are distributed
round-robin via BackendPool, giving each backend an equal share of batches.
"""

import os
import re
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


# ── Deprecation warning ───────────────────────────────────────────────

def warn_deprecated_env_vars() -> None:
    """
    Print a tip suggesting backends.yaml when the user is configuring topology
    via env vars alone (no backends.yaml present).

    This is not a deprecation — env vars are fully supported.  backends.yaml
    just unlocks features that env vars can't express (multi-backend pools,
    named backends, per-backend ZDR/fallbacks).
    """
    print(
        "\n  [config] Tip: create a backends.yaml for multi-backend pools, named\n"
        "  backends, and per-backend ZDR/fallback_models.  Env vars still work\n"
        "  fine for simple single-backend-per-pass setups.\n"
        "  See backends.yaml.example for the format.\n"
    )


def _interpolate_env(s: str) -> str:
    """Expand ${VAR_NAME} references in a string using os.environ."""
    return re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), s)


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
    ollama_num_ctx: Optional[int] = None,
    structured_output: bool = False,
    json_mode: bool = False,
    wrap_array_schema: bool = False,
    preferred_batch_size: Optional[int] = None,
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
        ollama_num_ctx=ollama_num_ctx,
        structured_output=structured_output,
        json_mode=json_mode,
        wrap_array_schema=wrap_array_schema,
        preferred_batch_size=preferred_batch_size,
    )


def _create_backend_from_config(config: BackendConfig) -> LLMBackend:
    """Instantiate the correct backend class from a config."""
    if config.provider == "openai_compat":
        from backend.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(config)
    elif config.provider == "anthropic":
        from backend.anthropic import AnthropicBackend
        return AnthropicBackend(config)
    elif config.provider == "gemini":
        from backend.gemini import GeminiBackend
        return GeminiBackend(config)
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


# ── backends.yaml loader ──────────────────────────────────────────────

def _build_backend_from_spec(spec: dict, name: str) -> LLMBackend:
    """
    Build a single LLMBackend from a backends.yaml backend spec dict.

    Required fields: base_url, model
    Optional fields: provider, api_key_env, max_tokens, temperature, fallback_models, zdr,
                     num_ctx (Ollama only: overrides default context window)
    """
    model = spec.get("model", _DEFAULT_MODEL)

    # Infer provider from spec if not explicit
    explicit_provider = spec.get("provider", "")
    if explicit_provider == "anthropic" or "anthropic" in spec.get("base_url", ""):
        provider = "anthropic"
    elif explicit_provider == "gemini":
        provider = "gemini"
    elif explicit_provider:
        provider = explicit_provider
    else:
        provider = "openai_compat"

    # Gemini native backend has no base_url — use a descriptive placeholder for logging
    if provider == "gemini":
        base_url = "googleapis.com (native SDK)"
    else:
        base_url = _interpolate_env(spec.get("base_url", _DEFAULT_BASE_URL))

    # API key: read from the named env var, or fall back to standard key resolution
    api_key_env = spec.get("api_key_env")
    api_key = get_env_secret(api_key_env) if api_key_env else _resolve_api_key(provider)

    max_tokens = spec.get("max_tokens") or _default_max_tokens()
    temperature = spec.get("temperature", _DEFAULT_TEMPERATURE)
    zdr = bool(spec.get("zdr", False))
    num_ctx = spec.get("num_ctx") or None
    structured_output = bool(spec.get("structured_output", False))
    json_mode = bool(spec.get("json_mode", False))
    wrap_array_schema = bool(spec.get("wrap_array_schema", False))
    preferred_batch_size = spec.get("batch_size") or None

    config = _build_config(
        provider, base_url, model,
        label=name,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        openrouter_zdr=zdr,
        ollama_num_ctx=int(num_ctx) if num_ctx else None,
        structured_output=structured_output,
        json_mode=json_mode,
        wrap_array_schema=wrap_array_schema,
        preferred_batch_size=int(preferred_batch_size) if preferred_batch_size else None,
    )
    log_backend_config(name, provider, base_url, model, config.api_key)
    backend = _create_backend_from_config(config)

    fallback_models = spec.get("fallback_models", [])
    if fallback_models and getattr(backend, "is_openrouter", False):
        backend.set_fallback_models(fallback_models)
        print(f"  [{name}] Fallback models: {fallback_models}")

    return backend


def _build_pass_env_override(
    pass_num: int,
    label: str,
    global_provider: str,
    global_base_url: str,
    global_model: str,
    global_max_tokens: Optional[int],
    global_zdr: bool,
) -> LLMBackend:
    """Build a single backend for a pass entirely from PASS{N}_* env vars."""
    prefix = f"PASS{pass_num}"
    provider = get_env(f"{prefix}_PROVIDER", global_provider)
    base_url = get_env(f"{prefix}_BASE_URL", global_base_url)
    model = get_env(f"{prefix}_MODEL", global_model)
    max_tokens_raw = get_env(f"{prefix}_MAX_TOKENS", "")
    max_tokens = int(max_tokens_raw) if max_tokens_raw.isdigit() else global_max_tokens
    zdr = _env_flag(f"{prefix}_ZDR", default=global_zdr)

    config = _build_config(
        provider, base_url, model,
        label=label,
        max_tokens=max_tokens,
        openrouter_zdr=zdr,
    )
    log_backend_config(label, provider, base_url, model, config.api_key)
    backend = _create_backend_from_config(config)

    fallback_raw = get_env(f"{prefix}_FALLBACK_MODELS", "")
    if fallback_raw.strip() and getattr(backend, "is_openrouter", False):
        fallback_models = [m.strip() for m in fallback_raw.split(",") if m.strip()]
        if fallback_models:
            backend.set_fallback_models(fallback_models)
            print(f"  [{label}] Fallback models: {fallback_models}")

    return backend


def create_pass_backends_from_config(
    path: str,
) -> tuple[LLMBackend, LLMBackend, LLMBackend]:
    """
    Build pass backends from a backends.yaml file, with env var override support.

    Precedence (standard Unix chain):
        shell env vars  >  .env file  >  backends.yaml

    If PASS{N}_MODEL or PASS{N}_BASE_URL is set in the environment for a given
    pass, that pass is rebuilt entirely from PASS{N}_* env vars, overriding the
    backends.yaml pool for that pass.  This lets users quickly swap a model
    without editing the file.

    backends.yaml format:
        backends:
          local-fast:
            base_url: http://localhost:11434/v1
            model: gemma3:4b
            workers: 4
          remote:
            base_url: https://openrouter.ai/api/v1
            model: google/gemma-3-27b-it:free
            api_key_env: OPENROUTER_API_KEY
        passes:
          pass1: [local-fast]
          pass2: [local-fast, remote]
          pass3: [local-fast]

    Returns a 3-tuple (pass1, pass2, pass3).
    """
    import yaml
    from backend.pool import BackendPool

    with open(path, "r") as f:
        file_config = yaml.safe_load(f)

    backend_specs: dict = file_config.get("backends", {})
    passes: dict = file_config.get("passes", {})

    # Build all named backends from the file
    built: dict[str, LLMBackend] = {}
    backend_workers: dict[str, int] = {}
    for name, spec in backend_specs.items():
        built[name] = _build_backend_from_spec(spec, name)
        backend_workers[name] = max(1, int(spec.get("workers", 1)))

    # Globals for env var overrides (same fallback chain as create_pass_backends)
    global_provider = get_env("LLM_PROVIDER", _DEFAULT_PROVIDER)
    global_base_url = get_env("LLM_BASE_URL", _DEFAULT_BASE_URL)
    global_model = get_env("LLM_MODEL", _DEFAULT_MODEL)
    global_max_tokens_raw = get_env("LLM_MAX_TOKENS", "")
    global_max_tokens = int(global_max_tokens_raw) if global_max_tokens_raw.isdigit() else None
    global_zdr = _env_flag("OPENROUTER_ZDR", default=False)

    def _resolve_pass(pass_key: str, pass_num: int, label: str) -> LLMBackend:
        # Env var override: if PASS{N}_MODEL or PASS{N}_BASE_URL is explicitly set,
        # rebuild this pass from env vars rather than the YAML pool.
        env_model = get_env(f"PASS{pass_num}_MODEL", "")
        env_base_url = get_env(f"PASS{pass_num}_BASE_URL", "")
        if env_model or env_base_url:
            override_model = env_model or global_model
            print(f"  [{label}] env var override: using PASS{pass_num}_MODEL={override_model!r} "
                  f"(overrides backends.yaml)")
            return _build_pass_env_override(
                pass_num, label,
                global_provider, global_base_url, global_model,
                global_max_tokens, global_zdr,
            )

        names = passes.get(pass_key, [])
        if not names:
            raise LLMBackendError(
                f"backends.yaml: no backends assigned to '{pass_key}'. "
                f"Add a '{pass_key}' entry under 'passes'."
            )
        missing = [n for n in names if n not in built]
        if missing:
            raise LLMBackendError(
                f"backends.yaml: '{pass_key}' references unknown backends: {missing}"
            )
        members = [built[n] for n in names]
        if len(members) == 1:
            return members[0]
        weights = [backend_workers[n] for n in names]
        pool = BackendPool(members, label=label, weights=weights)
        member_summary = ", ".join(f"{n} ({built[n].model}, w={backend_workers[n]})" for n in names)
        print(f"  [{label}] Pool: {member_summary}")
        return pool

    print("\n=== LLM BACKEND CONFIGURATION (backends.yaml) ===")
    pass1 = _resolve_pass("pass1", 1, "pass1-extract")
    pass2 = _resolve_pass("pass2", 2, "pass2-assign")
    pass3 = _resolve_pass("pass3", 3, "pass3-synthesize")

    return pass1, pass2, pass3


def backends_config_path() -> Optional[str]:
    """Return the path to backends.yaml if it exists in the working directory, else None."""
    path = os.path.join(os.getcwd(), "backends.yaml")
    return path if os.path.exists(path) else None