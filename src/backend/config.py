"""
Environment variable loading for LLM backend configuration.

All secrets (API keys) are read from environment variables or a .env file.
Keys are NEVER logged, printed, or included in error messages.
"""

import os
from pathlib import Path
from typing import Optional


def _load_dotenv() -> None:
    """
    Load variables from a .env file in the project root, if it exists.

    Only sets variables that aren't already in the environment
    (real env vars take precedence over .env).
    """
    # Walk up from this file to find the project root .env
    search = Path(__file__).resolve().parent.parent
    for _ in range(5):  # Don't search forever
        env_path = search / ".env"
        if env_path.is_file():
            break
        parent = search.parent
        if parent == search:
            return  # Hit filesystem root
        search = parent
    else:
        return

    if not env_path.is_file():
        return

    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            # Don't override existing env vars
            if key and key not in os.environ:
                os.environ[key] = value


def _mask_key(key: Optional[str]) -> str:
    """Report whether an API key is set, without revealing any part of it."""
    if not key:
        return "(not set)"
    return "(set)"


def get_env(name: str, default: str = "") -> str:
    """Get an environment variable, loading .env first if needed."""
    _load_dotenv()
    return os.getenv(name, default)


def get_env_secret(name: str) -> Optional[str]:
    """
    Get a secret environment variable (API key).

    Returns None if not set.  Never logs the actual value.
    """
    _load_dotenv()
    value = os.getenv(name)
    if value and not value.strip():
        return None
    return value.strip() if value else None


def log_backend_config(label: str, provider: str, base_url: str,
                       model: str, api_key: Optional[str]) -> None:
    """Print backend config with masked API key for debugging."""
    print(f"  [{label}] provider={provider} base_url={base_url} "
          f"model={model} api_key={_mask_key(api_key)}")