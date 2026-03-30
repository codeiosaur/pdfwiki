"""
Tests for the backend abstraction layer.

These are all deterministic — no LLM calls, no network, no Ollama needed.
"""

import os
import pytest

from backend.base import BackendConfig, LLMBackendError
from backend.config import _mask_key, get_env, get_env_secret, log_backend_config


# ===================================================================
# BackendConfig tests
# ===================================================================

class TestBackendConfig:
    def test_valid_openai_compat(self):
        config = BackendConfig(
            provider="openai_compat",
            base_url="http://localhost:11434/v1",
            model="llama3.1:8b",
        )
        assert config.provider == "openai_compat"
        assert config.api_key is None

    def test_valid_anthropic(self):
        config = BackendConfig(
            provider="anthropic",
            base_url="https://api.anthropic.com",
            model="claude-haiku-4-5-20251001",
            api_key="sk-ant-test",
        )
        assert config.provider == "anthropic"

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            BackendConfig(
                provider="gpt4all",
                base_url="http://localhost:1234",
                model="some-model",
            )

    def test_api_key_hidden_from_repr(self):
        config = BackendConfig(
            provider="openai_compat",
            base_url="http://localhost:11434/v1",
            model="llama3.1:8b",
            api_key="super-secret-key-12345-abcdef",
        )
        repr_str = repr(config)
        assert "super-secret" not in repr_str
        assert "api_key" not in repr_str  # repr=False hides the field entirely

    def test_api_key_hidden_from_str(self):
        config = BackendConfig(
            provider="openai_compat",
            base_url="http://localhost:11434/v1",
            model="llama3.1:8b",
            api_key="sk-ant-my-real-key-here",
        )
        str_repr = str(config)
        assert "my-real-key" not in str_repr

    def test_frozen_config(self):
        config = BackendConfig(
            provider="openai_compat",
            base_url="http://localhost:11434/v1",
            model="llama3.1:8b",
        )
        with pytest.raises(AttributeError):
            config.model = "different-model"  # type: ignore[misc]

    def test_default_values(self):
        config = BackendConfig(
            provider="openai_compat",
            base_url="http://localhost:11434/v1",
            model="llama3.1:8b",
        )
        assert config.temperature == 0.0
        assert config.max_tokens == 800
        assert config.label == ""


# ===================================================================
# Key masking tests
# ===================================================================

class TestKeyMasking:
    def test_mask_none(self):
        assert _mask_key(None) == "(not set)"

    def test_mask_empty(self):
        assert _mask_key("") == "(not set)"

    def test_mask_short_key(self):
        result = _mask_key("abc")
        assert result == "(set)"
        assert "abc" not in result

    def test_mask_normal_key(self):
        result = _mask_key("sk-ant-1234567890abcdef")
        assert result == "(set)"
        assert "sk-ant" not in result
        assert "1234567890" not in result
        assert "cdef" not in result

    def test_mask_exact_boundary(self):
        result = _mask_key("12345678")
        assert result == "(set)"

    def test_mask_nine_chars(self):
        result = _mask_key("123456789")
        assert result == "(set)"

    def test_mask_never_reveals_any_key_content(self):
        """No key, regardless of length, should appear in any form."""
        for key in ["a", "ab", "abc", "abcd", "abcde", "sk-ant-xxxxxxxxxxxx"]:
            masked = _mask_key(key)
            if key:  # non-empty keys
                assert key not in masked
                assert masked == "(set)"


# ===================================================================
# MockBackend tests (validates the test fixture itself)
# ===================================================================

class TestMockBackend:
    def test_returns_responses_in_order(self, mock_backend):
        mock_backend.responses = ["first", "second", "third"]
        assert mock_backend.generate("a") == "first"
        assert mock_backend.generate("b") == "second"
        assert mock_backend.generate("c") == "third"

    def test_returns_default_when_empty(self, mock_backend):
        mock_backend.default_response = "fallback"
        assert mock_backend.generate("anything") == "fallback"

    def test_logs_calls(self, mock_backend):
        mock_backend.generate("hello", max_tokens=100)
        mock_backend.generate("world", max_tokens=200)
        assert len(mock_backend.call_log) == 2
        assert mock_backend.call_log[0]["prompt"] == "hello"
        assert mock_backend.call_log[0]["max_tokens"] == 100
        assert mock_backend.call_log[1]["prompt"] == "world"

    def test_failing_backend_raises(self, failing_backend):
        with pytest.raises(LLMBackendError, match="Mock failure"):
            failing_backend.generate("anything")

    def test_model_property(self, mock_backend):
        assert mock_backend.model == "mock-model"

    def test_label_property(self, mock_backend):
        assert mock_backend.label == "mock"


# ===================================================================
# Environment variable loading tests
# ===================================================================

class TestEnvLoading:
    def test_get_env_returns_value(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR_ABC", "hello")
        assert get_env("TEST_VAR_ABC") == "hello"

    def test_get_env_returns_default(self):
        # Use a var name that definitely doesn't exist
        assert get_env("DEFINITELY_NOT_SET_12345", "fallback") == "fallback"

    def test_get_env_secret_returns_none_when_missing(self):
        assert get_env_secret("DEFINITELY_NOT_SET_SECRET_12345") is None

    def test_get_env_secret_returns_stripped(self, monkeypatch):
        monkeypatch.setenv("TEST_SECRET", "  sk-test-key  ")
        assert get_env_secret("TEST_SECRET") == "sk-test-key"

    def test_get_env_secret_returns_none_for_blank(self, monkeypatch):
        monkeypatch.setenv("TEST_SECRET_BLANK", "   ")
        assert get_env_secret("TEST_SECRET_BLANK") is None


# ===================================================================
# Factory tests (without actually creating backends — just config logic)
# ===================================================================

class TestFactory:
    def test_default_config_is_local_ollama(self, monkeypatch):
        """With no env vars set, factory should default to local Ollama."""
        # Clear any existing vars
        for var in ["LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL",
                     "PASS1_PROVIDER", "PASS2_PROVIDER"]:
            monkeypatch.delenv(var, raising=False)

        from backend.factory import _build_config, _DEFAULT_PROVIDER, _DEFAULT_BASE_URL, _DEFAULT_MODEL
        config = _build_config(
            _DEFAULT_PROVIDER, _DEFAULT_BASE_URL, _DEFAULT_MODEL, label="test"
        )
        assert config.provider == "openai_compat"
        assert "localhost" in config.base_url
        assert config.api_key is None  # No key needed for local

    def test_anthropic_gets_default_base_url(self):
        from backend.factory import _build_config, _DEFAULT_BASE_URL
        config = _build_config(
            "anthropic", _DEFAULT_BASE_URL, "claude-haiku-4-5-20251001",
            label="test", api_key="sk-ant-test",
        )
        assert config.base_url == "https://api.anthropic.com"

    def test_anthropic_preserves_custom_base_url(self):
        from backend.factory import _build_config
        config = _build_config(
            "anthropic", "https://custom-proxy.example.com",
            "claude-haiku-4-5-20251001", label="test", api_key="sk-ant-test",
        )
        assert config.base_url == "https://custom-proxy.example.com"

    def test_resolve_api_key_generic(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "generic-key")
        from backend.factory import _resolve_api_key
        assert _resolve_api_key("openai_compat") == "generic-key"
        assert _resolve_api_key("anthropic") == "generic-key"

    def test_resolve_api_key_anthropic_specific(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        from backend.factory import _resolve_api_key
        assert _resolve_api_key("anthropic") == "ant-key"

    def test_resolve_api_key_openrouter(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        from backend.factory import _resolve_api_key
        assert _resolve_api_key("openai_compat") == "or-key"

    def test_resolve_api_key_none_when_unset(self, monkeypatch):
        for var in ["LLM_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"]:
            monkeypatch.delenv(var, raising=False)
        from backend.factory import _resolve_api_key
        assert _resolve_api_key("openai_compat") is None

    def test_generic_key_takes_priority(self, monkeypatch):
        """LLM_API_KEY should override provider-specific keys."""
        monkeypatch.setenv("LLM_API_KEY", "generic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "specific")
        from backend.factory import _resolve_api_key
        assert _resolve_api_key("anthropic") == "generic"