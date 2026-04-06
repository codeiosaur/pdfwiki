"""
Tests for BackendPool — multi-backend round-robin dispatcher.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backend.base import LLMBackendError
from backend.pool import BackendPool
from backend.factory import create_pass_backends_from_config, backends_config_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TrackingBackend:
    """Records which calls it received; behaves like LLMBackend for the pool."""

    def __init__(self, name: str, response: str = "ok") -> None:
        from backend.base import BackendConfig
        self._config = BackendConfig(
            provider="openai_compat",
            base_url="http://test:0/v1",
            model=f"mock-{name}",
            label=name,
        )
        self.name = name
        self.response = response
        self.calls: list[str] = []

    @property
    def label(self) -> str:
        return self.name

    @property
    def model(self) -> str:
        return f"mock-{self.name}"

    def generate(self, prompt: str, max_tokens=None, json_schema=None, context="", system_prompt=None) -> str:
        self.calls.append(prompt)
        return self.response

    def metrics(self) -> dict:
        return {"total_requests": len(self.calls), "retry_count": 0, "fallback_hops": 0}


class _FailingBackend(_TrackingBackend):
    def generate(self, prompt: str, max_tokens=None, json_schema=None, context="", system_prompt=None) -> str:
        raise LLMBackendError(f"{self.name} failed")


# ---------------------------------------------------------------------------
# BackendPool unit tests
# ---------------------------------------------------------------------------

class TestBackendPoolRoundRobin:
    def test_single_backend_all_calls_go_there(self):
        b = _TrackingBackend("only")
        pool = BackendPool([b], label="test")
        for i in range(5):
            pool.generate(f"prompt-{i}")
        assert len(b.calls) == 5

    def test_two_backends_alternate(self):
        a = _TrackingBackend("a")
        b = _TrackingBackend("b")
        pool = BackendPool([a, b], label="test")
        for i in range(6):
            pool.generate(f"p{i}")
        assert len(a.calls) == 3
        assert len(b.calls) == 3

    def test_three_backends_distribute_evenly(self):
        backends = [_TrackingBackend(str(i)) for i in range(3)]
        pool = BackendPool(backends, label="test")
        for i in range(9):
            pool.generate(f"p{i}")
        for b in backends:
            assert len(b.calls) == 3

    def test_returns_correct_response(self):
        a = _TrackingBackend("a", response="from-a")
        b = _TrackingBackend("b", response="from-b")
        pool = BackendPool([a, b], label="test")
        results = [pool.generate("x") for _ in range(4)]
        assert results == ["from-a", "from-b", "from-a", "from-b"]

    def test_empty_pool_raises(self):
        with pytest.raises(ValueError):
            BackendPool([], label="empty")


class TestBackendPoolMetrics:
    def test_aggregates_call_counts(self):
        a = _TrackingBackend("a")
        b = _TrackingBackend("b")
        pool = BackendPool([a, b], label="test")
        for _ in range(4):
            pool.generate("x")
        m = pool.metrics()
        assert m["total_requests"] == 4

    def test_label_property(self):
        pool = BackendPool([_TrackingBackend("x")], label="my-pool")
        assert pool.label == "my-pool"

    def test_member_labels(self):
        a = _TrackingBackend("alpha")
        b = _TrackingBackend("beta")
        pool = BackendPool([a, b], label="test")
        assert pool.member_labels() == ["alpha", "beta"]

    def test_model_summary(self):
        a = _TrackingBackend("a")
        b = _TrackingBackend("b")
        pool = BackendPool([a, b], label="test")
        assert "mock-a" in pool.model
        assert "mock-b" in pool.model


class TestBackendPoolPassthrough:
    def test_passes_json_schema(self):
        received = []

        class _SchemaCapture(_TrackingBackend):
            def generate(self, prompt, max_tokens=None, json_schema=None, context="", system_prompt=None):
                received.append(json_schema)
                return "ok"

        pool = BackendPool([_SchemaCapture("x")], label="test")
        schema = {"type": "array"}
        pool.generate("p", json_schema=schema)
        assert received[0] is schema

    def test_passes_max_tokens(self):
        received = []

        class _TokenCapture(_TrackingBackend):
            def generate(self, prompt, max_tokens=None, json_schema=None, context="", system_prompt=None):
                received.append(max_tokens)
                return "ok"

        pool = BackendPool([_TokenCapture("x")], label="test")
        pool.generate("p", max_tokens=1234)
        assert received[0] == 1234


# ---------------------------------------------------------------------------
# create_pass_backends_from_config integration tests
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG = {
    "backends": {
        "b1": {"base_url": "http://localhost:11434/v1", "model": "test-model"},
    },
    "passes": {
        "pass1": ["b1"],
        "pass2": ["b1"],
        "pass3": ["b1"],
    },
}

_MULTI_BACKEND_CONFIG = {
    "backends": {
        "local": {"base_url": "http://localhost:11434/v1", "model": "small-model"},
        "remote": {"base_url": "http://localhost:11434/v1", "model": "large-model"},
    },
    "passes": {
        "pass1": ["local"],
        "pass2": ["local", "remote"],
        "pass3": ["local"],
    },
}


class TestCreatePassBackendsFromConfig:
    def _write_config(self, tmp_path: Path, cfg: dict) -> str:
        p = tmp_path / "backends.yaml"
        p.write_text(yaml.dump(cfg))
        return str(p)

    def test_returns_three_backends(self, tmp_path):
        path = self._write_config(tmp_path, _MINIMAL_CONFIG)
        p1, p2, p3 = create_pass_backends_from_config(path)
        assert p1 is not None
        assert p2 is not None
        assert p3 is not None

    def test_single_backend_per_pass_is_not_pool(self, tmp_path):
        path = self._write_config(tmp_path, _MINIMAL_CONFIG)
        p1, p2, p3 = create_pass_backends_from_config(path)
        # Single backend — should not be wrapped in a pool
        assert not isinstance(p1, BackendPool)

    def test_multi_backend_pass_is_pool(self, tmp_path):
        path = self._write_config(tmp_path, _MULTI_BACKEND_CONFIG)
        p1, p2, p3 = create_pass_backends_from_config(path)
        assert not isinstance(p1, BackendPool)
        assert isinstance(p2, BackendPool)
        assert not isinstance(p3, BackendPool)

    def test_pool_has_correct_member_count(self, tmp_path):
        path = self._write_config(tmp_path, _MULTI_BACKEND_CONFIG)
        _, p2, _ = create_pass_backends_from_config(path)
        assert isinstance(p2, BackendPool)
        assert len(p2._backends) == 2

    def test_missing_pass_raises(self, tmp_path):
        cfg = {
            "backends": {"b": {"base_url": "http://x/v1", "model": "m"}},
            "passes": {"pass1": ["b"], "pass2": ["b"]},  # missing pass3
        }
        path = self._write_config(tmp_path, cfg)
        with pytest.raises(LLMBackendError, match="pass3"):
            create_pass_backends_from_config(path)

    def test_unknown_backend_reference_raises(self, tmp_path):
        cfg = {
            "backends": {"b": {"base_url": "http://x/v1", "model": "m"}},
            "passes": {"pass1": ["b"], "pass2": ["nonexistent"], "pass3": ["b"]},
        }
        path = self._write_config(tmp_path, cfg)
        with pytest.raises(LLMBackendError, match="nonexistent"):
            create_pass_backends_from_config(path)

    def test_pass_env_override_replaces_pool(self, tmp_path, monkeypatch):
        """If PASS2_MODEL is set, pass2 should be a single backend, not the pool."""
        path = self._write_config(tmp_path, _MULTI_BACKEND_CONFIG)
        monkeypatch.setenv("PASS2_MODEL", "override-model")
        monkeypatch.setenv("PASS2_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("PASS2_PROVIDER", "openai_compat")
        _, p2, _ = create_pass_backends_from_config(path)
        assert not isinstance(p2, BackendPool)
        assert p2.model == "override-model"


class TestBackendsConfigPath:
    def test_returns_none_when_absent(self, tmp_path):
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            assert backends_config_path() is None
        finally:
            os.chdir(original_cwd)

    def test_returns_path_when_present(self, tmp_path):
        (tmp_path / "backends.yaml").write_text("backends: {}\npasses: {}\n")
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = backends_config_path()
            assert result is not None
            assert result.endswith("backends.yaml")
        finally:
            os.chdir(original_cwd)
