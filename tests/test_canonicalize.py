"""Canonicalization should be subject-neutral and retry transient failures."""

from unittest.mock import Mock

import transform.canonicalize as canonicalize
from extract.fact_extractor import Fact
from postprocess import consolidate_concepts_llm


def test_prompt_preserves_meaning_and_avoids_accounting_examples(tmp_path, monkeypatch):
    monkeypatch.setattr(canonicalize, "CANONICAL_CACHE_PATH", tmp_path / "cache.json")
    backend = Mock()
    backend.generate.return_value = '{"LCM": "LCM", "Average Velocity": "Average Velocity"}'

    result = canonicalize.canonicalize_concepts(["LCM", "Average Velocity"], backend)

    assert result == {"LCM": "LCM", "Average Velocity": "Average Velocity"}
    prompt = backend.generate.call_args.args[0]
    assert "unambiguous" in prompt
    assert "Cost of Goods Sold" not in prompt
    assert "FIFO" not in prompt


def test_backend_failure_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(canonicalize, "CANONICAL_CACHE_PATH", tmp_path / "cache.json")
    backend = Mock()
    backend.generate.side_effect = RuntimeError("temporarily unavailable")

    assert canonicalize.canonicalize_concepts(["LCM"], backend) == {"LCM": None}
    assert not (tmp_path / "cache.json").exists()

    backend.generate.side_effect = None
    backend.generate.return_value = '{"LCM": "LCM"}'
    assert canonicalize.canonicalize_concepts(["LCM"], backend) == {"LCM": "LCM"}


def test_incomplete_response_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(canonicalize, "CANONICAL_CACHE_PATH", tmp_path / "cache.json")
    backend = Mock()
    backend.generate.return_value = '{}'

    assert canonicalize.canonicalize_concepts(["Graph Theory"], backend) == {"Graph Theory": None}
    backend.generate.return_value = '{"Graph Theory": "Graph Theory"}'
    assert canonicalize.canonicalize_concepts(["Graph Theory"], backend) == {"Graph Theory": "Graph Theory"}
    assert backend.generate.call_count == 2


def test_consolidation_prompt_is_domain_neutral():
    names = ("Graph", "Graph Theory", "Public Key Cryptography", "Public-Key Cryptography")
    grouped = {
        name: [Fact(name, name, "A claim.", "chunk-1")]
        for name in names
    }
    backend = Mock()
    backend.generate.return_value = "{}"

    assert consolidate_concepts_llm(grouped, backend) == grouped
    prompt = backend.generate.call_args.args[0]
    assert "Inventory" not in prompt
    assert "FIFO" not in prompt
