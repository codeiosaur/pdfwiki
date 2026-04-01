"""Tests for extract.fact_extractor — two-pass extraction and batch defaults."""

import inspect
import json

import pytest

from extract.fact_extractor import (
    extract_raw_statements_batched,
    assign_concepts_to_statements,
    derive_seed_concepts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Chunk:
    """Minimal stand-in for ingest.pdf_loader.Chunk."""
    def __init__(self, id: str, text: str):
        self.id = id
        self.text = text


class _Backend:
    """Minimal mock backend that returns canned JSON responses in order."""
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_log: list[str] = []

    def generate(self, prompt: str, max_tokens=None, json_schema=None, context: str = "") -> str:
        self.call_log.append(prompt)
        if self._responses:
            return self._responses.pop(0)
        return "[]"


# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------

class TestDefaultBatchSizes:
    def test_extract_raw_statements_default_batch_size_is_4(self):
        sig = inspect.signature(extract_raw_statements_batched)
        assert sig.parameters["batch_size"].default == 4

    def test_assign_concepts_default_batch_size_is_16(self):
        sig = inspect.signature(assign_concepts_to_statements)
        assert sig.parameters["batch_size"].default == 16

    def test_derive_seed_concepts_default_target_count_is_40(self):
        sig = inspect.signature(derive_seed_concepts)
        assert sig.parameters["target_count"].default == 40

    def test_derive_seed_concepts_default_sample_size_is_120(self):
        sig = inspect.signature(derive_seed_concepts)
        assert sig.parameters["sample_size"].default == 120


# ---------------------------------------------------------------------------
# extract_raw_statements_batched — batching behaviour
# ---------------------------------------------------------------------------

class TestExtractRawStatementsBatched:
    def _make_chunks(self, n: int) -> list:
        return [_Chunk(id=f"chunk-{i}", text=f"Text about topic {i}.") for i in range(n)]

    def _response_for_chunk(self, chunk_id: str) -> str:
        return json.dumps([{"statement": f"Fact from {chunk_id}.", "source_chunk_id": chunk_id}])

    def test_returns_empty_for_no_chunks(self):
        backend = _Backend([])
        assert extract_raw_statements_batched([], backend) == []
        assert backend.call_log == []

    def test_single_batch_when_chunks_le_batch_size(self):
        chunks = self._make_chunks(4)
        # 4 chunks, default batch_size=4 → exactly 1 LLM call
        response = json.dumps([
            {"statement": f"Fact from {c.id}.", "source_chunk_id": c.id}
            for c in chunks
        ])
        backend = _Backend([response])
        results = extract_raw_statements_batched(chunks, backend, batch_size=4)
        assert len(backend.call_log) == 1
        assert len(results) == 4

    def test_two_batches_for_five_chunks_at_batch_size_4(self):
        chunks = self._make_chunks(5)
        # batch 1: chunks 0-3, batch 2: chunk 4
        resp1 = json.dumps([
            {"statement": f"Fact from {c.id}.", "source_chunk_id": c.id}
            for c in chunks[:4]
        ])
        resp2 = json.dumps([
            {"statement": f"Fact from {chunks[4].id}.", "source_chunk_id": chunks[4].id}
        ])
        backend = _Backend([resp1, resp2])
        results = extract_raw_statements_batched(chunks, backend, batch_size=4)
        assert len(backend.call_log) == 2
        assert len(results) == 5

    def test_explicit_batch_size_overrides_default(self):
        # batch_size=2 with 4 chunks → 2 LLM calls
        chunks = self._make_chunks(4)
        responses = [
            json.dumps([
                {"statement": f"Fact from {c.id}.", "source_chunk_id": c.id}
                for c in chunks[i:i+2]
            ])
            for i in range(0, 4, 2)
        ]
        backend = _Backend(responses)
        results = extract_raw_statements_batched(chunks, backend, batch_size=2)
        assert len(backend.call_log) == 2
        assert len(results) == 4

    def test_rejects_statements_with_wrong_chunk_id(self):
        chunks = self._make_chunks(1)
        response = json.dumps([
            {"statement": "Valid fact.", "source_chunk_id": chunks[0].id},
            {"statement": "Spoofed fact.", "source_chunk_id": "chunk-999"},
        ])
        backend = _Backend([response])
        results = extract_raw_statements_batched(chunks, backend, batch_size=4)
        assert len(results) == 1
        assert results[0]["statement"] == "Valid fact."

    def test_handles_malformed_json_gracefully(self):
        chunks = self._make_chunks(1)
        backend = _Backend(["not json at all"])
        results = extract_raw_statements_batched(chunks, backend, batch_size=4)
        assert results == []

    def test_batch_size_1_sends_one_call_per_chunk(self):
        chunks = self._make_chunks(3)
        responses = [
            json.dumps([{"statement": f"Fact.", "source_chunk_id": c.id}])
            for c in chunks
        ]
        backend = _Backend(responses)
        extract_raw_statements_batched(chunks, backend, batch_size=1)
        assert len(backend.call_log) == 3


# ---------------------------------------------------------------------------
# assign_concepts_to_statements — batching behaviour
# ---------------------------------------------------------------------------

class TestAssignConceptsToStatements:
    def _make_statements(self, n: int) -> list[dict]:
        return [{"statement": f"Statement number {i}.", "chunk_id": f"chunk-{i}"} for i in range(n)]

    def _response_for(self, statements: list[dict], offset: int = 0) -> str:
        return json.dumps([
            {"index": j + 1, "concept": "Test Concept"}
            for j, _ in enumerate(statements)
        ])

    def test_returns_empty_for_no_statements(self):
        from extract.fact_extractor import assign_concepts_to_statements
        backend = _Backend([])
        assert assign_concepts_to_statements([], backend, seed_concepts=[]) == []

    def test_single_batch_when_statements_le_batch_size(self):
        stmts = self._make_statements(16)
        backend = _Backend([self._response_for(stmts)])
        facts = assign_concepts_to_statements(stmts, backend, seed_concepts=["Test Concept"], batch_size=16)
        assert len(backend.call_log) == 1
        assert len(facts) == 16

    def test_two_batches_for_17_statements_at_batch_size_16(self):
        stmts = self._make_statements(17)
        resp1 = self._response_for(stmts[:16])
        resp2 = self._response_for(stmts[16:])
        backend = _Backend([resp1, resp2])
        facts = assign_concepts_to_statements(stmts, backend, seed_concepts=["Test Concept"], batch_size=16)
        assert len(backend.call_log) == 2
        assert len(facts) == 17

    def test_explicit_batch_size_overrides_default(self):
        stmts = self._make_statements(6)
        responses = [self._response_for(stmts[i:i+3]) for i in range(0, 6, 3)]
        backend = _Backend(responses)
        facts = assign_concepts_to_statements(stmts, backend, seed_concepts=["Test Concept"], batch_size=3)
        assert len(backend.call_log) == 2
        assert len(facts) == 6

    def test_handles_malformed_json_gracefully(self):
        stmts = self._make_statements(2)
        backend = _Backend(["not json"])
        facts = assign_concepts_to_statements(stmts, backend, seed_concepts=[], batch_size=16)
        assert facts == []
