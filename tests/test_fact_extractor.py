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
        self.label = "mock"

    def generate(self, prompt: str, max_tokens=None, json_schema=None, context: str = "", system_prompt=None) -> str:
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


# ---------------------------------------------------------------------------
# Source attribution invariant
# ---------------------------------------------------------------------------

class TestSourceChunkIdPreserved:
    """source_chunk_id from Pass 1 statements must survive through Pass 2 Facts."""

    def _make_statements_with_chunks(self, chunk_ids: list[str]) -> list[dict]:
        return [
            {"statement": f"Statement from {cid}", "chunk_id": cid, "source": "test.pdf"}
            for cid in chunk_ids
        ]

    def _response_for(self, stmts: list[dict]) -> str:
        # assign_concepts_to_statements expects {"index": N, "concept": "..."} items
        import json
        return json.dumps([
            {"index": i + 1, "concept": "Test Concept"}
            for i in range(len(stmts))
        ])

    def _expected_ids(self, chunk_ids: list[str]) -> set[str]:
        # source_chunk_id is composed as "source::chunk_id" when source is present
        return {f"test.pdf::{cid}" for cid in chunk_ids}

    def test_source_chunk_id_preserved_single_batch(self):
        chunk_ids = ["chunk-a", "chunk-b", "chunk-c"]
        stmts = self._make_statements_with_chunks(chunk_ids)
        backend = _Backend([self._response_for(stmts)])
        facts = assign_concepts_to_statements(stmts, backend, seed_concepts=["Test Concept"], batch_size=16)
        returned_ids = {f.source_chunk_id for f in facts}
        assert returned_ids == self._expected_ids(chunk_ids)

    def test_source_chunk_id_preserved_across_batches(self):
        chunk_ids = [f"chunk-{i}" for i in range(6)]
        stmts = self._make_statements_with_chunks(chunk_ids)
        resp1 = self._response_for(stmts[:3])
        resp2 = self._response_for(stmts[3:])
        backend = _Backend([resp1, resp2])
        facts = assign_concepts_to_statements(stmts, backend, seed_concepts=["Test Concept"], batch_size=3)
        returned_ids = {f.source_chunk_id for f in facts}
        assert returned_ids == self._expected_ids(chunk_ids)

    def test_no_fact_has_empty_source_chunk_id(self):
        chunk_ids = ["chunk-x", "chunk-y"]
        stmts = self._make_statements_with_chunks(chunk_ids)
        backend = _Backend([self._response_for(stmts)])
        facts = assign_concepts_to_statements(stmts, backend, seed_concepts=["Test Concept"], batch_size=16)
        assert all(f.source_chunk_id for f in facts), "Every Fact must carry a non-empty source_chunk_id"
