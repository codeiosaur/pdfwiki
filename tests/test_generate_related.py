"""Tests for generate.related — related concept discovery and citation helpers."""

import pytest

from extract.fact_extractor import Fact
from generate.related import (
    looks_like_uuid,
    fact_sources,
    build_related_concepts,
    build_related_concepts_by_chunks,
    citation_suffixes,
)


def make_fact(concept: str, content: str, chunk_id: str) -> Fact:
    return Fact(id="x", concept=concept, content=content, source_chunk_id=chunk_id)


class TestLooksLikeUuid:
    def test_valid_uuid(self):
        assert looks_like_uuid("550e8400-e29b-41d4-a716-446655440000") is True

    def test_uppercase_uuid(self):
        assert looks_like_uuid("550E8400-E29B-41D4-A716-446655440000") is True

    def test_not_uuid_random_string(self):
        assert looks_like_uuid("not-a-uuid") is False

    def test_not_uuid_filename(self):
        assert looks_like_uuid("chapter1.pdf") is False

    def test_empty_string(self):
        assert looks_like_uuid("") is False

    def test_wrong_segment_lengths(self):
        assert looks_like_uuid("550e8400-e29b-41d4-a716") is False


class TestFactSources:
    def test_maps_content_to_chunk_id(self):
        facts = [
            make_fact("FIFO", "Oldest costs are used first.", "chunk-1"),
        ]
        result = fact_sources(facts)
        assert "Oldest costs are used first." in result
        assert result["Oldest costs are used first."] == ["chunk-1"]

    def test_deduplicates_chunk_ids(self):
        facts = [
            make_fact("FIFO", "Same content.", "chunk-1"),
            make_fact("FIFO", "Same content.", "chunk-1"),
        ]
        result = fact_sources(facts)
        assert result["Same content."] == ["chunk-1"]

    def test_multiple_chunks_for_same_content(self):
        facts = [
            make_fact("FIFO", "Shared content.", "chunk-1"),
            make_fact("LIFO", "Shared content.", "chunk-2"),
        ]
        result = fact_sources(facts)
        assert set(result["Shared content."]) == {"chunk-1", "chunk-2"}

    def test_empty_content_skipped(self):
        facts = [make_fact("FIFO", "  ", "chunk-1")]
        result = fact_sources(facts)
        assert result == {}

    def test_multiple_distinct_facts(self):
        facts = [
            make_fact("FIFO", "Fact A.", "chunk-1"),
            make_fact("LIFO", "Fact B.", "chunk-2"),
        ]
        result = fact_sources(facts)
        assert "Fact A." in result
        assert "Fact B." in result


class TestBuildRelatedConcepts:
    def test_finds_related_by_token_overlap(self):
        concepts = ["Inventory Turnover", "Inventory Fraud", "Balance Sheet"]
        result = build_related_concepts(concepts)
        # "Inventory Turnover" and "Inventory Fraud" share the token "inventory"
        assert "Inventory Fraud" in result.get("Inventory Turnover", [])

    def test_excludes_self(self):
        concepts = ["FIFO", "LIFO", "Balance Sheet"]
        result = build_related_concepts(concepts)
        assert "FIFO" not in result.get("FIFO", [])

    def test_no_overlap_not_related(self):
        concepts = ["FIFO", "Balance Sheet"]
        result = build_related_concepts(concepts)
        # No token overlap
        assert result.get("FIFO", []) == []

    def test_respects_max_related(self):
        concepts = ["Inventory Cost", "Inventory Fraud", "Inventory Method", "Inventory System", "Balance Sheet"]
        result = build_related_concepts(concepts, max_related=2)
        for related_list in result.values():
            assert len(related_list) <= 2

    def test_ordered_by_overlap_descending(self):
        concepts = ["Inventory Turnover Ratio", "Inventory Turnover", "Inventory Fraud"]
        result = build_related_concepts(concepts)
        related = result.get("Inventory Turnover Ratio", [])
        if len(related) >= 2:
            # "Inventory Turnover" shares 2 tokens, "Inventory Fraud" shares 1
            assert related[0] == "Inventory Turnover"

    def test_excludes_siblings_by_default(self):
        concepts = ["Inventory Fraud", "Inventory Shrinkage", "Inventory Valuation"]
        result = build_related_concepts(concepts)
        assert "Inventory Shrinkage" not in result.get("Inventory Fraud", [])
        assert "Inventory Valuation" not in result.get("Inventory Fraud", [])

    def test_excludes_antonyms_by_default(self):
        concepts = ["First In First Out", "Last In First Out", "Cost of Goods Sold"]
        result = build_related_concepts(concepts)
        assert "Last In First Out" not in result.get("First In First Out", [])

    def test_excludes_internal_pipeline_concepts(self):
        concepts = [
            "Inventory Turnover Ratio",
            "Canonicalize Concept Names",
            "Inventory Management",
        ]
        result = build_related_concepts(concepts)
        assert "Canonicalize Concept Names" not in result
        assert "Canonicalize Concept Names" not in result.get("Inventory Turnover Ratio", [])


class TestBuildRelatedConceptsByChunks:
    def test_shared_chunks_weighted_higher(self):
        concept_chunks = {
            "FIFO": {"chunk-1", "chunk-2"},
            "LIFO": {"chunk-1"},
            "Balance Sheet": {"chunk-3"},
        }
        result = build_related_concepts_by_chunks("FIFO", concept_chunks, ["FIFO", "LIFO", "Balance Sheet"])
        assert "LIFO" in result
        assert result.index("LIFO") < (result.index("Balance Sheet") if "Balance Sheet" in result else len(result))

    def test_excludes_self(self):
        concept_chunks = {
            "FIFO": {"chunk-1"},
            "LIFO": {"chunk-1"},
        }
        result = build_related_concepts_by_chunks("FIFO", concept_chunks, ["FIFO", "LIFO"])
        assert "FIFO" not in result

    def test_token_overlap_fallback(self):
        concept_chunks: dict = {
            "Inventory Turnover": set(),
            "Inventory Fraud": set(),
        }
        result = build_related_concepts_by_chunks(
            "Inventory Turnover", concept_chunks,
            ["Inventory Turnover", "Inventory Fraud"]
        )
        assert "Inventory Fraud" in result

    def test_respects_max_related(self):
        concept_chunks = {
            "A": {"c1", "c2"},
            "B": {"c1"},
            "C": {"c2"},
            "D": {"c1", "c2"},
        }
        result = build_related_concepts_by_chunks("A", concept_chunks, list(concept_chunks.keys()), max_related=2)
        assert len(result) <= 2

    def test_no_relation_empty_result(self):
        concept_chunks = {
            "FIFO": {"chunk-1"},
            "Balance Sheet": {"chunk-99"},
        }
        result = build_related_concepts_by_chunks(
            "FIFO", concept_chunks, ["FIFO", "Balance Sheet"]
        )
        assert "Balance Sheet" not in result

    def test_excludes_siblings_and_antonyms_by_default(self):
        concept_chunks = {
            "First In First Out": {"chunk-1", "chunk-2"},
            "Last In First Out": {"chunk-1"},
            "Inventory Fraud": {"chunk-2"},
            "Inventory Shrinkage": {"chunk-2"},
        }
        grouped = {
            "First In First Out": [make_fact("FIFO", "A", "chunk-1"), make_fact("FIFO", "B", "chunk-2")],
            "Last In First Out": [make_fact("LIFO", "A", "chunk-1"), make_fact("LIFO", "B", "chunk-1")],
            "Inventory Fraud": [make_fact("Inventory Fraud", "A", "chunk-2"), make_fact("Inventory Fraud", "B", "chunk-2")],
            "Inventory Shrinkage": [make_fact("Inventory Shrinkage", "A", "chunk-2"), make_fact("Inventory Shrinkage", "B", "chunk-2")],
        }

        result = build_related_concepts_by_chunks(
            "First In First Out",
            concept_chunks,
            list(concept_chunks.keys()),
            grouped=grouped,
        )

        assert "Last In First Out" not in result

        fraud_related = build_related_concepts_by_chunks(
            "Inventory Fraud",
            concept_chunks,
            list(concept_chunks.keys()),
            grouped=grouped,
        )
        assert "Inventory Shrinkage" not in fraud_related

    def test_chunk_related_excludes_internal_concepts(self):
        concept_chunks = {
            "Inventory Turnover Ratio": {"chunk-1"},
            "Canonicalize Concept Names": {"chunk-1"},
            "Inventory Management": {"chunk-1"},
        }
        grouped = {
            "Inventory Turnover Ratio": [make_fact("Inventory Turnover Ratio", "A", "chunk-1")],
            "Canonicalize Concept Names": [make_fact("Canonicalize Concept Names", "B", "chunk-1")],
            "Inventory Management": [make_fact("Inventory Management", "C", "chunk-1")],
        }

        result = build_related_concepts_by_chunks(
            "Inventory Turnover Ratio",
            concept_chunks,
            list(concept_chunks.keys()),
            grouped=grouped,
        )
        assert "Canonicalize Concept Names" not in result


class TestCitationSuffixes:
    def test_appends_footnote_to_each_item(self):
        items = ["Fact A.", "Fact B."]
        text_to_sources = {"Fact A.": ["chunk-1"], "Fact B.": ["chunk-2"]}
        rendered, notes, next_idx = citation_suffixes(items, text_to_sources, {}, start_index=1)
        assert all("[^" in r for r in rendered)

    def test_reuses_index_for_same_source(self):
        items = ["Fact A.", "Fact B."]
        text_to_sources = {"Fact A.": ["chunk-1"], "Fact B.": ["chunk-1"]}
        rendered, notes, _ = citation_suffixes(items, text_to_sources, {}, start_index=1)
        # Same source key → same footnote index
        assert rendered[0].split("[^")[1] == rendered[1].split("[^")[1]

    def test_increments_index_for_new_sources(self):
        items = ["Fact A.", "Fact B."]
        text_to_sources = {"Fact A.": ["chunk-1"], "Fact B.": ["chunk-2"]}
        _, notes, next_idx = citation_suffixes(items, text_to_sources, {}, start_index=1)
        assert next_idx == 3  # started at 1, added 2 new sources

    def test_note_contains_chunk_id(self):
        items = ["Fact A."]
        text_to_sources = {"Fact A.": ["chunk-abc"]}
        _, notes, _ = citation_suffixes(items, text_to_sources, {}, start_index=1)
        assert any("chunk-abc" in note for note in notes)

    def test_empty_items_returns_empty(self):
        rendered, notes, next_idx = citation_suffixes([], {}, {}, start_index=1)
        assert rendered == []
        assert notes == []
        assert next_idx == 1

    def test_unknown_source_labeled(self):
        items = ["Orphan Fact."]
        text_to_sources = {}
        _, notes, _ = citation_suffixes(items, text_to_sources, {}, start_index=1)
        assert any("unknown" in note for note in notes)
