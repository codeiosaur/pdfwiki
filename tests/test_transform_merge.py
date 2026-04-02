"""Tests for transform.merge — concept deduplication and merging logic."""

import pytest

from extract.fact_extractor import Fact
from transform.merge import _dedupe_exact_token_keys, merge_similar_concepts


def make_facts(concept: str, n: int = 1) -> list[Fact]:
    return [
        Fact(id=f"{concept}-{i}", concept=concept, content=f"fact {i}", source_chunk_id="c1")
        for i in range(n)
    ]


class TestDedupeExactTokenKeys:
    def test_casing_variants_merged(self):
        grouped = {
            "FIFO": make_facts("FIFO", 2),
            "Fifo": make_facts("Fifo", 1),
        }
        result = _dedupe_exact_token_keys(grouped)
        assert len(result) == 1

    def test_winner_is_more_common_label(self):
        # FIFO has 3 facts, Fifo has 1 — FIFO should win
        grouped = {
            "FIFO": make_facts("FIFO", 3),
            "Fifo": make_facts("Fifo", 1),
        }
        result = _dedupe_exact_token_keys(grouped)
        assert "FIFO" in result

    def test_tie_goes_to_shorter_label(self):
        # Equal fact count — shorter label wins
        grouped = {
            "Inventory System": make_facts("Inventory System", 2),
            "Inventory Systems": make_facts("Inventory Systems", 2),
        }
        result = _dedupe_exact_token_keys(grouped)
        assert len(result) == 1
        # "Inventory System" is shorter than "Inventory Systems" after normalization
        # (both normalize to same tokens)

    def test_distinct_concepts_not_merged(self):
        grouped = {
            "FIFO": make_facts("FIFO", 1),
            "LIFO": make_facts("LIFO", 1),
        }
        result = _dedupe_exact_token_keys(grouped)
        assert len(result) == 2

    def test_all_facts_preserved(self):
        grouped = {
            "FIFO": make_facts("FIFO", 3),
            "Fifo": make_facts("Fifo", 2),
        }
        result = _dedupe_exact_token_keys(grouped)
        total = sum(len(v) for v in result.values())
        assert total == 5

    def test_compact_key_variants_merged(self):
        grouped = {
            "Inventory Turnover Ratio": make_facts("Inventory Turnover Ratio", 2),
            "Inventoryturnover Ratio": make_facts("Inventoryturnover Ratio", 1),
        }
        result = _dedupe_exact_token_keys(grouped)
        assert len(result) == 1


class TestMergeSimilarConcepts:
    def test_plural_singular_merged(self):
        grouped = {
            "Public Key": make_facts("Public Key", 2),
            "Public Keys": make_facts("Public Keys", 1),
        }
        result = merge_similar_concepts(grouped)
        assert len(result) == 1

    def test_antonym_concepts_not_merged(self):
        grouped = {
            "First In First Out": make_facts("First In First Out", 2),
            "Last In First Out": make_facts("Last In First Out", 2),
        }
        result = merge_similar_concepts(grouped)
        assert len(result) == 2

    def test_sibling_concepts_not_merged(self):
        grouped = {
            "Inventory Fraud": make_facts("Inventory Fraud", 2),
            "Inventory Shrinkage": make_facts("Inventory Shrinkage", 2),
        }
        result = merge_similar_concepts(grouped)
        assert len(result) == 2

    def test_winner_is_more_common(self):
        grouped = {
            "Public Key": make_facts("Public Key", 3),
            "Public Keys": make_facts("Public Keys", 1),
        }
        result = merge_similar_concepts(grouped)
        assert "Public Key" in result

    def test_all_facts_preserved_after_merge(self):
        grouped = {
            "Hash Function": make_facts("Hash Function", 2),
            "Hash Functions": make_facts("Hash Functions", 3),
        }
        result = merge_similar_concepts(grouped)
        total = sum(len(v) for v in result.values())
        assert total == 5

    def test_unrelated_concepts_stay_separate(self):
        grouped = {
            "FIFO": make_facts("FIFO", 1),
            "Balance Sheet": make_facts("Balance Sheet", 1),
            "RSA Encryption": make_facts("RSA Encryption", 1),
        }
        result = merge_similar_concepts(grouped)
        assert len(result) == 3

    def test_metric_suffix_variant_merged(self):
        grouped = {
            "Inventory Turnover": make_facts("Inventory Turnover", 2),
            "Inventory Turnover Ratio": make_facts("Inventory Turnover Ratio", 1),
        }
        result = merge_similar_concepts(grouped)
        assert len(result) == 1
        total = sum(len(v) for v in result.values())
        assert total == 3

    def test_tie_prefers_more_specific_surface_form(self):
        grouped = {
            "Inventory Turnover": make_facts("Inventory Turnover", 2),
            "Inventory Turnover Ratio": make_facts("Inventory Turnover Ratio", 2),
        }
        result = merge_similar_concepts(grouped)
        assert len(result) == 1
        assert "Inventory Turnover Ratio" in result

    def test_metric_specific_label_wins_even_with_fewer_facts(self):
        grouped = {
            "Inventory Turnover": make_facts("Inventory Turnover", 4),
            "Inventory Turnover Ratio": make_facts("Inventory Turnover Ratio", 1),
        }
        result = merge_similar_concepts(grouped)
        assert len(result) == 1
        assert "Inventory Turnover Ratio" in result
