"""Tests for transform.cluster — concept clustering logic."""

import pytest

from extract.fact_extractor import Fact
from transform.cluster import find_head_word, is_clusterable, cluster_related_concepts


def make_facts(concept: str, n: int = 1, chunk_id: str = "c1") -> list[Fact]:
    return [
        Fact(id=f"{concept}-{i}", concept=concept, content=f"fact {i}", source_chunk_id=chunk_id)
        for i in range(n)
    ]


class TestFindHeadWord:
    def test_single_word(self):
        assert find_head_word("FIFO") == "fifo"

    def test_multi_word_returns_last(self):
        assert find_head_word("Inventory System") == "system"

    def test_plural_last_token_singularized(self):
        assert find_head_word("Inventory Systems") == "system"

    def test_empty_string(self):
        assert find_head_word("") == ""


class TestIsClusterable:
    def test_near_duplicate_tokens_clusterable(self):
        # "FIFO Method" and "FIFO Methods" normalize to the same tokens
        # → not siblings (identical prefix), same head → clusterable
        assert is_clusterable("FIFO Method", "FIFO Methods") is True

    def test_perpetual_periodic_not_clusterable_due_to_antonym(self):
        # periodic/perpetual is an antonym pair — antonym conflict prevents clustering
        assert is_clusterable("Perpetual Inventory System", "Periodic Inventory System") is False

    def test_sibling_not_clusterable(self):
        # Same head (last token) but different modifiers = sibling, not clusterable
        assert is_clusterable("Inventory Fraud", "Inventory Valuation") is False

    def test_antonym_conflict_not_clusterable(self):
        assert is_clusterable("First In First Out", "Last In First Out") is False

    def test_different_head_words_not_clusterable(self):
        assert is_clusterable("Public Key", "Private Lock") is False

    def test_single_word_concepts_not_clusterable(self):
        assert is_clusterable("FIFO", "LIFO") is False

    def test_distinct_acronym_prefixes_not_clusterable(self):
        # ECB and CBC are distinct uppercase prefixes
        assert is_clusterable("ECB Mode", "CBC Mode") is False


class TestClusterRelatedConcepts:
    def test_singleton_unchanged(self):
        grouped = {"FIFO": make_facts("FIFO", 2)}
        result = cluster_related_concepts(grouped)
        assert "FIFO" in result
        assert len(result["FIFO"]) == 2

    def test_multiple_singletons_all_preserved(self):
        grouped = {
            "FIFO": make_facts("FIFO", 1),
            "Balance Sheet": make_facts("Balance Sheet", 1),
        }
        result = cluster_related_concepts(grouped)
        assert len(result) == 2

    def test_near_duplicate_concepts_merged(self):
        # "FIFO Method" and "FIFO Methods" normalize to the same token sequence
        grouped = {
            "FIFO Method": make_facts("FIFO Method", 2),
            "FIFO Methods": make_facts("FIFO Methods", 1),
        }
        result = cluster_related_concepts(grouped)
        assert len(result) == 1
        total = sum(len(v) for v in result.values())
        assert total == 3

    def test_antonym_pairs_not_merged(self):
        grouped = {
            "First In First Out": make_facts("First In First Out", 2),
            "Last In First Out": make_facts("Last In First Out", 2),
        }
        result = cluster_related_concepts(grouped)
        assert len(result) == 2

    def test_sibling_concepts_not_merged(self):
        grouped = {
            "Inventory Fraud": make_facts("Inventory Fraud", 1),
            "Inventory Shrinkage": make_facts("Inventory Shrinkage", 1),
        }
        result = cluster_related_concepts(grouped)
        assert len(result) == 2

    def test_all_facts_preserved(self):
        grouped = {
            "FIFO Method": make_facts("FIFO Method", 3),
            "FIFO Methods": make_facts("FIFO Methods", 2),
        }
        result = cluster_related_concepts(grouped)
        total = sum(len(v) for v in result.values())
        assert total == 5
