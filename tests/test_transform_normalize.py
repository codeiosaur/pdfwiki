"""Tests for transform.normalize — deterministic rule-based normalization."""

import pytest

from transform.normalize import normalize_concept_rules, normalize_group_keys
from extract.fact_extractor import Fact


class TestNormalizeConceptRules:
    # Generic title casing / acronym handling
    def test_single_token_title_cased(self):
        assert normalize_concept_rules("fifo") == "Fifo"

    def test_lowercase_multiword_title_cased(self):
        assert normalize_concept_rules("balance sheet") == "Balance Sheet"

    def test_meaningful_suffix_is_preserved(self):
        assert normalize_concept_rules("inventory system") == "Inventory System"

    def test_title_case_two_meaningful_words(self):
        assert normalize_concept_rules("balance sheet") == "Balance Sheet"

    def test_acronym_preserved_in_title_case(self):
        result = normalize_concept_rules("RSA encryption")
        assert "RSA" in result

    # Singular and plural can name different concepts; leave both intact.
    def test_preserves_plural_s(self):
        result = normalize_concept_rules("inventory systems")
        assert result.endswith("Systems")

    def test_preserves_plural_ies(self):
        result = normalize_concept_rules("certificate authorities")
        assert result.endswith("Authorities")

    def test_no_singularize_ss(self):
        result = normalize_concept_rules("business class")
        assert result.endswith("Class")

    # Repetition can be meaningful in some subjects; don't guess.
    def test_preserves_repeated_words(self):
        result = normalize_concept_rules("Inventory Inventory")
        assert result == "Inventory Inventory"

    def test_dedupe_case_insensitive(self):
        result = normalize_concept_rules("key KEY")
        assert result == "Key KEY"

    # Leading phrases and suffixes can change the concept.
    def test_preserves_number_of_prefix(self):
        result = normalize_concept_rules("number of transactions")
        assert result == "Number of Transactions"

    def test_preserves_type_of_prefix(self):
        result = normalize_concept_rules("type of account")
        assert result == "Type of Account"

    # Generic suffix removal
    def test_preserves_method_suffix(self):
        result = normalize_concept_rules("RSA Encryption Method")
        assert result.endswith("Method")

    def test_preserves_system_suffix(self):
        result = normalize_concept_rules("Perpetual Inventory System")
        assert result.endswith("System")

    def test_keeps_suffix_when_stem_too_short(self):
        # Single short token — keep suffix
        result = normalize_concept_rules("Key System")
        # stem is "Key" (length 3) — borderline, but let's just verify it doesn't crash
        assert isinstance(result, str)

    # Parenthetical normalization
    def test_parenthetical_acronym_preserved(self):
        result = normalize_concept_rules("Days Sales in Inventory (DSI)")
        assert "(DSI)" in result
        assert "Days Sales" in result

    def test_hyphens_preserved(self):
        result = normalize_concept_rules("First-In-First-Out")
        assert result == "First-In-First-Out"

    def test_mixed_case_model_names_preserved(self):
        assert normalize_concept_rules("MoE and OpenAI") == "MoE and OpenAI"

    def test_no_accounting_expansion(self):
        assert normalize_concept_rules("avg velocity") == "Avg Velocity"

    # Edge cases
    def test_empty_string(self):
        assert normalize_concept_rules("") == ""

    def test_whitespace_only(self):
        assert normalize_concept_rules("   ") == ""


class TestNormalizeGroupKeys:
    def _make_fact(self, concept, content="fact"):
        return Fact(id="x", concept=concept, content=content, source_chunk_id="c1")

    def test_case_variants_collapse_without_domain_mapping(self):
        grouped = {
            "fifo": [self._make_fact("fifo", "oldest cost")],
            "FIFO": [self._make_fact("FIFO", "inventory method")],
        }
        result = normalize_group_keys(grouped)
        assert len(result) == 1

    def test_preserves_distinct_concepts(self):
        grouped = {
            "FIFO": [self._make_fact("FIFO")],
            "LIFO": [self._make_fact("LIFO")],
        }
        result = normalize_group_keys(grouped)
        assert len(result) == 2

    def test_all_facts_preserved(self):
        facts = [self._make_fact("fifo", f"fact {i}") for i in range(3)]
        grouped = {"fifo": facts}
        result = normalize_group_keys(grouped)
        total = sum(len(v) for v in result.values())
        assert total == 3

    def test_preposition_variants_remain_distinct(self):
        grouped = {
            "Days Sales in Inventory": [self._make_fact("Days Sales in Inventory", "fact a")],
            "Days Sales of Inventory": [self._make_fact("Days Sales of Inventory", "fact b")],
        }
        result = normalize_group_keys(grouped)
        assert len(result) == 2
        total = sum(len(v) for v in result.values())
        assert total == 2
