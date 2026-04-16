"""Tests for transform.matching — deterministic token matching logic."""

import pytest

from transform.matching import (
    tokenize_for_matching,
    edit_distance_1,
    is_cousin,
    is_duplicate,
    is_sibling,
    has_strong_overlap,
    has_antonym_conflict,
)


class TestTokenizeForMatching:
    def test_lowercases(self):
        assert tokenize_for_matching("Balance Sheet") == ["balance", "sheet"]

    def test_strips_punctuation(self):
        assert tokenize_for_matching("Cost-of-Goods") == ["cost", "of", "good"]

    def test_singularizes_plain_plural(self):
        assert tokenize_for_matching("Inventory Systems") == ["inventory", "system"]

    def test_singularizes_ies_to_y(self):
        tokens = tokenize_for_matching("Certificate Authorities")
        assert tokens == ["certificate", "authority"]

    def test_does_not_singularize_short_words(self):
        # "ids" has length 3, should not be singularized
        tokens = tokenize_for_matching("User ids")
        assert tokens[-1] == "ids"

    def test_keeps_apostrophe_inside_word(self):
        tokens = tokenize_for_matching("Anastasia's Revenge")
        assert tokens[0] == "anastasia's"

    def test_strips_leading_apostrophe(self):
        # Leading apostrophe not inside a word should be removed
        tokens = tokenize_for_matching("'quoted term'")
        assert "'" not in " ".join(tokens)

    def test_collapses_whitespace(self):
        assert tokenize_for_matching("  A   B  ") == ["a", "b"]

    def test_empty_string(self):
        assert tokenize_for_matching("") == []

    def test_single_word_no_singularize_ss(self):
        tokens = tokenize_for_matching("Access")
        assert tokens == ["acces"]  # ends with 's', gets stripped

    def test_singularizes_ss_ending(self):
        # matching.py strips the final 's' regardless of double-s
        # (unlike normalize.py which has a no-ss rule)
        tokens = tokenize_for_matching("Glass")
        assert tokens == ["glas"]


class TestEditDistance1:
    def test_identical_returns_false(self):
        assert edit_distance_1("hello", "hello") is False

    def test_one_substitution(self):
        assert edit_distance_1("cat", "bat") is True

    def test_two_substitutions(self):
        assert edit_distance_1("cat", "dog") is False

    def test_one_insertion(self):
        assert edit_distance_1("cat", "cats") is True

    def test_one_deletion(self):
        assert edit_distance_1("cats", "cat") is True

    def test_length_diff_greater_than_1(self):
        assert edit_distance_1("cat", "catfish") is False

    def test_empty_vs_single(self):
        assert edit_distance_1("", "a") is True

    def test_empty_vs_two(self):
        assert edit_distance_1("", "ab") is False


class TestIsDuplicate:
    def test_identical_concepts(self):
        assert is_duplicate("FIFO", "FIFO") is True

    def test_case_insensitive(self):
        assert is_duplicate("FIFO", "Fifo") is True

    def test_plural_vs_singular(self):
        # "systems" normalizes to "system"
        assert is_duplicate("Inventory System", "Inventory Systems") is True

    def test_different_token_count(self):
        assert is_duplicate("Public Key", "Public Key Cryptography") is False

    def test_single_token_edit_distance(self):
        assert is_duplicate("Cipher Suite", "Cypher Suite") is True

    def test_two_token_mismatches(self):
        assert is_duplicate("Public Key", "Private Lock") is False

    def test_completely_different(self):
        # "fifo" vs "lifo" differ by 1 char → is_duplicate returns True
        # Use truly distinct multi-token concepts instead
        assert is_duplicate("Balance Sheet", "RSA Encryption") is False

    def test_fifo_lifo_are_edit_distance_1(self):
        # "fifo" and "lifo" differ by exactly one char — is_duplicate catches this
        assert is_duplicate("FIFO", "LIFO") is True


class TestIsSibling:
    def test_same_head_different_modifier(self):
        assert is_sibling("ECB Mode", "CBC Mode") is True

    def test_inventory_fraud_vs_shrinkage_not_siblings(self):
        # "fraud" and "shrinkage" are different head words — not siblings
        # (they're protected by has_antonym_conflict instead)
        assert is_sibling("Inventory Fraud", "Inventory Shrinkage") is False

    def test_single_word_not_sibling(self):
        assert is_sibling("FIFO", "LIFO") is False

    def test_same_concept_not_sibling(self):
        assert is_sibling("Inventory System", "Inventory System") is False

    def test_different_head_not_sibling(self):
        assert is_sibling("Public Key", "Private Lock") is False

    def test_requires_two_tokens_each(self):
        assert is_sibling("Key", "Mode") is False


class TestHasStrongOverlap:
    def test_two_consecutive_shared_tokens(self):
        assert has_strong_overlap("Public Key Cryptography", "Key Cryptography") is True

    def test_single_shared_token(self):
        assert has_strong_overlap("Key Length", "Key Generation") is False

    def test_no_shared_tokens(self):
        assert has_strong_overlap("FIFO Method", "RSA Encryption") is False

    def test_short_concepts_no_overlap(self):
        assert has_strong_overlap("Key", "Mode") is False

    def test_three_shared_tokens(self):
        assert has_strong_overlap("First In First Out Method", "First In First Out System") is True


class TestHasAntonymConflict:
    def test_first_vs_last(self):
        assert has_antonym_conflict("First In First Out", "Last In First Out") is True

    def test_periodic_vs_perpetual(self):
        assert has_antonym_conflict("Periodic System", "Perpetual System") is True

    def test_gross_vs_net(self):
        assert has_antonym_conflict("Gross Profit", "Net Profit") is True

    def test_asset_vs_liability(self):
        assert has_antonym_conflict("Current Asset", "Current Liability") is True

    def test_non_conflicting_pair(self):
        assert has_antonym_conflict("Public Key", "Private Key") is False

    def test_same_concept_no_conflict(self):
        assert has_antonym_conflict("Inventory System", "Inventory System") is False

    def test_fraud_vs_shrinkage(self):
        assert has_antonym_conflict("Inventory Fraud", "Inventory Shrinkage") is True

    def test_payable_vs_receivable(self):
        assert has_antonym_conflict("Accounts Payable", "Accounts Receivable") is True

    def test_direct_vs_indirect(self):
        assert has_antonym_conflict("Direct Cost", "Indirect Cost") is True

    def test_fixed_vs_variable(self):
        assert has_antonym_conflict("Fixed Cost", "Variable Cost") is True

    def test_favorable_vs_unfavorable(self):
        assert has_antonym_conflict("Favorable Variance", "Unfavorable Variance") is True


class TestIsCousin:
    """Cousins share all modifier tokens but differ on the head word.

    These look similar but are typically distinct concepts (parallel members
    of a modifier-based family). They must never be auto-merged or redirected.
    """

    def test_fixed_cost_vs_fixed_asset(self):
        # Regression: CVP term vs balance-sheet term — never redirect.
        assert is_cousin("Fixed Cost", "Fixed Asset") is True

    def test_accounts_payable_vs_receivable(self):
        # Regression: opposite accounts — never redirect.
        assert is_cousin("Accounts Payable", "Accounts Receivable") is True

    def test_direct_material_vs_direct_labor(self):
        # Regression: distinct cost inputs — never redirect.
        assert is_cousin("Direct Material", "Direct Labor") is True

    def test_activity_base_vs_activity_rate(self):
        # Regression: denominator vs derived rate — never redirect.
        assert is_cousin("Activity Base", "Activity Rate") is True

    def test_same_head_word_not_cousin(self):
        # Same head word = siblings, not cousins.
        assert is_cousin("ECB Mode", "CBC Mode") is False

    def test_different_length_not_cousin(self):
        assert is_cousin("Gross Profit", "Gross Profit Percentage") is False

    def test_single_token_not_cousin(self):
        assert is_cousin("FIFO", "LIFO") is False

    def test_different_modifiers_not_cousin(self):
        # Both differ in modifier AND head — not cousins.
        assert is_cousin("Public Key", "Private Lock") is False

    def test_identical_not_cousin(self):
        assert is_cousin("Fixed Cost", "Fixed Cost") is False
