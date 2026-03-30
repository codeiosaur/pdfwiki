"""Tests for generate.titles — acronym maps, title casing, and page title normalization."""

import pytest

import generate.titles as titles_module
from generate.titles import build_acronym_map, normalize_page_title, concept_tokens


@pytest.fixture(autouse=True)
def clear_acronym_canonical():
    """Reset the global ACRONYM_CANONICAL dict before each test."""
    titles_module.ACRONYM_CANONICAL.clear()
    yield
    titles_module.ACRONYM_CANONICAL.clear()


class TestBuildAcronymMap:
    def test_extracts_uppercase_token(self):
        result = build_acronym_map(["FIFO Method"])
        assert "fifo" in result
        assert result["fifo"] == "FIFO"

    def test_ignores_lowercase_tokens(self):
        result = build_acronym_map(["Balance Sheet"])
        assert "balance" not in result
        assert "sheet" not in result

    def test_multiple_concepts(self):
        result = build_acronym_map(["FIFO Method", "LIFO Method", "Balance Sheet"])
        assert "fifo" in result
        assert "lifo" in result

    def test_only_2_to_6_char_uppercase(self):
        # "A" is too short (1 char), "TOOLONG" is too long (7 chars)
        result = build_acronym_map(["A Token TOOLONG"])
        assert "a" not in result
        assert "toolong" not in result

    def test_empty_list(self):
        assert build_acronym_map([]) == {}

    def test_mixed_case_token_not_included(self):
        # "Fifo" is not all-caps
        result = build_acronym_map(["Fifo Method"])
        assert "fifo" not in result


class TestNormalizePageTitle:
    def test_acronym_tail_added_to_parens(self):
        assert normalize_page_title("Days Sales In Inventory DSI") == "Days Sales in Inventory (DSI)"

    def test_fob_acronym(self):
        result = normalize_page_title("Free On Board FOB")
        assert result == "Free on Board (FOB)"

    def test_no_acronym_tail_unchanged(self):
        result = normalize_page_title("Balance Sheet")
        assert result == "Balance Sheet"

    def test_single_word_unchanged(self):
        result = normalize_page_title("FIFO")
        assert result == "FIFO"

    def test_connector_words_lowercased(self):
        result = normalize_page_title("Cost Of Goods Sold")
        assert "of" in result.lower()
        # "of" should be lowercase after first word
        assert "Of" not in result or result.startswith("Of")

    def test_empty_string(self):
        result = normalize_page_title("")
        assert result == ""

    def test_no_false_acronym_from_non_matching_tail(self):
        # "ABCD" is not the initials of "Balance Sheet"
        result = normalize_page_title("Balance Sheet ABCD")
        # Should NOT add parens because ABCD doesn't match initials of "Balance Sheet"
        # (initials would be "BS")
        assert "(ABCD)" not in result


class TestConceptTokens:
    def test_returns_meaningful_tokens(self):
        result = concept_tokens("Balance Sheet")
        assert "balance" in result
        assert "sheet" in result

    def test_excludes_stopwords(self):
        result = concept_tokens("Cost of Goods Sold")
        assert "of" not in result
        assert "cost" in result
        assert "goods" in result
        assert "sold" in result

    def test_excludes_single_char_tokens(self):
        result = concept_tokens("A Balance Sheet")
        assert "a" not in result

    def test_lowercased(self):
        result = concept_tokens("FIFO Method")
        assert "fifo" in result or "FIFO" not in result

    def test_empty_concept(self):
        assert concept_tokens("") == set()
