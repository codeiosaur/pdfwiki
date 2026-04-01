"""Tests for generate.wiki_helpers — concept typing, wikilink injection, and fact promotion."""

import pytest

from generate.wiki_helpers import (
    classify_concept_type,
    infer_concept_type_from_facts,
    inject_wikilinks,
    promote_all_facts_to_content,
)


class TestClassifyConceptType:
    def test_ratio_keyword_in_name(self):
        assert classify_concept_type("Inventory Turnover Ratio") == "ratio"

    def test_rate_keyword_in_name(self):
        assert classify_concept_type("Interest Rate") == "ratio"

    def test_method_keyword_in_name(self):
        assert classify_concept_type("FIFO Method") == "method"

    def test_system_keyword_in_name(self):
        assert classify_concept_type("Perpetual Inventory System") == "system"

    def test_general_when_no_keyword(self):
        assert classify_concept_type("Balance Sheet") == "general"

    def test_general_no_facts(self):
        assert classify_concept_type("Unknown Concept") == "general"

    def test_infers_ratio_from_facts(self):
        facts = ["Turnover = COGS / Average Inventory"]
        result = classify_concept_type("Inventory Turnover", fact_contents=facts)
        assert result == "ratio"

    def test_infers_method_from_facts(self):
        facts = [
            "This method assigns the oldest costs to COGS.",
            "It allocates costs based on purchase order.",
        ]
        result = classify_concept_type("Cost Flow Assumption", fact_contents=facts)
        assert result == "method"


class TestInferConceptTypeFromFacts:
    def test_formula_signal_returns_ratio(self):
        facts = ["Inventory Turnover = COGS / Average Inventory"]
        assert infer_concept_type_from_facts(facts) == "ratio"

    def test_divided_by_signal_returns_ratio(self):
        # Needs "ratio" keyword + "divided by" (exact phrase, not "dividing")
        facts = [
            "The ratio is net sales divided by average assets.",
            "It measures how efficiently a company uses assets.",
        ]
        assert infer_concept_type_from_facts(facts) == "ratio"

    def test_method_signals_return_method(self):
        facts = [
            "This method records costs at the time of purchase.",
            "It allocates expenses based on units sold.",
        ]
        assert infer_concept_type_from_facts(facts) == "method"

    def test_system_signals_return_system(self):
        facts = [
            "The system updates inventory records continuously.",
            "It tracks quantities at the time of each sale.",
        ]
        assert infer_concept_type_from_facts(facts) == "system"

    def test_weak_signals_return_none(self):
        facts = ["This concept is important for financial reporting."]
        assert infer_concept_type_from_facts(facts) is None

    def test_empty_facts_return_none(self):
        assert infer_concept_type_from_facts([]) is None


class TestInjectWikilinks:
    def test_replaces_first_occurrence(self):
        text = "FIFO is used. FIFO was introduced in the 1920s."
        result = inject_wikilinks(text, {"FIFO"}, "Current Page")
        assert "[[FIFO]]" in result

    def test_only_first_occurrence_linked(self):
        text = "FIFO is used. FIFO was introduced in the 1920s."
        result = inject_wikilinks(text, {"FIFO"}, "Current Page")
        # Should have exactly one [[FIFO]] link
        assert result.count("[[FIFO]]") == 1

    def test_does_not_link_current_title(self):
        text = "Balance Sheet is a financial statement."
        result = inject_wikilinks(text, {"Balance Sheet"}, "Balance Sheet")
        assert "[[Balance Sheet]]" not in result

    def test_does_not_nest_existing_wikilinks(self):
        text = "See [[FIFO]] for details. FIFO is common."
        result = inject_wikilinks(text, {"FIFO"}, "Current Page")
        # The existing [[FIFO]] should not become [[[[FIFO]]]]
        assert "[[[[" not in result

    def test_empty_titles_no_change(self):
        text = "Some text here."
        result = inject_wikilinks(text, set(), "Current Page")
        assert result == text

    def test_longer_titles_matched_first(self):
        # "Balance Sheet" should match before "Balance" if both are in the set
        text = "The Balance Sheet shows assets."
        result = inject_wikilinks(text, {"Balance Sheet", "Balance"}, "Current Page")
        assert "[[Balance Sheet]]" in result

    def test_case_insensitive_match(self):
        text = "The fifo method is used here."
        result = inject_wikilinks(text, {"FIFO"}, "Current Page")
        assert "[[FIFO]]" in result

    # --- alias_map tests ---

    def test_alias_links_acronym_to_full_title(self):
        """An acronym alias should link to its canonical title even when the full title is absent."""
        text = "FIFO assigns older costs to expenses first."
        alias_map = {"fifo": "First In First Out"}
        result = inject_wikilinks(text, {"First In First Out"}, "Current Page", alias_map=alias_map)
        assert "[[First In First Out]]" in result

    def test_alias_match_is_case_insensitive(self):
        text = "Under fifo, the oldest costs are used."
        alias_map = {"fifo": "First In First Out"}
        result = inject_wikilinks(text, {"First In First Out"}, "Current Page", alias_map=alias_map)
        assert "[[First In First Out]]" in result

    def test_alias_does_not_link_current_page(self):
        """An alias pointing to the current page must not create a self-link."""
        text = "FIFO assigns the oldest costs."
        alias_map = {"fifo": "First In First Out"}
        result = inject_wikilinks(text, set(), "First In First Out", alias_map=alias_map)
        assert "[[" not in result

    def test_alias_linked_only_once(self):
        """A canonical title reached via alias should be linked at most once."""
        text = "FIFO is a method. FIFO reduces ending inventory cost."
        alias_map = {"fifo": "First In First Out"}
        result = inject_wikilinks(text, {"First In First Out"}, "Current Page", alias_map=alias_map)
        assert result.count("[[First In First Out]]") == 1

    def test_alias_exact_title_not_doubled_after_alias(self):
        """Once an alias injects the canonical title, the exact-title pass must not add it again."""
        text = "FIFO is used. First In First Out was introduced in the 1920s."
        alias_map = {"fifo": "First In First Out"}
        result = inject_wikilinks(text, {"First In First Out"}, "Current Page", alias_map=alias_map)
        assert result.count("[[First In First Out]]") == 1

    def test_no_alias_map_behaviour_unchanged(self):
        """Calling without alias_map must produce the same output as before."""
        text = "Balance Sheet is a financial statement."
        result = inject_wikilinks(text, {"Balance Sheet"}, "Current Page")
        assert "[[Balance Sheet]]" in result


class TestPromoteAllFactsToContent:
    def test_excludes_definition(self):
        facts = [
            "FIFO is a method for inventory valuation.",
            "Under FIFO, oldest costs are used first.",
        ]
        definition = "FIFO is a method for inventory valuation."
        result = promote_all_facts_to_content(facts, definition)
        assert "FIFO is a method for inventory valuation." not in result

    def test_includes_non_definition_facts(self):
        facts = [
            "FIFO is a method.",
            "Under FIFO, oldest costs are used first.",
            "FIFO often results in higher net income during inflation.",
        ]
        result = promote_all_facts_to_content(facts, "FIFO is a method.")
        assert "Under FIFO, oldest costs are used first." in result

    def test_excludes_instruction_facts(self):
        facts = [
            "FIFO is a method.",
            "Calculate ending inventory using the FIFO approach.",
        ]
        result = promote_all_facts_to_content(facts, "FIFO is a method.")
        assert not any("Calculate" in f for f in result)

    def test_excludes_low_signal_facts(self):
        # Less than 4 words is low-signal
        facts = [
            "FIFO is a method.",
            "See figure.",
        ]
        result = promote_all_facts_to_content(facts, "FIFO is a method.")
        assert not any(len(f.split()) < 4 for f in result)

    def test_excludes_template_marker_facts(self):
        facts = [
            "FIFO is a method.",
            "... units at $5.00 each were sold.",
        ]
        result = promote_all_facts_to_content(facts, "FIFO is a method.")
        assert not any("$..." in f for f in result)

    def test_empty_fact_list(self):
        result = promote_all_facts_to_content([], "definition")
        assert result == []
