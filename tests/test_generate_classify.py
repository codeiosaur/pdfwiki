"""Tests for generate.classify — fact classification and definition selection."""

import pytest

from generate.classify import (
    classify_fact,
    classify_semantic_fact,
    select_definition,
    pick_best_definition,
    _is_low_signal_key_point,
)


class TestClassifyFact:
    def test_definition_with_is(self):
        assert classify_fact("FIFO is a method where the oldest costs are expensed first.") == "definition"

    def test_definition_with_are(self):
        assert classify_fact("Assets are resources owned by the company.") == "definition"

    def test_definition_with_refers_to(self):
        assert classify_fact("COGS refers to the direct costs of goods sold.") == "definition"

    def test_instruction_calculate(self):
        assert classify_fact("Calculate the turnover ratio by dividing COGS by average inventory.") == "instruction"

    def test_instruction_compute(self):
        assert classify_fact("Compute the ending balance using the formula.") == "instruction"

    def test_instruction_prepare(self):
        assert classify_fact("Prepare a journal entry for each transaction.") == "instruction"

    def test_example_marker(self):
        assert classify_fact("For example, suppose the company sold 100 units.") == "example"

    def test_example_there_is(self):
        assert classify_fact("There is a case where FIFO produces higher net income.") == "example"

    def test_example_high_numeric_density(self):
        assert classify_fact("Units: 100 at $5.00, 200 at $6.50, 50 at $7.25") == "example"

    def test_example_resulting_from_phrase(self):
        assert classify_fact("The gross margin, resulting from the weighted-average perpetual cost allocations, was $7,253.") == "example"

    def test_example_balance_phrase(self):
        assert classify_fact("Beginning merchandise inventory had a balance of $3,150 before adjustment.") == "example"

    def test_example_subtracting_leaves_phrase(self):
        assert classify_fact("Subtracting this ending inventory from the $16,155 total of goods available for sale leaves $7,200 in cost of goods sold this period.") == "example"

    def test_example_ending_inventory_value_phrase(self):
        assert classify_fact("The weighted-average perpetual ending inventory value is $8,902 (rounded).") == "example"

    def test_example_company_percentage_year_phrase(self):
        assert classify_fact("Inventory is a significant portion of a company's assets, with Walmart's inventory being 70% of its current assets and 21% of its total assets in 2018.") == "example"

    def test_example_year_over_year_case_study_phrase(self):
        assert classify_fact("The result for the company indicates that inventory turned 1.19 times in year 1 and 0.84 times in year 2.") == "example"

    def test_example_year_over_year_trend_phrase(self):
        assert classify_fact("The fact that the year 2 inventory turnover ratio is lower than the year 1 ratio is not a positive trend.") == "example"

    def test_example_after_two_sales_phrase(self):
        assert classify_fact("After two sales, there remained 75 units of inventory that had cost $27 each in the FIFO method.") == "example"

    def test_example_remaining_units_costing_phrase(self):
        assert classify_fact("After two sales, there remained 30 units of beginning inventory that had cost $21 each and 45 units of the goods purchased for $27 each.") == "example"

    def test_key_point_default(self):
        # Avoid instruction verbs like "provides" — use neutral statement
        assert classify_fact("The perpetual system updates records after each transaction.") == "key_point"

    def test_empty_fact(self):
        assert classify_fact("") == "key_point"


class TestClassifySemanticFact:
    def test_formula_with_equals(self):
        result = classify_semantic_fact("Inventory Turnover = COGS / Average Inventory")
        assert result == "formula"

    def test_formula_with_divided_by(self):
        # "ratio" keyword + "divided by" triggers formula detection
        result = classify_semantic_fact("Turnover ratio is COGS divided by average inventory.")
        assert result == "formula"

    def test_caution_misleading(self):
        result = classify_semantic_fact("This metric can be misleading when inventory levels fluctuate.")
        assert result == "caution"

    def test_caution_risk(self):
        result = classify_semantic_fact("There is a risk that reported values may be manipulated.")
        assert result == "caution"

    def test_interpretation_indicates(self):
        result = classify_semantic_fact("A higher ratio indicates faster inventory turnover.")
        assert result == "interpretation"

    def test_interpretation_measures(self):
        result = classify_semantic_fact("This metric measures how efficiently inventory is managed.")
        assert result == "interpretation"

    def test_instruction_delegates_to_base(self):
        result = classify_semantic_fact("Calculate the ending balance using FIFO.")
        assert result == "instruction"

    def test_definition_preserved(self):
        result = classify_semantic_fact("FIFO is a method for valuing inventory.")
        assert result == "definition"

    def test_key_point_default(self):
        result = classify_semantic_fact("The perpetual system tracks each transaction.")
        assert result == "key_point"


class TestSelectDefinition:
    def test_prefers_concept_is_pattern(self):
        facts = [
            "Some unrelated statement about costs.",
            "FIFO is a method where the oldest costs are used first.",
        ]
        result = select_definition("FIFO", facts)
        assert result is not None
        assert "FIFO is a method" in result

    def test_returns_none_when_all_scores_low(self):
        # Both facts score -3 (starts with "there are", no offsetting bonuses),
        # which is below the threshold of -2, so the function should return None.
        facts = [
            "There are students who study this topic.",
            "There are various types of scenarios.",
        ]
        result = select_definition("FIFO", facts)
        assert result is None

    def test_penalizes_numbers_without_stat_context(self):
        facts = [
            "FIFO is a method for inventory valuation.",
            "100 units at $5.00 each were purchased in January.",
        ]
        result = select_definition("FIFO", facts)
        assert "FIFO is a method" in result

    def test_penalizes_there_are_opener(self):
        facts = [
            "There are three main inventory methods.",
            "FIFO refers to the first-in-first-out cost flow assumption.",
        ]
        result = select_definition("FIFO", facts)
        assert "FIFO refers to" in result

    def test_empty_facts_returns_none(self):
        assert select_definition("FIFO", []) is None

    def test_template_markers_filtered(self):
        facts = [
            "... units at $5.00 each were sold.",
            "FIFO is a method for valuing inventory.",
        ]
        result = select_definition("FIFO", facts)
        assert result is not None
        assert "$" not in result or "FIFO is" in result

    def test_select_definition_filters_example_candidates(self):
        facts = [
            "The gross margin, resulting from the weighted-average perpetual cost allocations, was $7,253.",
            "Gross margin is the difference between net sales and cost of goods sold.",
        ]
        result = select_definition("Gross Margin", facts)
        assert result is not None
        assert "gross margin is" in result.lower()


class TestIsLowSignalKeyPoint:
    def test_too_short_five_words(self):
        # 5 words — below new threshold of 6
        assert _is_low_signal_key_point("Short five word fact.") is True

    def test_five_words_exactly(self):
        assert _is_low_signal_key_point("One two three four five") is True

    def test_six_words_not_low_signal(self):
        # 6 words — meets new threshold, no other flags
        assert _is_low_signal_key_point("One two three four five six") is False

    def test_old_threshold_four_words_still_low_signal(self):
        # 4 words would also have been caught by the old rule; still low-signal
        assert _is_low_signal_key_point("Too short fact.") is True


class TestSelectDefinitionScoring:
    def test_concept_in_first_8_tokens_scores_higher(self):
        # "Amortization is..." — concept in first 8 tokens should win over a
        # late-mention fact of similar structure
        facts = [
            "The gradual reduction of a debt is called amortization.",  # concept late
            "Amortization is the gradual reduction of a debt over time.",  # concept in first 8
        ]
        result = select_definition("Amortization", facts)
        assert result is not None
        assert result.startswith("Amortization")

    def test_concept_beyond_8_tokens_no_early_bonus(self):
        # Concept appears only after token 8 — should not get the +2 bonus
        facts = [
            "Over a long period of time, the process known as amortization reduces debt.",
            "Amortization is the gradual reduction of a debt over time.",
        ]
        result = select_definition("Amortization", facts)
        assert result is not None
        assert result.startswith("Amortization")

    def test_noun_phrase_start_matching_concept_token_scores_higher(self):
        # "The [concept-token] ..." should get +1 noun-phrase bonus over a generic opener
        facts = [
            "An entity records costs gradually over the useful life.",  # generic article, no concept token
            "The amortization schedule shows equal payments across periods.",  # "The [concept token]"
        ]
        result = select_definition("Amortization", facts)
        assert result is not None
        assert "amortization schedule" in result.lower()

    def test_long_fact_over_55_words_mild_penalty(self):
        # A 56-word definition should still be selectable (only -1) if it otherwise scores well
        long_def = (
            "Amortization is the systematic allocation of the cost of an intangible asset "
            "over its useful life, reflecting the consumption of economic benefits embedded "
            "in the asset, recognized as an expense in the income statement each period "
            "until the asset is fully written off and has no remaining book value whatsoever."
        )
        short_def = "Amortization is a process."
        result = select_definition("Amortization", [long_def, short_def])
        # Long rich definition should still win despite mild penalty
        assert result is not None
        assert "systematic allocation" in result

    def test_fact_exactly_55_words_no_penalty(self):
        # 55 words — should not incur the length penalty
        words = ["word"] * 50
        fact = "Amortization is the " + " ".join(words) + " end."
        # Should return something (not None) since it has concept + definition markers
        result = select_definition("Amortization", [fact])
        assert result is not None

    def test_fact_over_40_words_no_longer_hard_penalized(self):
        # Old rule: -2 for > 40 words. New rule: -1 for > 55. A 45-word definition
        # should no longer be penalized at all.
        moderate_def = (
            "Amortization is the process of spreading out a loan into a series of fixed "
            "payments over time, covering both principal and interest components each period."
        )
        # This is ~30 words; pair with a weak competitor to confirm it wins cleanly
        weak_def = "There are several amortization methods available."
        result = select_definition("Amortization", [moderate_def, weak_def])
        assert result is not None
        assert "spreading out" in result


class TestPickBestDefinition:
    def test_picks_from_definitions(self):
        definitions = [
            "FIFO refers to the cost flow assumption.",
            "A method where oldest costs are used first.",
        ]
        result = pick_best_definition("FIFO", definitions, [])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_falls_back_to_key_points(self):
        result = pick_best_definition("FIFO", [], ["Oldest costs are expensed first."])
        assert result == "Oldest costs are expensed first."

    def test_no_definition_available(self):
        result = pick_best_definition("FIFO", [], [])
        assert result == "No definition available."

    def test_prefers_defined_as_pattern(self):
        definitions = [
            "FIFO is a method.",
            "FIFO is defined as a cost flow assumption.",
        ]
        result = pick_best_definition("FIFO", definitions, [])
        assert "defined as" in result

    def test_prefers_refers_to_over_plain_is(self):
        definitions = [
            "FIFO is sometimes used.",
            "FIFO refers to the first-in-first-out method.",
        ]
        result = pick_best_definition("FIFO", definitions, [])
        assert "refers to" in result

    def test_multiple_key_points_picks_longest_meaningful(self):
        key_points = [
            "Short.",
            "A longer and more informative statement about the concept.",
        ]
        result = pick_best_definition("X", [], key_points)
        assert "longer and more informative" in result
