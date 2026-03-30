"""Tests for generate.classify — fact classification and definition selection."""

import pytest

from generate.classify import (
    classify_fact,
    classify_semantic_fact,
    select_definition,
    pick_best_definition,
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
        facts = [
            "There are many methods available.",
            "Calculate using the following formula.",
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
