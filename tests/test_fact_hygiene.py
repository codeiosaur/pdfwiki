"""Tests for transform.fact_hygiene."""

from extract.fact_extractor import Fact
from transform.fact_hygiene import apply_fact_hygiene


def _fact(content: str, concept: str = "Inventory") -> Fact:
    return Fact(
        id="f1",
        concept=concept,
        content=content,
        source_chunk_id="c1",
    )


def test_apply_fact_hygiene_drops_answer_key_artifacts() -> None:
    facts = [
        _fact("The correct answer is B. beginning inventory + purchases = cost of goods sold"),
        _fact("Cost of goods sold is an expense account used on the income statement."),
        _fact("Which of the following indicates a positive inventory trend?"),
    ]

    cleaned, dropped = apply_fact_hygiene(facts)

    assert dropped == 2
    assert len(cleaned) == 1
    assert "expense account" in cleaned[0].content.lower()


def test_apply_fact_hygiene_keeps_normal_fact() -> None:
    facts = [_fact("Inventory turnover ratio is calculated as cost of goods sold divided by average inventory.")]

    cleaned, dropped = apply_fact_hygiene(facts)

    assert dropped == 0
    assert len(cleaned) == 1
