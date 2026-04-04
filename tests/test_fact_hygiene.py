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


def test_apply_fact_hygiene_keeps_worked_example_facts() -> None:
    # Worked examples are no longer treated as noise — they are routed to
    # a dedicated "Worked Example" section by the renderer.
    facts = [
        _fact("The gross margin, resulting from the weighted-average perpetual cost allocations, was $7,253.", concept="Gross Margin"),
        _fact("Because inventory values are wrong, the associated accounts are also wrong.", concept="Inventory"),
    ]

    cleaned, dropped = apply_fact_hygiene(facts)

    assert dropped == 0
    assert len(cleaned) == 2


def test_apply_fact_hygiene_keeps_balance_examples() -> None:
    # Numeric example facts (e.g. "had a balance of $3,150") are kept so the
    # renderer can place them in the "Worked Example" section.
    facts = [
        _fact("Beginning merchandise inventory had a balance of $3,150 before adjustment.", concept="Beginning Inventory"),
        _fact("Inventory is an asset reported on the balance sheet.", concept="Inventory"),
    ]

    cleaned, dropped = apply_fact_hygiene(facts)

    assert dropped == 0
    assert len(cleaned) == 2
