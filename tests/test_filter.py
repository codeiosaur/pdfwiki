"""Tests for transform.filter.is_valid_concept."""

import pytest

from transform.filter import is_valid_concept
from transform.filter import filter_publishable_grouped_concepts, filter_example_saturated_concepts
from extract.fact_extractor import Fact


@pytest.mark.parametrize("concept, expected", [
    # Valid concepts
    ("AES", True),
    ("European History", True),
    ("Balance Sheet", True),
    ("RSA Encryption", True),

    # Reject: years
    ("2024 Cryptography", False),
    ("SSL 2021 Update", False),

    # Reject: possessive
    ("France's Encryption", False),
    ("Bob's Key", False),

    # Reject: vague descriptors
    ("Example of Hashing", False),
    ("Case Study: Enron", False),
    ("Scenario Analysis", False),
    ("Impact on Security", False),
    ("Effect of Policies", False),

    # Reject: countries
    ("France History", False),
    ("Germany War", False),
    ("United States RSA", False),

    # Reject: verb-ing tokens (filter rejects common action verbs as first word)
    ("Running Encryption", False),
    ("Breaking System", False),

    # Allow: common -ing nouns
    ("String Operations", True),
    ("Testing Framework", True),
    ("Training Data", True),

    # Reject: too long
    ("This is a very long concept that has too many words", False),
    ("One Two Three Four Five Six Seven", False),

    # Reject: U S / U K patterns
    ("U S Encryption", False),
    ("U K Standards", False),

    # Reject: internal pipeline concept leakage
    ("Canonicalize Concept Names", False),
    ("Canonicalize Concept Name", False),
])
def test_is_valid_concept(concept, expected):
    assert is_valid_concept(concept) == expected


def test_filter_publishable_grouped_concepts_removes_internal_pages():
    grouped = {
        "Inventory Turnover Ratio": [],
        "Canonicalize Concept Names": [],
        "Balance Sheet": [],
    }

    result = filter_publishable_grouped_concepts(grouped)

    assert "Canonicalize Concept Names" not in result
    assert "Inventory Turnover Ratio" in result
    assert "Balance Sheet" in result


def test_filter_example_saturated_concepts_drops_example_only_groups():
    # Create fake Fact objects where most facts are examples
    f_example = Fact(id="f1", concept="X", content="The result was $1,234.", source_chunk_id="c1")
    f_example2 = Fact(id="f2", concept="X", content="After the adjustment, the balance was $2,345.", source_chunk_id="c1")
    f_good = Fact(id="f3", concept="Y", content="Inventory is reported on the balance sheet.", source_chunk_id="c2")

    grouped = {
        "Example Group": [f_example, f_example2],
        "Good Group": [f_good],
    }

    filtered, dropped = filter_example_saturated_concepts(grouped, threshold=0.5)

    assert "Example Group" not in filtered
    assert "Good Group" in filtered
    assert dropped == 1


def test_filter_example_saturated_concepts_keeps_mixed_groups_with_non_example_facts():
    f_example = Fact(id="f1", concept="Z", content="The result was $9,999.", source_chunk_id="c1")
    f_non_example = Fact(
        id="f2",
        concept="Z",
        content="Inventory turnover measures how quickly inventory is sold and replaced.",
        source_chunk_id="c1",
    )

    grouped = {"Mixed Group": [f_example, f_non_example]}

    filtered, dropped = filter_example_saturated_concepts(grouped, threshold=0.5)

    assert dropped == 0
    assert "Mixed Group" in filtered
    assert len(filtered["Mixed Group"]) == 1
    assert "turnover measures" in filtered["Mixed Group"][0].content.lower()
