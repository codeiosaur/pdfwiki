"""Tests for transform.filter.is_valid_concept."""

import pytest

from transform.filter import is_valid_concept
from transform.filter import filter_publishable_grouped_concepts


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
