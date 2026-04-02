"""Tests for transform.filter.is_valid_concept."""

import pytest

from transform.filter import is_valid_concept


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
])
def test_is_valid_concept(concept, expected):
    assert is_valid_concept(concept) == expected
