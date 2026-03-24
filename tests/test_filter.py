#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from transform.filter import is_valid_concept

test_cases = [
    # (concept, should_pass, reason)
    ("AES", True, "short acronym - valid concept"),
    ("European History", True, "basic concept - valid"),
    ("Balance Sheet", True, "short phrase - valid"),
    ("RSA Encryption", True, "2-word concept - valid"),
    
    # Should reject: years
    ("2024 Cryptography", False, "contains year"),
    ("SSL 2021 Update", False, "contains year"),
    
    # Should reject: possessive
    ("France's Encryption", False, "possessive proper noun"),
    ("Bob's Key", False, "possessive"),
    
    # Should reject: vague descriptors
    ("Example of Hashing", False, "starts with Example"),
    ("Case Study: Enron", False, "starts with Case"),
    ("Scenario Analysis", False, "starts with Scenario"),
    ("Impact on Security", False, "starts with Impact"),
    ("Effect of Policies", False, "starts with Effect"),
    
    # Should reject: countries
    ("France History", False, "contains country name"),
    ("Germany War", False, "contains country name"),
    ("United States RSA", False, "contains country name"),
    
    # Should reject: verbs (ing ending)
    ("Running Encryption", False, "contains verb 'Running'"),
    ("Testing Process", False, "contains verb 'Testing'"),
    ("Breaking System", False, "contains verb 'Breaking'"),
    
    # Should pass: common ing nouns
    ("String Operations", True, "String is common noun"),
    ("Testing Framework", True, "Framework is not a verb form"),
    ("Training Data", True, "Training is valid noun"),
    
    # Should reject: too long
    ("This is a very long concept that has too many words", False, "too many words"),
    ("One Two Three Four Five Six Seven", False, "more than 6 words"),
    
    # Should reject: U S pattern
    ("U S Encryption", False, "U S pattern - possessive proper noun"),
    ("U K Standards", False, "U K pattern"),
]

passed = 0
failed = 0

for concept, expected, reason in test_cases:
    result = is_valid_concept(concept)
    status = "✓" if result == expected else "✗"
    if result == expected:
        passed += 1
    else:
        failed += 1
    print(f"{status} {concept:40} -> {result!s:5} (expected {expected!s:5}) [{reason}]")

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
