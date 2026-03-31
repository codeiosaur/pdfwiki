from typing import List
from extract.fact_extractor import Fact

import re

def normalize_concept(concept: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", concept.strip())
    raw_tokens = cleaned.split()

    normalized_tokens: List[str] = []
    for token in raw_tokens:
        if token.isalpha() and token.isupper() and len(token) <= 6:
            normalized_tokens.append(token)
        else:
            normalized_tokens.append(token.lower().title())

    return " ".join(normalized_tokens)

def group_facts_by_concept(facts: List[Fact]) -> dict[str, List[Fact]]:
    grouped: dict[str, List[Fact]] = {}
    for fact in facts:
        concept_key = normalize_concept(fact.concept)
        grouped.setdefault(concept_key, []).append(fact)
    return grouped