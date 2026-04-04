"""Fact-level hygiene filters applied before concept grouping.

This stage removes extraction artifacts (for example quiz answer stems)
that degrade page quality when passed through to rendering.
"""

from __future__ import annotations

from typing import Iterable
import re

from extract.fact_extractor import Fact


# Patterns that indicate answer-key/test-stem artifacts rather than reusable facts.
# Worked examples are NOT treated as noise here — they are routed by the renderer
# to a dedicated "Worked Example" section instead of being discarded.
_FACT_NOISE_PATTERNS = [
    re.compile(r"\bthe correct answer is\b", re.IGNORECASE),
    re.compile(r"\bwhich of the following\b", re.IGNORECASE),
    re.compile(r"\bmultiple\s*choice\b", re.IGNORECASE),
    re.compile(r"\bchoose\s+(?:one|all|the\s+best)\b", re.IGNORECASE),
    re.compile(r"\boption\s+[a-d]\b", re.IGNORECASE),
    re.compile(r"\bas\s+explicitly\s+stated\s+in\s+the\s+existing\s+facts\b", re.IGNORECASE),
]


def _is_noise_fact(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return True
    return any(pattern.search(text) for pattern in _FACT_NOISE_PATTERNS)


def apply_fact_hygiene(facts: Iterable[Fact]) -> tuple[list[Fact], int]:
    """Return clean facts and number of dropped facts.

    This filter runs after extraction but before concept-level filtering/grouping.
    """
    clean_facts: list[Fact] = []
    dropped = 0
    for fact in facts:
        if _is_noise_fact(fact.content):
            dropped += 1
            continue
        clean_facts.append(fact)
    return clean_facts, dropped
