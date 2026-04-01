"""
Concept name canonicalization.

Uses an LLM to normalize variant concept names to canonical forms,
with a persistent cache to avoid redundant calls.
"""

from typing import Optional, TYPE_CHECKING
from pathlib import Path

import json
import re

if TYPE_CHECKING:
    from backend.base import LLMBackend

CANONICAL_CACHE_PATH = Path(__file__).with_name("canonical_cache.json")


def load_canonical_cache() -> dict[str, Optional[str]]:
    if not CANONICAL_CACHE_PATH.exists():
        return {}

    try:
        with CANONICAL_CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    cache: dict[str, Optional[str]] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        cache[key] = value if isinstance(value, str) else None
    return cache


def save_canonical_cache(cache: dict[str, Optional[str]]) -> None:
    with CANONICAL_CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _title_case_concept(text: str) -> str:
    lowercase_words = {"of", "in", "and", "or", "to"}
    tokens = text.split()
    if not tokens:
        return text

    out: list[str] = []
    for i, token in enumerate(tokens):
        lower = token.lower()
        if i > 0 and lower in lowercase_words:
            out.append(lower)
        else:
            out.append(lower.capitalize())
    return " ".join(out)


def normalize_concept_rules(concept: str) -> str:
    """
    Deterministic concept normalization before LLM canonicalization.

    - Expand common abbreviations: FIFO, LIFO, LCM, COGS, AVG
    - Remove redundant trailing suffixes like Method/System/Approach when safe
    - Fix spacing/casing issues
    - Normalize selected variants to canonical textbook forms
    """
    if not concept:
        return concept

    normalized = re.sub(r"\s+", " ", concept).strip()
    normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)

    lower = normalized.lower()
    if re.search(r"\bfifo\b", lower) or re.search(r"\bfirst\s+in\s+first\s+out\b", lower):
        normalized = "First In First Out"
    elif re.search(r"\blifo\b", lower) or re.search(r"\blast\s+in\s+first\s+out\b", lower):
        normalized = "Last In First Out"
    elif re.search(r"\blcm\b", lower) or re.search(r"\blower\s+of\s+cost\s+or\s+market\b", lower):
        normalized = "Lower of Cost or Market"
    elif re.search(r"\bcogs\b", lower) or re.search(r"\bcost\s+of\s+goods\s+sold\b", lower):
        normalized = "Cost of Goods Sold"
    elif re.search(r"\bavg\b", lower) or re.search(r"\baverage\b", lower):
        normalized = "Average Cost"
    elif (
        ("epcs" in lower or "electronic product" in lower)
        and "code" in lower
    ):
        normalized = "Electronic Product Code"

    words = normalized.split()
    if len(words) >= 3 and words[-1].lower() in {"method", "system", "approach"}:
        stem = " ".join(words[:-1]).lower()
        if any(key in stem for key in [
            "first in first out",
            "last in first out",
            "lower of cost or market",
            "cost of goods sold",
            "average cost",
            "electronic product code",
        ]):
            normalized = " ".join(words[:-1])

    return _title_case_concept(normalized)


def canonicalize_concepts(
    concepts: list[str],
    backend: "LLMBackend",
) -> dict[str, Optional[str]]:
    """
    Canonicalize concept names using an LLM, with caching.

    Args:
        concepts: List of concept names to canonicalize.
        backend:  The LLM backend to use for canonicalization.

    Returns:
        Mapping from original name to canonical name (or None if invalid).
    """
    if not concepts:
        return {}

    cache = load_canonical_cache()
    missing = [concept for concept in concepts if concept not in cache]

    # Fast path: every concept is already cached — skip prompt construction entirely.
    if len(missing) == 0:
        return {name: cache.get(name) for name in concepts}

    prompt = f"""
    Canonicalize concept names from academic material.

    Goals:
    - Fix spelling, casing, and malformed possessives.
    - Expand common abbreviations when appropriate:
        COGS -> Cost of Goods Sold
        LCM -> Lower of Cost or Market
    - Normalize variants to one textbook name:
        FIFO, First In First Out Method -> First In First Out
    - Fix acronym casing errors:
        Epcs -> Electronic Product Code
    - Remove redundant suffixes when non-essential:
        Method, System, Approach
    - Keep names concise noun phrases (1-4 words).

    Strict rules:
    - Do NOT merge distinct concepts (FIFO and LIFO must stay separate).
    - Do NOT generalize (do not map FIFO -> Inventory Method).
    - Do NOT invent concepts.
    - Preserve meaning exactly.
    - Return null only if the concept is invalid, vague, or not a real concept.

    Output:
    - Return ONLY valid JSON object mapping original -> canonical_or_null.
    - Keep every input key in the output.

    Example format:
    {{
        "Concept A": "Canonical Name",
        "Concept B": null
    }}

    Concepts:
    {chr(10).join("- " + c for c in missing)}
    """

    try:
        raw_content = backend.generate(prompt, max_tokens=600)
    except Exception:
        for name in missing:
            cache[name] = None
        save_canonical_cache(cache)
        return {name: cache.get(name) for name in concepts}

    # Parse JSON safely.
    try:
        parsed = json.loads(raw_content)
    except Exception:
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            for name in missing:
                cache[name] = None
            save_canonical_cache(cache)
            return {name: cache.get(name) for name in concepts}
        try:
            parsed = json.loads(raw_content[start : end + 1])
        except Exception:
            for name in missing:
                cache[name] = None
            save_canonical_cache(cache)
            return {name: cache.get(name) for name in concepts}

    if not isinstance(parsed, dict):
        for name in missing:
            cache[name] = None
        save_canonical_cache(cache)
        return {name: cache.get(name) for name in concepts}

    for name in missing:
        value = parsed.get(name)
        cache[name] = value if isinstance(value, str) else None

    save_canonical_cache(cache)
    return {name: cache.get(name) for name in concepts}


def needs_canonicalization(concept: str) -> bool:
    if any(len(token) == 1 for token in re.findall(r"[A-Za-z]+", concept)):
        return True
    if concept != concept.strip():
        return True
    if re.search(r"\s{2,}", concept):
        return True
    if re.search(r"[\-_/]{2,}|[()]{2,}|[,:;.]\s*[,:;.]", concept):
        return True

    words = re.findall(r"[A-Za-z]+", concept.lower())
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            return True

    for token in re.findall(r"[A-Za-z]+", concept):
        if any(c.islower() for c in token) and any(c.isupper() for c in token):
            if not token.istitle():
                return True

    return False