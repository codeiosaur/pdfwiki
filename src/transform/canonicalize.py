"""
Concept name canonicalization.

Uses an LLM to normalize variant concept names to canonical forms,
with a persistent cache to avoid redundant calls.
"""

from typing import Optional, TYPE_CHECKING
from pathlib import Path

import json
import re

from transform.normalize import normalize_concept_rules

if TYPE_CHECKING:
    from backend.base import LLMBackend

# Old responses used an accounting-biased prompt. Do not reuse them with the
# domain-neutral prompt; leave the old ignored cache untouched for recovery.
CANONICAL_CACHE_PATH = Path(__file__).with_name("canonical_cache_v2.json")


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
    - Fix obvious spelling, spacing, and casing errors without changing meaning.
    - Expand an abbreviation only if its meaning is unambiguous from these names.
    - Keep informative words such as Method, System, Model, or Theory when they
      distinguish a concept from another one.

    Strict rules:
    - Do NOT merge related-but-distinct concepts or names that differ in
      meaningful qualifiers, including prepositions, numbers, and suffixes.
    - Do NOT guess the expansion of an ambiguous abbreviation.
    - Do NOT generalize a specific name to a broader topic.
    - Do NOT invent concepts.
    - Preserve meaning exactly.
    - If uncertain, return the original name unchanged. Use null only for an
      obvious placeholder or malformed artifact, never for an unfamiliar term.

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
        # A temporary backend failure is not a canonicalization decision.
        return {name: cache.get(name) for name in concepts}

    # Parse JSON safely.
    try:
        parsed = json.loads(raw_content)
    except Exception:
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {name: cache.get(name) for name in concepts}
        try:
            parsed = json.loads(raw_content[start : end + 1])
        except Exception:
            return {name: cache.get(name) for name in concepts}

    if not isinstance(parsed, dict):
        return {name: cache.get(name) for name in concepts}

    for name in missing:
        if name not in parsed:
            continue  # An incomplete response should be retried next run.
        value = parsed[name]
        if value is None:
            cache[name] = None
        elif isinstance(value, str) and value.strip():
            cache[name] = value.strip()

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
