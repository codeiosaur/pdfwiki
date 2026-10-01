import re
from typing import List, Tuple

from generate.classify import classify_fact, _is_low_signal_key_point, _looks_like_worked_example

_PLACEHOLDER_NAMES = {"example", "overview", "summary", "introduction", "conclusion"}

# Internal pipeline/process labels that should never surface as user-facing concepts.
_INTERNAL_CONCEPT_EXACT = {
    "canonicalize concept names",
    "source chunk id",
    "chunk id",
    "json array",
    "existing facts",
}

# Python built-in/reserved values that should never be concept names
_PYTHON_RESERVED_VALUES = {
    "none", "true", "false", "null",  # JSON/Python literals
    "nan", "inf", "infinity",  # Special numeric values
}


def is_valid_concept(name: str) -> bool:
    """
    Determine if a concept name is likely a reusable knowledge concept.
    
    Reject only obvious placeholders and pipeline artifacts. A name's subject,
    grammar, or date alone cannot establish whether it is a useful concept.
    
    Args:
        name: Concept name to validate
        
    Returns:
        True if concept is valid, False otherwise
    """
    if not name or not isinstance(name, str):
        return False

    name = name.strip()

    normalized = re.sub(r"\s+", " ", name.lower()).strip()
    if not normalized or len(name.split()) > 6:
        return False
    if normalized in _PYTHON_RESERVED_VALUES | _PLACEHOLDER_NAMES | _INTERNAL_CONCEPT_EXACT:
        return False
    if "canonicalize" in normalized and "concept" in normalized:
        return False

    return True


def filter_concepts(facts: List) -> List:
    """
    Filter facts by concept validity.
    
    Args:
        facts: List of Fact objects with .concept attribute
        
    Returns:
        Filtered list of facts with only valid concepts
    """
    return [fact for fact in facts if is_valid_concept(fact.concept)]


def filter_publishable_grouped_concepts(grouped: dict[str, list]) -> dict[str, list]:
    """Remove grouped concept pages that should never be published."""
    return {
        concept: facts
        for concept, facts in grouped.items()
        if is_valid_concept(concept)
    }


def filter_example_saturated_concepts(grouped: dict[str, list], threshold: float = 0.7) -> Tuple[dict[str, list], int]:
    """Remove or suppress concept groups dominated by example-like facts.

    Behavior:
    - Remove example-like facts from each concept group.
    - Keep the concept if at least one non-example fact remains.
    - Drop the concept only when all facts are example-like.

    Returns a tuple of (filtered_grouped, dropped_count).
    """
    result: dict[str, list] = {}
    dropped = 0
    for concept, facts in grouped.items():
        if not facts:
            # Preserve empty buckets to allow other pruning logic to handle them
            result[concept] = facts
            continue
        kept_facts = []
        for f in facts:
            content = f.content if hasattr(f, "content") else str(f)
            # Treat facts as example-like if they are classified as examples
            # or appear to be low-signal/worked-example content (numeric-heavy,
            # short worksheet-style lines, or marker phrases).
            is_example_like = (
                classify_fact(content) == "example"
                or _is_low_signal_key_point(content)
                or _looks_like_worked_example(content)
            )
            if not is_example_like:
                kept_facts.append(f)

        # Keep concept with any surviving non-example facts.
        if kept_facts:
            result[concept] = kept_facts
            continue

        # If everything is example-like, drop the concept page.
        if len(facts) > 0:
            dropped += 1
            continue

        result[concept] = kept_facts
    return result, dropped
