"""
Concept title formatting and normalization.

Handles acronym detection, title casing, and display-title normalization
for page headings and wikilinks.
"""

import re


STOPWORDS = {"a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "with", "by"}

# Populated dynamically by build_acronym_map() at render time.
ACRONYM_CANONICAL: dict[str, str] = {}


def build_acronym_map(concept_names: list[str]) -> dict[str, str]:
    """
    Build an acronym normalization map from the actual concept names.
    Any all-caps token (2-6 chars) found in concept names becomes canonical.
    """
    acronyms: dict[str, str] = {}
    for name in concept_names:
        for token in name.split():
            clean = re.sub(r"[^A-Za-z0-9]", "", token)
            if clean.isupper() and 2 <= len(clean) <= 6:
                acronyms[clean.lower()] = clean
    return acronyms


def _title_case_with_connectors(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return text

    lower_connectors = {"of", "in", "and", "or", "to", "for", "on", "with", "by"}
    out: list[str] = []
    for i, token in enumerate(tokens):
        acronym = ACRONYM_CANONICAL.get(token.lower())
        if acronym:
            out.append(acronym)
            continue
        if token.isupper() and 2 <= len(token) <= 6:
            out.append(token)
            continue
        lower = token.lower()
        if i > 0 and lower in lower_connectors:
            out.append(lower)
        else:
            out.append(lower.capitalize())
    return " ".join(out)


def normalize_page_title(concept: str) -> str:
    """
    Normalize display title for pages.

    Example:
    - "Days Sales In Inventory DSI" -> "Days Sales in Inventory (DSI)"
    - "Free On Board FOB" -> "Free on Board (FOB)"
    """
    words = concept.split()
    if not words:
        return concept

    if len(words) >= 2:
        tail = words[-1]
        head_words = words[:-1]
        if tail.isupper() and 2 <= len(tail) <= 6:
            initials = "".join(w[0].upper() for w in head_words if re.search(r"[A-Za-z]", w))
            if tail in initials or tail == initials[: len(tail)]:
                return f"{_title_case_with_connectors(' '.join(head_words))} ({tail})"

    return _title_case_with_connectors(concept)


def concept_tokens(concept: str) -> set[str]:
    """Extract meaningful lowercase tokens from a concept name."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", concept.lower())
        if token not in STOPWORDS and len(token) > 1
    }