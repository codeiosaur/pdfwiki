"""
Wiki renderer helpers.

Concept-type inference, wikilink injection, and fact promotion logic
used by generate_pages_wiki().
"""

import re
from typing import Optional

from generate.classify import (
    classify_fact,
    _has_template_markers,
    _is_low_signal_key_point,
)
from generate.page_layout import is_question_prompt


# Concept-type classification keywords (domain-agnostic)
RATIO_KEYWORDS = {"ratio", "rate", "turnover", "margin", "index", "coefficient"}
METHOD_KEYWORDS = {"method", "technique", "procedure", "approach", "identification", "allocation"}
SYSTEM_KEYWORDS = {"system", "framework", "model"}


def infer_concept_type_from_facts(fact_contents: list[str]) -> Optional[str]:
    """
    Infer concept type from the content of its facts rather than its name.
    Returns 'ratio' | 'method' | 'system' | None.
    """
    bag = " ".join(fact_contents).lower()

    # Strong formula signal -> ratio
    formula_signals = 0
    if re.search(r"\S\s*[=/÷]\s*\S", bag):
        formula_signals += 2
    if any(w in bag for w in ("divided by", "multiplied by", "per unit")):
        formula_signals += 1
    if any(w in bag for w in ("measures", "indicates", "higher ratio", "lower ratio")):
        formula_signals += 1
    if formula_signals >= 2:
        return "ratio"

    # System signal: ongoing/periodic process language
    system_signals = 0
    if any(w in bag for w in ("continuously", "periodically", "real-time", "ongoing")):
        system_signals += 2
    if any(w in bag for w in ("updates", "tracks", "maintains")):
        system_signals += 1
    if any(w in bag for w in ("at the time of", "at the end of", "as each")):
        system_signals += 1
    if system_signals >= 2:
        return "system"

    # Method signal: technique/procedure that assigns, records, or allocates
    method_signals = 0
    if any(w in bag for w in ("method", "technique", "procedure", "approach")):
        method_signals += 2
    if any(w in bag for w in ("records", "assigns", "allocates", "computes", "estimates")):
        method_signals += 1
    if any(w in bag for w in ("cost", "value", "price", "amount", "sold")):
        method_signals += 1
    if any(w in bag for w in ("as if", "based on", "in order of", "oldest", "most recent")):
        method_signals += 1
    if method_signals >= 2:
        return "method"

    return None


def classify_concept_type(concept: str, fact_contents: Optional[list[str]] = None) -> str:
    """
    Classify a concept for template selection: ratio | method | system | general.

    Strategy (domain-agnostic):
    1. Check if concept name contains generic template keywords.
    2. If no keyword match, infer from the concept's fact contents.
    3. Default to 'general'.
    """
    lower = concept.lower().strip()
    tokens = set(re.findall(r"[a-z]+", lower))

    if tokens.intersection(RATIO_KEYWORDS):
        return "ratio"
    if tokens.intersection(METHOD_KEYWORDS):
        return "method"
    if tokens.intersection(SYSTEM_KEYWORDS):
        return "system"

    if fact_contents:
        inferred = infer_concept_type_from_facts(fact_contents)
        if inferred:
            return inferred

    return "general"


def inject_wikilinks(
    text: str,
    all_titles: set[str],
    current_title: str,
    alias_map: Optional[dict[str, str]] = None,
) -> str:
    """
    Replace occurrences of other concept titles with [[wikilinks]] in body text.
    Only links each concept once (first occurrence).
    Skips text already inside [[ ]] to prevent nested links.

    alias_map: optional {alias_lowercase: canonical_display_title}.  Aliases
    are tried before exact titles, allowing acronyms (e.g. "FIFO") to link to
    their full-name page even when the full title doesn't appear in the text.
    """
    if not all_titles and not alias_map:
        return text

    linked: set[str] = set()  # canonical titles already injected

    def _inject_one(match_text: str, canonical: str) -> bool:
        """Inject [[canonical]] for the first match of match_text. Returns True on success."""
        nonlocal text
        parts = re.split(r'(\[\[[^\]]*\]\])', text)
        pattern = re.compile(r'\b' + re.escape(match_text) + r'\b', re.IGNORECASE)
        replaced = False
        new_parts = []
        for part in parts:
            if part.startswith("[[") and part.endswith("]]"):
                new_parts.append(part)
            elif not replaced and pattern.search(part):
                part = pattern.sub(f"[[{canonical}]]", part, count=1)
                replaced = True
                new_parts.append(part)
            else:
                new_parts.append(part)
        if replaced:
            text = "".join(new_parts)
        return replaced

    # Pass 1: alias injection — longest alias first so more specific wins
    if alias_map:
        for alias_lower, canonical in sorted(alias_map.items(), key=lambda x: len(x[0]), reverse=True):
            if canonical == current_title or canonical in linked:
                continue
            if _inject_one(alias_lower, canonical):
                linked.add(canonical)

    # Pass 2: exact-title injection — longest title first
    for title in sorted(all_titles - {current_title}, key=len, reverse=True):
        if title in linked:
            continue
        if _inject_one(title, title):
            linked.add(title)

    return text


def promote_all_facts_to_content(
    fact_contents: list[str],
    definition: str,
) -> list[str]:
    """
    Return all facts that aren't the definition, aren't instructions,
    and aren't low-signal.
    """
    from generate.classify import _normalize_text_for_compare

    definition_norm = _normalize_text_for_compare(definition)
    promoted: list[str] = []
    seen_normalized: set[str] = set()
    template_counts: dict[str, int] = {}

    for item in fact_contents:
        if _has_template_markers(item):
            continue
        if is_question_prompt(item):
            continue
        if _is_low_signal_key_point(item):
            continue
        if _normalize_text_for_compare(item) == definition_norm:
            continue
        base_class = classify_fact(item)
        if base_class == "instruction":
            continue

        normalized = _normalize_text_for_compare(item)
        if normalized in seen_normalized:
            continue

        # Collapse repetitive numeric variants that share the same sentence template.
        # Keep at most two examples so method comparisons can still survive.
        template = re.sub(r"\b\$?\d[\d,]*(?:\.\d+)?%?\b", "<num>", normalized)
        template_count = template_counts.get(template, 0)
        if template_count >= 2:
            continue

        seen_normalized.add(normalized)
        template_counts[template] = template_count + 1
        promoted.append(item)

    return promoted