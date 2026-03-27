from extract.fact_extractor import Fact
import re


STOPWORDS = {"a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "with", "by"}


def _unique_fact_contents(facts: list[Fact]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []

    for fact in facts:
        text = fact.content.strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(text)

    return items


def _build_summary(fact_contents: list[str], max_items: int = 2, max_chars: int = 320) -> str:
    if not fact_contents:
        return "No summary available."

    summary = " ".join(fact_contents[:max_items]).strip()
    if len(summary) <= max_chars:
        return summary

    return summary[: max_chars - 3].rstrip() + "..."


def _normalize_text_for_compare(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _title_case_with_connectors(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return text

    lower_connectors = {"of", "in", "and", "or", "to", "for", "on", "with", "by"}
    out: list[str] = []
    for i, token in enumerate(tokens):
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

    # Convert trailing acronym token into parenthetical form when plausible.
    if len(words) >= 2:
        tail = words[-1]
        head_words = words[:-1]
        if tail.isupper() and 2 <= len(tail) <= 6:
            initials = "".join(w[0].upper() for w in head_words if re.search(r"[A-Za-z]", w))
            if tail in initials or tail == initials[: len(tail)]:
                return f"{_title_case_with_connectors(' '.join(head_words))} ({tail})"

    return _title_case_with_connectors(concept)


def _concept_tokens(concept: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", concept.lower())
        if token not in STOPWORDS and len(token) > 1
    }


def _build_related_concepts(concepts: list[str], max_related: int = 3) -> dict[str, list[str]]:
    related: dict[str, list[str]] = {}
    token_map = {concept: _concept_tokens(concept) for concept in concepts}

    for concept in concepts:
        current = token_map[concept]
        scored: list[tuple[int, int, str]] = []

        for candidate in concepts:
            if candidate == concept:
                continue
            overlap = len(current.intersection(token_map[candidate]))
            if overlap == 0:
                continue
            scored.append((overlap, len(token_map[candidate]), candidate))

        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        related[concept] = [name for _, _, name in scored[:max_related]]

    return related


def classify_fact(fact: str) -> str:
    """
    Classify a fact into one of:
    definition | key_point | example | instruction
    """
    text = fact.strip()
    if not text:
        return "key_point"

    lower = text.lower()

    # Instructions first: actionable directives.
    instruction_phrases = [
        "calculate",
        "provide",
        "use the following",
        "compute",
        "determine",
        "find",
        "apply",
        "solve",
    ]
    if any(phrase in lower for phrase in instruction_phrases):
        return "instruction"

    # Imperative starts are usually instructional.
    if re.match(r"^(calculate|compute|determine|find|apply|use|solve)\b", lower):
        return "instruction"

    # Examples: numbers, scenarios, exercises.
    example_markers = ["for example", "e.g.", "scenario", "exercise", "suppose"]
    has_example_marker = any(marker in lower for marker in example_markers)
    number_count = len(re.findall(r"\b\d+([.,]\d+)?\b", lower))
    token_count = max(1, len(re.findall(r"[a-z0-9]+", lower)))
    numeric_density = number_count / token_count
    if has_example_marker or numeric_density >= 0.25:
        return "example"

    # Definitions: statements describing what something is.
    definition_markers = [" is ", " are ", " refers to ", " is defined as "]
    if re.search(r"\bthere are\b", lower):
        return "key_point"
    if any(marker in lower for marker in definition_markers):
        return "definition"

    return "key_point"


def _pick_best_definition(concept: str, definitions: list[str], key_points: list[str]) -> str:
    if not definitions:
        if key_points:
            # Fallback to a descriptive key point instead of leaving a blank definition.
            return max(key_points, key=lambda text: (len(re.findall(r"[A-Za-z0-9]+", text)), len(text)))
        return "No definition available."

    concept_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", concept.lower())
        if token not in {"the", "a", "an", "of", "for", "and", "or", "to", "in", "on"}
    ]

    def score(text: str) -> tuple[int, int, int, int]:
        lower = text.lower()
        fact_tokens = re.findall(r"[a-z0-9]+", lower)
        fact_token_set = set(fact_tokens)

        overlap = sum(1 for token in concept_tokens if token in fact_token_set)

        # Prefer explicit definition patterns over generic assertions.
        if " is defined as " in lower:
            pattern_rank = 0
            marker = "defined"
        elif " refers to " in lower:
            pattern_rank = 1
            marker = "refers"
        elif " is " in lower:
            pattern_rank = 2
            marker = "is"
        elif " are " in lower:
            pattern_rank = 3
            marker = "are"
        else:
            pattern_rank = 4
            marker = ""

        # Prefer definitions where concept mention appears before or near the marker.
        concept_pos = lower.find(concept.lower())
        marker_pos = lower.find(marker) if marker else -1
        if concept_pos != -1 and marker_pos != -1:
            distance_penalty = abs(marker_pos - concept_pos)
            order_penalty = 0 if concept_pos <= marker_pos else 1
        else:
            distance_penalty = 999
            order_penalty = 1

        # Prefer richer-but-not-too-long definitions.
        length_penalty = abs(len(fact_tokens) - 18)

        # Higher overlap is better, so negate it for min() scoring.
        return (-overlap, pattern_rank + order_penalty, distance_penalty, length_penalty)

    return min(definitions, key=score)


def generate_pages(grouped: dict[str, list[Fact]], include_empty_pages: bool = False) -> dict[str, str]:
    """
    Generate concept pages from grouped facts using deterministic fact types.

    Each page contains:
    - Definition (best definition fact)
    - Key Points (only key_point facts)
    - Excludes examples and instructions
    """
    pages: dict[str, str] = {}
    concept_names = list(grouped.keys())
    related_map = _build_related_concepts(concept_names)

    for concept, facts in grouped.items():
        display_title = normalize_page_title(concept)
        fact_contents = _unique_fact_contents(facts)

        definitions: list[str] = []
        key_points: list[str] = []

        for item in fact_contents:
            label = classify_fact(item)
            if label == "definition":
                definitions.append(item)
            elif label == "key_point":
                key_points.append(item)

        definition = _pick_best_definition(concept, definitions, key_points)
        definition_norm = _normalize_text_for_compare(definition)

        # Remove key points that duplicate the selected definition.
        key_points = [
            item
            for item in key_points
            if _normalize_text_for_compare(item) != definition_norm
        ]

        # Optionally skip pages that still have no meaningful content.
        if definition == "No definition available." and not key_points and not include_empty_pages:
            continue

        lines: list[str] = [
            f"# {display_title}",
            "",
            "## Definition",
            definition,
            "",
            "## Key Points",
        ]

        if key_points:
            for item in key_points:
                lines.append(f"- {item}")
        else:
            lines.append("- No key points available.")

        lines.append("")
        lines.append("## Related Concepts")
        related = related_map.get(concept, [])
        if related:
            for item in related:
                lines.append(f"- {normalize_page_title(item)}")
        else:
            lines.append("- None")

        pages[display_title] = "\n".join(lines)

    return pages


def render_pages_document(pages: dict[str, str]) -> str:
    """
    Render all pages into one clearly delineated text document.
    """
    if not pages:
        return "# Concept Pages\n\nNo pages generated.\n"

    sections: list[str] = ["# Concept Pages", ""]
    for i, (concept, page_text) in enumerate(pages.items(), start=1):
        sections.append(f"--- PAGE {i}: {concept} ---")
        sections.append(page_text)
        sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def render_pages_preview(pages: dict[str, str], max_pages: int = 2) -> str:
    """
    Render only the first N pages for console preview.
    """
    if max_pages < 1:
        return ""

    selected = list(pages.items())[:max_pages]
    if not selected:
        return ""

    sections: list[str] = []
    for i, (concept, page_text) in enumerate(selected, start=1):
        sections.append(f"--- PREVIEW PAGE {i}: {concept} ---")
        sections.append(page_text)
        sections.append("")

    return "\n".join(sections).rstrip()