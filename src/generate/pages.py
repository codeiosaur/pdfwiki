from extract.fact_extractor import Fact
import re
from typing import List, Optional
from transform.matching import has_antonym_conflict


STOPWORDS = {"a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "with", "by"}
ACRONYM_CANONICAL = {
    "avg": "AVG",
    "cogs": "COGS",
    "dsi": "DSI",
    "epc": "EPC",
    "epcs": "EPC",
    "fifo": "FIFO",
    "gaap": "GAAP",
    "ifrs": "IFRS",
    "lifo": "LIFO",
    "upc": "UPC",
}
KEY_POINT_NOISE_MARKERS = {
    "figure",
    "table",
    "see page",
    "as shown",
    "wave of the future",
}

# Template variable markers that indicate incomplete fact extraction
TEMPLATE_MARKERS = {
    r"\$\.\.\.",           # $...
    r"units at \$",        # incomplete variable substitution
    r"\bFigure\s+\d+",     # Figure references
    r"^\w+\s+was\s+made\s+up\s+of\s+\.\.\."  # "... was made up of ..."
}
KEY_POINT_INSTRUCTION_MARKERS = {
    "calculate",
    "compute",
    "determine",
    "find",
    "assume",
    "use the following",
    "indicate the effect",
    "search",
    "submit",
    "locate",
    "prepare",
    "write",
    "show that",
    "demonstrate",
    "identify",
    "analyze",
    "examine",
    "verify",
    "check",
    "state",
    "consider",
}

SEMANTIC_FORMULA_MARKERS = {
    "=",
    "equation",
    "formula",
    "divided by",
    "multiplied by",
}
SEMANTIC_INTERPRETATION_MARKERS = {
    "means",
    "indicates",
    "shows",
    "suggests",
    "measures",
    "implies",
}
SEMANTIC_CAUTION_MARKERS = {
    "misleading",
    "manipulate",
    "risk",
    "caution",
    "however",
    "can be",
    "may be",
    "warning",
}

ACCOUNTING_DOMAIN_MARKERS = {
    "inventory",
    "cogs",
    "cost of goods sold",
    "gross margin",
    "gross profit",
    "ratio",
    "fifo",
    "lifo",
    "ifrs",
    "gaap",
    "perpetual",
    "periodic",
    "consignment",
    "valuation",
}

INCOMPATIBLE_KEY_POINT_PATTERNS = {
    "periodic": [
        r"real[\s-]?time",
        r"ongoing\s+basis",
        r"each\s+individual\s+sale",
        r"continuously\s+as\s+each\s+transaction",
    ],
    "perpetual": [
        r"end\s+of\s+the\s+reporting\s+period",
        r"end\s+of\s+each\s+month",
        r"month,\s*quarter,\s*or\s*year",
        r"updated\s+at\s+the\s+end\s+of\s+the\s+period",
    ],
}


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


def _numeric_density(text: str) -> float:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return 0.0

    numeric = sum(1 for token in tokens if re.search(r"\d", token))
    return numeric / len(tokens)


def _has_template_markers(text: str) -> bool:
    """Check if text contains incomplete template variable markers."""
    for pattern in TEMPLATE_MARKERS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _is_low_signal_key_point(text: str) -> bool:
    lower = text.lower().strip()
    if not lower:
        return True

    if _has_template_markers(text):
        return True

    if any(marker in lower for marker in KEY_POINT_NOISE_MARKERS):
        return True

    if any(marker in lower for marker in KEY_POINT_INSTRUCTION_MARKERS):
        return True

    if "website" in lower and any(marker in lower for marker in {"search", "submit", "locate"}):
        return True

    if _numeric_density(text) >= 0.22:
        return True

    if len(re.findall(r"\S+", text)) < 4:
        return True

    return False


def _select_fallback_definition(
    concept: str,
    definitions: list[str],
    key_points: list[str],
    fact_contents: list[str],
) -> Optional[str]:
    # 1) Try scored definition candidates first.
    selected = select_definition(concept, definitions)
    if selected is not None:
        return selected

    # 2) Fall back to scored key-point candidates.
    selected = select_definition(concept, key_points)
    if selected is not None:
        return selected

    # 3) Fall back to strongest non-instruction/non-example fact.
    non_instruction_non_example = [
        item for item in fact_contents if classify_fact(item) in {"definition", "key_point"}
    ]
    selected = select_definition(concept, non_instruction_non_example)
    if selected is not None:
        return selected

    # 4) Final fallback to first compatible key point sentence.
    if key_points:
        first = key_points[0].strip()
        if first:
            return first

    return None


def _build_lead(definition: str, key_points: list[str]) -> str:
    if definition and definition != "No definition available.":
        parts = re.split(r"(?<=[.!?])\s+", definition.strip())
        return parts[0].strip() if parts and parts[0].strip() else definition

    if key_points:
        return key_points[0]

    return "No lead available."


def _looks_like_formula(text: str) -> bool:
    lower = text.lower()
    # Strong signal: explicit equation form.
    if re.search(r"\S\s*=\s*\S", text):
        return True

    # Strong signal: equation/formula keywords.
    if any(marker in lower for marker in {"equation", "formula"}):
        return True

    # Ratio-like formulas need a computation cue, not just a mention.
    has_ratio_word = "ratio" in lower
    has_compute_cue = any(op in lower for op in {"divided by", "multiplied by", "plus", "minus", "/", "÷", "×", "*"})
    if has_ratio_word and has_compute_cue:
        return True

    # Generic arithmetic expression with at least two numeric operands.
    number_count = len(re.findall(r"\b\d+(?:[.,]\d+)?\b", text))
    has_operator = bool(re.search(r"[+\-/*=÷×]", text))
    if number_count >= 2 and has_operator:
        return True

    return False


def _is_accounting_domain(concept: str, fact_contents: list[str]) -> bool:
    bag = " ".join([concept] + fact_contents).lower()
    return any(marker in bag for marker in ACCOUNTING_DOMAIN_MARKERS)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = _normalize_text_for_compare(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _trim_section(items: list[str], limit: int) -> list[str]:
    cleaned = _dedupe_preserve_order(items)
    return cleaned[:limit] if limit > 0 else cleaned


def classify_semantic_fact(fact: str) -> str:
    """
    Semantic classification used by the enhanced renderer.

    Returns one of:
    definition | formula | interpretation | example | caution | key_point | instruction
    """
    base = classify_fact(fact)
    if base == "instruction":
        return "instruction"

    lower = fact.lower().strip()
    if _looks_like_formula(fact):
        return "formula"

    if any(marker in lower for marker in SEMANTIC_CAUTION_MARKERS):
        return "caution"

    if any(marker in lower for marker in SEMANTIC_INTERPRETATION_MARKERS):
        return "interpretation"

    if base == "example":
        return "example"

    if base == "definition":
        return "definition"

    return "key_point"


def _emphasize_concept_once(text: str, display_title: str) -> str:
    pattern = re.compile(re.escape(display_title), re.IGNORECASE)
    if pattern.search(text):
        return pattern.sub(f"**{display_title}**", text, count=1)
    return f"**{display_title}**: {text}"


def _build_enhanced_intro(
    display_title: str,
    definition: str,
    interpretations: list[str],
    key_points: list[str],
) -> str:
    primary = definition if definition and definition != "No definition available." else ""
    if not primary and key_points:
        primary = key_points[0]
    if not primary and interpretations:
        primary = interpretations[0]
    if not primary:
        return f"**{display_title}** is a concept in this source material."

    primary = _emphasize_concept_once(primary.strip(), display_title)

    secondary_candidates = interpretations + key_points
    secondary = ""
    norm_primary = _normalize_text_for_compare(primary)
    for item in secondary_candidates:
        if _normalize_text_for_compare(item) != norm_primary:
            secondary = item.strip()
            break

    if not secondary:
        return primary

    secondary = secondary[0].upper() + secondary[1:] if secondary else secondary
    return f"{primary} {secondary}"


def _fact_sources(facts: list[Fact]) -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {}
    for fact in facts:
        text = fact.content.strip()
        if not text:
            continue
        source = fact.source_chunk_id.strip() if fact.source_chunk_id else "unknown"
        existing = sources.setdefault(text, [])
        if source not in existing:
            existing.append(source)
    return sources


def _citation_suffixes(
    items: list[str],
    text_to_sources: dict[str, list[str]],
    source_key_to_note_index: dict[tuple[str, ...], int],
    start_index: int,
) -> tuple[list[str], list[str], int]:
    rendered: list[str] = []
    notes: list[str] = []
    next_index = start_index

    if not items:
        return rendered, notes, next_index

    for item in items:
        source_ids = text_to_sources.get(item, [])
        source_key = tuple(source_ids)

        if source_key in source_key_to_note_index:
            note_index = source_key_to_note_index[source_key]
        else:
            note_index = next_index
            source_key_to_note_index[source_key] = note_index
            joined_sources = ", ".join(source_ids) if source_ids else "unknown"
            notes.append(f"[^{note_index}]: chunk={joined_sources}")
            next_index += 1

        rendered.append(f"{item} [^{note_index}]")

    return rendered, notes, next_index


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


def select_definition(concept: str, facts: List[str]) -> Optional[str]:
    """
    Select the best definition candidate using deterministic scoring rules.

    Returns None if all candidate scores are low.
    """
    if not facts:
        return None

    # Filter out facts with incomplete template markers
    facts = [f for f in facts if not _has_template_markers(f)]

    concept_lower = concept.lower().strip()
    concept_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", concept_lower)
        if token not in STOPWORDS and len(token) > 1
    ]

    def score_fact(text: str) -> int:
        lower = text.lower().strip()
        score = 0

        # +3 if fact starts with: "X is...", "X refers to...", "X is a..."
        starts_with_patterns = [
            f"{concept_lower} is ",
            f"{concept_lower} refers to ",
            f"{concept_lower} is a ",
            f"the {concept_lower} is ",
            f"the {concept_lower} refers to ",
            f"the {concept_lower} is a ",
        ]
        if any(lower.startswith(pattern) for pattern in starts_with_patterns):
            score += 3

        # +2 if fact contains definitional nouns.
        if any(term in lower for term in ["method", "process", "system", "measure", "ratio", "term", "account"]):
            score += 2

        # +1 if sentence starts with article + noun phrase style definition.
        if re.match(r"^(a|an|the)\s+[a-z]", lower):
            score += 1

        # +1 if fact contains concept name or main concept tokens.
        if concept_lower in lower:
            score += 1
        elif concept_tokens and any(token in lower for token in concept_tokens):
            score += 1

        # -2 if fact contains numbers ($, %, digits), unless scientific/stat data context exists.
        has_number = bool(re.search(r"[\d$%]", text))
        stat_context_terms = [
            "mean",
            "median",
            "variance",
            "standard deviation",
            "sample",
            "dataset",
            "confidence interval",
            "p-value",
            "correlation",
            "statistic",
            "ratio",
        ]
        has_stat_context = any(term in lower for term in stat_context_terms)
        if has_number and not has_stat_context:
            score -= 2

        # -2 if fact contains instructional phrasing.
        if any(term in lower for term in ["calculate", "use the following", "assume"]):
            score -= 2

        # -2 if fact is very long (> 40 words).
        if len(re.findall(r"\S+", text)) > 40:
            score -= 2

        # -3 if fact is vague.
        vague_markers = [
            "key aspect",
            "important",
            "various",
            "etc",
            "kind of",
            "type of",
            "challenge of",
            "wave of the future",
        ]
        if any(marker in lower for marker in vague_markers):
            score -= 3

        # Penalize list-enumeration statements that are usually not definitions.
        if lower.startswith("there are"):
            score -= 3

        return score

    best_fact: Optional[str] = None
    best_score = -999
    for fact in facts:
        current_score = score_fact(fact)
        if current_score > best_score:
            best_score = current_score
            best_fact = fact

    # "Low" means no positive evidence of being a useful definition.
    if best_score < 0:
        return None

    return best_fact


def classify_fact(fact: str) -> str:
    """
    Classify a fact into one of:
    definition | key_point | example | instruction
    """
    text = fact.strip()
    if not text:
        return "key_point"

    lower = text.lower()

    # Enumeration statements are usually listing context, not definitions.
    if re.match(r"^there\s+(is|are)\b", lower):
        return "example"

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
        "search",
        "submit",
        "locate",
        "prepare",
        "write",
    ]
    if any(phrase in lower for phrase in instruction_phrases):
        return "instruction"

    # Imperative starts are usually instructional.
    if re.match(r"^(calculate|compute|determine|find|apply|use|solve|search|submit|locate|prepare|write)\b", lower):
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
        text_to_sources = _fact_sources(facts)

        definitions: list[str] = []
        key_points: list[str] = []

        for item in fact_contents:
            label = classify_fact(item)
            if label == "definition":
                definitions.append(item)
            elif label == "key_point":
                key_points.append(item)

        key_points = [
            item
            for item in key_points
            if not _is_low_signal_key_point(item)
            and not has_antonym_conflict(concept, item)
        ]

        concept_lower = concept.lower()
        for concept_marker, patterns in INCOMPATIBLE_KEY_POINT_PATTERNS.items():
            if concept_marker in concept_lower:
                key_points = [
                    item
                    for item in key_points
                    if not any(re.search(pattern, item.lower()) for pattern in patterns)
                ]

        selected_definition = _select_fallback_definition(
            concept=concept,
            definitions=definitions,
            key_points=key_points,
            fact_contents=fact_contents,
        )
        if selected_definition is None:
            definition = "No definition available."
        else:
            definition = selected_definition
        definition_norm = _normalize_text_for_compare(definition)

        # Remove key points that duplicate the selected definition.
        key_points = [
            item
            for item in key_points
            if _normalize_text_for_compare(item) != definition_norm
        ]

        lead = _build_lead(definition, key_points)

        citation_index = 1
        citation_map: dict[tuple[str, ...], int] = {}
        definition_rendered, definition_notes, citation_index = _citation_suffixes(
            [definition],
            text_to_sources,
            citation_map,
            citation_index,
        )
        key_point_rendered, key_point_notes, citation_index = _citation_suffixes(
            key_points,
            text_to_sources,
            citation_map,
            citation_index,
        )
        combined_notes = definition_notes + key_point_notes

        # Optionally skip pages that still have no meaningful content.
        if definition == "No definition available." and not key_points and not include_empty_pages:
            continue

        lines: list[str] = [
            f"# {display_title}",
            "",
            "## Lead",
            lead,
            "",
            "## Definition",
            definition_rendered[0] if definition_rendered else definition,
            "",
            "## Key Points",
        ]

        if key_point_rendered:
            for item in key_point_rendered:
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

        lines.append("")
        lines.append("## Sources")
        if combined_notes:
            lines.extend(combined_notes)
        else:
            lines.append("- None")

        pages[display_title] = "\n".join(lines)

    return pages


def generate_pages_enhanced(grouped: dict[str, list[Fact]], include_empty_pages: bool = False) -> dict[str, str]:
    """
    Generate richer wiki-style pages with semantic sections.

    This renderer keeps deterministic logic and source citations while structuring
    content into narrative sections similar to curated study notes.
    """
    pages: dict[str, str] = {}
    concept_names = list(grouped.keys())
    related_map = _build_related_concepts(concept_names)

    for concept, facts in grouped.items():
        display_title = normalize_page_title(concept)
        fact_contents = _unique_fact_contents(facts)
        text_to_sources = _fact_sources(facts)

        definitions: list[str] = []
        formulas: list[str] = []
        interpretations: list[str] = []
        examples: list[str] = []
        cautions: list[str] = []
        key_points: list[str] = []

        for item in fact_contents:
            if _has_template_markers(item):
                continue

            semantic = classify_semantic_fact(item)
            if semantic == "instruction":
                continue
            if semantic == "definition":
                definitions.append(item)
            elif semantic == "formula":
                formulas.append(item)
            elif semantic == "interpretation":
                interpretations.append(item)
            elif semantic == "example":
                examples.append(item)
            elif semantic == "caution":
                cautions.append(item)
            else:
                key_points.append(item)

        key_points = [item for item in key_points if not _is_low_signal_key_point(item)]
        interpretations = [item for item in interpretations if not _is_low_signal_key_point(item)]

        concept_lower = concept.lower()
        for concept_marker, patterns in INCOMPATIBLE_KEY_POINT_PATTERNS.items():
            if concept_marker in concept_lower:
                def _compatible(text: str) -> bool:
                    return not any(re.search(pattern, text.lower()) for pattern in patterns)

                key_points = [item for item in key_points if _compatible(item)]
                interpretations = [item for item in interpretations if _compatible(item)]

        selected_definition = _select_fallback_definition(
            concept=concept,
            definitions=definitions,
            key_points=key_points,
            fact_contents=fact_contents,
        )
        definition = selected_definition if selected_definition is not None else "No definition available."

        is_accounting = _is_accounting_domain(concept, fact_contents)

        definition_norm = _normalize_text_for_compare(definition)
        key_points = [
            item for item in key_points if _normalize_text_for_compare(item) != definition_norm
        ]
        interpretations = [
            item for item in interpretations if _normalize_text_for_compare(item) != definition_norm
        ]

        # Deterministic section caps keep pages concise and sample-like.
        if is_accounting:
            formulas = _trim_section(formulas, 3)
            interpretations = _trim_section(interpretations, 4)
            examples = _trim_section(examples, 2)
            cautions = _trim_section(cautions, 3)
            key_points = _trim_section(key_points, 4)
        else:
            formulas = _trim_section(formulas, 2)
            interpretations = _trim_section(interpretations, 3)
            examples = _trim_section(examples, 2)
            cautions = _trim_section(cautions, 2)
            key_points = _trim_section(key_points, 3)

        if (
            definition == "No definition available."
            and not formulas
            and not interpretations
            and not examples
            and not cautions
            and not key_points
            and not include_empty_pages
        ):
            continue

        intro = _build_enhanced_intro(display_title, definition, interpretations, key_points)

        citation_index = 1
        citation_map: dict[tuple[str, ...], int] = {}

        definition_rendered, definition_notes, citation_index = _citation_suffixes(
            [definition],
            text_to_sources,
            citation_map,
            citation_index,
        )
        formula_rendered, formula_notes, citation_index = _citation_suffixes(
            formulas,
            text_to_sources,
            citation_map,
            citation_index,
        )
        interpretation_rendered, interpretation_notes, citation_index = _citation_suffixes(
            interpretations,
            text_to_sources,
            citation_map,
            citation_index,
        )
        example_rendered, example_notes, citation_index = _citation_suffixes(
            examples,
            text_to_sources,
            citation_map,
            citation_index,
        )
        caution_rendered, caution_notes, citation_index = _citation_suffixes(
            cautions,
            text_to_sources,
            citation_map,
            citation_index,
        )
        key_point_rendered, key_point_notes, citation_index = _citation_suffixes(
            key_points,
            text_to_sources,
            citation_map,
            citation_index,
        )

        combined_notes = (
            definition_notes
            + formula_notes
            + interpretation_notes
            + example_notes
            + caution_notes
            + key_point_notes
        )

        lines: list[str] = [
            f"# {display_title}",
            "",
            intro,
            "",
            "---",
            "",
            "## Definition",
            definition_rendered[0] if definition_rendered else definition,
        ]

        # Accounting-first section ordering: Formula -> Interpretation -> Example -> Cautions.
        # Non-accounting pages still use the same order for consistency.
        if formula_rendered:
            lines.extend(["", "---", "", "## Formula", ""])
            for item in formula_rendered:
                lines.append("```")
                lines.append(item)
                lines.append("```")
                lines.append("")

        if interpretation_rendered:
            lines.extend(["---", "", "## Interpretation", ""])
            for item in interpretation_rendered:
                lines.append(f"- {item}")
            lines.append("")

        if example_rendered:
            lines.extend(["---", "", "## Example", ""])
            for item in example_rendered:
                lines.append(f"- {item}")
            lines.append("")

        if caution_rendered:
            lines.extend(["---", "", "## Cautions", ""])
            for item in caution_rendered:
                lines.append(f"- {item}")
            lines.append("")

        if key_point_rendered:
            lines.extend(["---", "", "## Key Points", ""])
            for item in key_point_rendered:
                lines.append(f"- {item}")
            lines.append("")

        lines.extend(["---", "", "## Related Concepts"])
        related = related_map.get(concept, [])
        if related:
            for item in related:
                title = normalize_page_title(item)
                lines.append(f"- [[{title}]]")
        else:
            lines.append("- None")

        lines.extend(["", "---", "", "## Sources"])
        if combined_notes:
            lines.extend(combined_notes)
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