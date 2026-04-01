"""
Fact classification and definition selection.

Classifies extracted facts into types (definition, key_point, example,
instruction, formula, interpretation, caution) and scores definition
candidates to pick the best one for each concept page.
"""

import re
from typing import List, Optional


STOPWORDS = {"a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "with", "by"}


def _normalize_text_for_compare(text: str) -> str:
    """Normalize text for dedup/comparison: lowercase, strip punctuation, collapse whitespace."""
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


KEY_POINT_NOISE_MARKERS = {
    "figure",
    "table",
    "see page",
    "as shown",
    "wave of the future",
}

TEMPLATE_MARKERS = {
    r"\$\.\.\.",
    r"units at \$",
    r"\bFigure\s+\d+",
    r"^\w+\s+was\s+made\s+up\s+of\s+\.\.\.",
}

KEY_POINT_INSTRUCTION_MARKERS = {
    "calculate", "compute", "determine", "find", "assume",
    "use the following", "indicate the effect", "search", "submit",
    "locate", "prepare", "write", "show that", "demonstrate",
    "identify", "analyze", "examine", "verify", "check", "state",
    "consider",
}

SEMANTIC_FORMULA_MARKERS = {
    "=", "equation", "formula", "divided by", "multiplied by",
}

SEMANTIC_INTERPRETATION_MARKERS = {
    "means", "indicates", "shows", "suggests", "measures", "implies",
}

SEMANTIC_CAUTION_MARKERS = {
    "misleading", "manipulate", "risk", "caution", "however",
    "can be", "may be", "warning",
}


def _has_template_markers(text: str) -> bool:
    """Check if text contains incomplete template variable markers."""
    for pattern in TEMPLATE_MARKERS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _numeric_density(text: str) -> float:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return 0.0
    numeric = sum(1 for token in tokens if re.search(r"\d", token))
    return numeric / len(tokens)


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
    if len(re.findall(r"\S+", text)) < 6:
        return True
    return False


def _looks_like_formula(text: str) -> bool:
    lower = text.lower()
    if re.search(r"\S\s*=\s*\S", text):
        return True
    if any(marker in lower for marker in {"equation", "formula"}):
        return True
    has_ratio_word = "ratio" in lower
    has_compute_cue = any(op in lower for op in {"divided by", "multiplied by", "plus", "minus", "/", "÷", "×", "*"})
    if has_ratio_word and has_compute_cue:
        return True
    number_count = len(re.findall(r"\b\d+(?:[.,]\d+)?\b", text))
    has_operator = bool(re.search(r"[+\-/*=÷×]", text))
    if number_count >= 2 and has_operator:
        return True
    return False


def classify_fact(fact: str) -> str:
    """
    Classify a fact into one of:
    definition | key_point | example | instruction
    """
    text = fact.strip()
    if not text:
        return "key_point"

    lower = text.lower()

    if re.match(r"^there\s+(is|are)\b", lower):
        return "example"

    instruction_phrases = [
        "calculate", "provide", "use the following", "compute",
        "determine", "find", "apply", "solve", "search", "submit",
        "locate", "prepare", "write",
    ]
    if any(phrase in lower for phrase in instruction_phrases):
        return "instruction"

    if re.match(r"^(calculate|compute|determine|find|apply|use|solve|search|submit|locate|prepare|write)\b", lower):
        return "instruction"

    example_markers = ["for example", "e.g.", "scenario", "exercise", "suppose"]
    has_example_marker = any(marker in lower for marker in example_markers)
    number_count = len(re.findall(r"\b\d+([.,]\d+)?\b", lower))
    token_count = max(1, len(re.findall(r"[a-z0-9]+", lower)))
    numeric_density = number_count / token_count
    if has_example_marker or numeric_density >= 0.25:
        return "example"

    definition_markers = [" is ", " are ", " refers to ", " is defined as "]
    if any(marker in lower for marker in definition_markers):
        return "definition"

    return "key_point"


def classify_semantic_fact(fact: str) -> str:
    """
    Semantic classification used by the enhanced and wiki renderers.

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


def select_definition(concept: str, facts: List[str]) -> Optional[str]:
    """
    Select the best definition candidate using deterministic scoring rules.
    Returns None if all candidate scores are low.
    """
    if not facts:
        return None

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

        if any(term in lower for term in ["method", "process", "system", "measure", "ratio", "term", "account"]):
            score += 2

        if re.match(r"^(a|an|the)\s+[a-z]", lower):
            score += 1

        first_8_text = " ".join(re.findall(r"\S+", lower)[:8])
        if concept_lower in first_8_text:
            score += 2

        if concept_tokens:
            article_match = re.match(r"^(?:a|an|the)\s+(\w+)", lower)
            if article_match and article_match.group(1) in concept_tokens:
                score += 1

        if concept_lower in lower:
            score += 1
        elif concept_tokens and any(token in lower for token in concept_tokens):
            score += 1

        has_number = bool(re.search(r"[\d$%]", text))
        stat_context_terms = [
            "mean", "median", "variance", "standard deviation", "sample",
            "dataset", "confidence interval", "p-value", "correlation",
            "statistic", "ratio",
        ]
        has_stat_context = any(term in lower for term in stat_context_terms)
        if has_number and not has_stat_context:
            score -= 2

        if any(term in lower for term in ["calculate", "use the following", "assume"]):
            score -= 2

        if len(re.findall(r"\S+", text)) > 55:
            score -= 1

        vague_markers = [
            "key aspect", "important", "various", "etc",
            "kind of", "type of", "challenge of", "wave of the future",
        ]
        if any(marker in lower for marker in vague_markers):
            score -= 3

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

    if best_score < -2:
        return None
    return best_fact


def select_fallback_definition(
    concept: str,
    definitions: list[str],
    key_points: list[str],
    fact_contents: list[str],
) -> Optional[str]:
    """Try multiple strategies to find a usable definition."""
    selected = select_definition(concept, definitions)
    if selected is not None:
        return selected

    selected = select_definition(concept, key_points)
    if selected is not None:
        return selected

    non_instruction_non_example = [
        item for item in fact_contents if classify_fact(item) in {"definition", "key_point"}
    ]
    selected = select_definition(concept, non_instruction_non_example)
    if selected is not None:
        return selected

    if key_points:
        first = key_points[0].strip()
        if first:
            return first

    return None


def pick_best_definition(concept: str, definitions: list[str], key_points: list[str]) -> str:
    """Legacy definition picker used by the standard renderer."""
    if not definitions:
        if key_points:
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

        if " is defined as " in lower:
            pattern_rank, marker = 0, "defined"
        elif " refers to " in lower:
            pattern_rank, marker = 1, "refers"
        elif " is " in lower:
            pattern_rank, marker = 2, "is"
        elif " are " in lower:
            pattern_rank, marker = 3, "are"
        else:
            pattern_rank, marker = 4, ""

        concept_pos = lower.find(concept.lower())
        marker_pos = lower.find(marker) if marker else -1
        if concept_pos != -1 and marker_pos != -1:
            distance_penalty = abs(marker_pos - concept_pos)
            order_penalty = 0 if concept_pos <= marker_pos else 1
        else:
            distance_penalty = 999
            order_penalty = 1

        length_penalty = abs(len(fact_tokens) - 18)
        return (-overlap, pattern_rank + order_penalty, distance_penalty, length_penalty)

    return min(definitions, key=score)