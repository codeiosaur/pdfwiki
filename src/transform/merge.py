from extract.fact_extractor import Fact
from transform.cluster import _concepts_are_similar
from transform.matching import tokenize_for_matching, has_antonym_conflict, is_sibling
from transform.normalize import normalize_concept_rules


_METRIC_SUFFIXES = {"ratio", "rate", "index", "coefficient"}


def _compact_concept_key(concept: str) -> str:
    """Return a punctuation-insensitive compact key for duplicate detection."""
    return "".join(tokenize_for_matching(concept))


def _metric_family_key(concept: str) -> tuple[str, ...]:
    """Return a family key that ignores trailing metric suffixes like ratio/rate."""
    tokens = tokenize_for_matching(concept)
    if not tokens:
        return (concept.lower().strip(),)
    if tokens[-1] in _METRIC_SUFFIXES and len(tokens) > 1:
        return tuple(tokens[:-1])
    return tuple(tokens)


def _is_metric_specific_label(concept: str) -> bool:
    tokens = tokenize_for_matching(concept)
    return bool(tokens) and tokens[-1] in _METRIC_SUFFIXES


def _choose_winner(left_label: str, left_facts: list[Fact], right_label: str, right_facts: list[Fact]) -> str:
    """Select canonical label by evidence first, then by concise naming."""
    def _score(label: str, facts: list[Fact]) -> tuple[int, int, int, int, str]:
        normalized = normalize_concept_rules(label)
        surface_words = [word for word in label.split() if word]
        return (
            len(facts),
            1 if normalized == label else 0,
            len(surface_words),
            -len(label),
            normalized.lower(),
        )

    left_score = _score(left_label, left_facts)
    right_score = _score(right_label, right_facts)
    return left_label if left_score >= right_score else right_label


def _dedupe_exact_token_keys(grouped: dict[str, list[Fact]]) -> dict[str, list[Fact]]:
    """
    Merge concepts that normalize to the exact same token sequence.

    This catches casing/punctuation variants early (for example FIFO vs Fifo).
    """
    merged: dict[str, list[Fact]] = {}
    token_key_to_label: dict[tuple[str, ...], str] = {}

    for concept, facts in grouped.items():
        token_key = tuple(tokenize_for_matching(concept))
        if not token_key:
            token_key = (concept.lower().strip(),)

        existing = token_key_to_label.get(token_key)
        if existing is None:
            token_key_to_label[token_key] = concept
            merged[concept] = list(facts)
            continue

        # Keep more common label; if tied, keep shorter label.
        existing_count = len(merged[existing])
        current_count = len(facts)
        if current_count > existing_count or (
            current_count == existing_count and len(concept) < len(existing)
        ):
            existing_facts = merged.pop(existing)
            merged[concept] = existing_facts + list(facts)
            token_key_to_label[token_key] = concept
        else:
            merged[existing].extend(facts)

    # Secondary compact-key merge catches spacing and join variants
    # like "Inventoryturnover Ratio" vs "Inventory Turnover Ratio".
    compact_key_to_label: dict[str, str] = {}
    collapsed: dict[str, list[Fact]] = {}
    for concept, facts in merged.items():
        compact_key = _compact_concept_key(concept)
        if not compact_key:
            compact_key = concept.lower().strip()
        existing = compact_key_to_label.get(compact_key)
        if existing is None:
            compact_key_to_label[compact_key] = concept
            collapsed[concept] = list(facts)
            continue

        winner = _choose_winner(existing, collapsed[existing], concept, list(facts))
        if winner == existing:
            collapsed[existing].extend(facts)
        else:
            existing_facts = collapsed.pop(existing)
            collapsed[concept] = existing_facts + list(facts)
            compact_key_to_label[compact_key] = concept

    return collapsed


def _merge_metric_suffix_variants(grouped: dict[str, list[Fact]]) -> dict[str, list[Fact]]:
    """Merge concept families that differ only by trailing metric suffixes.

    Example: "Inventory Turnover" and "Inventory Turnover Ratio".
    """
    merged: dict[str, list[Fact]] = {}
    family_to_label: dict[tuple[str, ...], str] = {}

    for concept, facts in grouped.items():
        family_key = _metric_family_key(concept)
        if not family_key:
            family_key = (concept.lower().strip(),)

        existing = family_to_label.get(family_key)
        if existing is None:
            family_to_label[family_key] = concept
            merged[concept] = list(facts)
            continue

        if has_antonym_conflict(existing, concept) or is_sibling(existing, concept):
            merged[concept] = list(facts)
            continue

        existing_metric_specific = _is_metric_specific_label(existing)
        current_metric_specific = _is_metric_specific_label(concept)
        if existing_metric_specific != current_metric_specific:
            winner = concept if current_metric_specific else existing
        else:
            winner = _choose_winner(existing, merged[existing], concept, list(facts))
        if winner == existing:
            merged[existing].extend(facts)
        else:
            existing_facts = merged.pop(existing)
            merged[concept] = existing_facts + list(facts)
            family_to_label[family_key] = concept

    return merged

def merge_similar_concepts(grouped: dict[str, list[Fact]]) -> dict[str, list[Fact]]:
    grouped = _dedupe_exact_token_keys(grouped)
    grouped = _merge_metric_suffix_variants(grouped)
    merged: dict[str, list[Fact]] = {}

    for concept, facts in grouped.items():
        merged_with_existing = False
        for existing_key in list(merged.keys()):
            if not _concepts_are_similar(concept, existing_key):
                continue

            # Keep more common name; if tied, keep shorter name.
            if len(merged[existing_key]) > len(facts):
                winner = existing_key
            elif len(merged[existing_key]) < len(facts):
                winner = concept
            else:
                winner = concept if len(concept) < len(existing_key) else existing_key

            if winner == existing_key:
                merged[existing_key].extend(facts)
            else:
                existing_facts = merged.pop(existing_key)
                merged[concept] = list(existing_facts) + list(facts)

            merged_with_existing = True
            break

        if not merged_with_existing:
            merged[concept] = list(facts)

    return merged