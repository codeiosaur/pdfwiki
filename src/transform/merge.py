from extract.fact_extractor import Fact
from transform.cluster import _concepts_are_similar
from transform.matching import tokenize_for_matching


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

    return merged

def merge_similar_concepts(grouped: dict[str, list[Fact]]) -> dict[str, list[Fact]]:
    grouped = _dedupe_exact_token_keys(grouped)
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