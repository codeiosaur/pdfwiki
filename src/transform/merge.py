from extract.fact_extractor import Fact
from transform.cluster import _concepts_are_similar

def merge_similar_concepts(grouped: dict[str, list[Fact]]) -> dict[str, list[Fact]]:
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