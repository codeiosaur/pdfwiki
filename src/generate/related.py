"""
Related concept discovery and citation formatting.

Provides token-overlap and chunk-co-occurrence methods for finding
related concepts, plus footnote citation helpers for the legacy renderers.
"""

import re

from extract.fact_extractor import Fact
from transform.matching import has_antonym_conflict, is_sibling
from generate.titles import concept_tokens, normalize_page_title


def build_related_concepts(
    concepts: list[str],
    max_related: int = 3,
    exclude_siblings: bool = True,
    exclude_antonyms: bool = True,
) -> dict[str, list[str]]:
    """Build related concepts map using token overlap (legacy renderers)."""
    related: dict[str, list[str]] = {}
    token_map = {c: concept_tokens(c) for c in concepts}

    for concept in concepts:
        current = token_map[concept]
        scored: list[tuple[int, float, int, str]] = []

        for candidate in concepts:
            if candidate == concept:
                continue
            if exclude_siblings and is_sibling(concept, candidate):
                continue
            if exclude_antonyms and has_antonym_conflict(concept, candidate):
                continue
            overlap = len(current.intersection(token_map[candidate]))
            if overlap == 0:
                continue
            candidate_tokens = token_map[candidate]
            union = len(current.union(candidate_tokens))
            jaccard = overlap / union if union else 0.0
            scored.append((overlap, jaccard, len(candidate_tokens), candidate))

        scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        related[concept] = [name for _, _, _, name in scored[:max_related]]

    return related


def build_related_concepts_by_chunks(
    concept: str,
    concept_chunks: dict[str, set[str]],
    all_concepts: list[str],
    max_related: int = 8,
    grouped: dict | None = None,
    exclude_siblings: bool = True,
    exclude_antonyms: bool = True,
) -> list[str]:
    """
    Build related concepts by source-chunk co-occurrence (domain-agnostic).

    Concepts extracted from the same text chunks are inherently related.
    Falls back to token overlap when chunk data is sparse.

    max_related: maximum number of related concepts to return (default 8).
    grouped: optional fact group map; when supplied, candidates with fewer than
             2 facts are skipped (stubs should not appear as related concepts).

    An IDF-style weight penalises concepts that appear across many chunks —
    ubiquitous concepts score lower so more specific neighbours surface first.
    """
    my_chunks = concept_chunks.get(concept, set())
    total_chunks = max(1, sum(len(v) for v in concept_chunks.values()))
    scored: list[tuple[float, int, str]] = []

    for candidate in all_concepts:
        if candidate == concept:
            continue

        if exclude_siblings and is_sibling(concept, candidate):
            continue
        if exclude_antonyms and has_antonym_conflict(concept, candidate):
            continue

        if grouped is not None and len(grouped.get(candidate, [])) < 2:
            continue

        candidate_chunks = concept_chunks.get(candidate, set())
        shared_chunks = len(my_chunks.intersection(candidate_chunks))
        token_overlap = len(
            concept_tokens(concept).intersection(concept_tokens(candidate))
        )

        if shared_chunks > 0 or token_overlap > 0:
            candidate_chunk_count = len(candidate_chunks)
            idf_weight = 1.0 / (1 + candidate_chunk_count / total_chunks)
            score = (shared_chunks * 10 + token_overlap) * idf_weight
            scored.append((score, token_overlap, candidate))

    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [name for _, _, name in scored[:max_related]]


def fact_sources(facts: list[Fact]) -> dict[str, list[str]]:
    """Map fact content to source chunk IDs (used by legacy citation system)."""
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


def citation_suffixes(
    items: list[str],
    text_to_sources: dict[str, list[str]],
    source_key_to_note_index: dict[tuple[str, ...], int],
    start_index: int,
) -> tuple[list[str], list[str], int]:
    """Append footnote citations to items (used by legacy renderers)."""
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


def looks_like_uuid(s: str) -> bool:
    """Check if a string looks like a UUID (hex with dashes)."""
    return bool(re.match(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        s.strip().lower()
    ))


def all_sources_are_uuids(text_to_sources: dict[str, list[str]]) -> bool:
    """Return True when every source ID in the map looks like a UUID.

    Used by renderers to decide whether to suppress the Sources section and
    inline footnote references.  When chunks only have UUID IDs (no page
    numbers / filenames yet), showing them adds noise without value.
    """
    for source_ids in text_to_sources.values():
        for sid in source_ids:
            if sid and sid != "unknown" and not looks_like_uuid(sid):
                return False
    return True