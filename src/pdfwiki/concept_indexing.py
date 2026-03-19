"""Concept indexing application service.

This module owns pass-1 concept extraction flow:
- build index output from chapter text,
- parse concepts,
- apply evidence filter,
- dedupe near-duplicates within the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ConceptIndexResult:
    concepts: list[str]
    index_text: str
    dropped_concepts: list[str]
    deduped_pairs: list[tuple[str, str]]


def run_concept_indexing(
    *,
    chapters: list[dict],
    all_chunks: list[str],
    build_index: Callable[[list[dict]], tuple[list[str], str]],
    filter_concepts_with_evidence: Callable[[list[str], list[str]], tuple[list[str], list[str]]],
    dedupe_concepts_for_run: Callable[[list[str]], tuple[list[str], list[tuple[str, str]]]],
) -> ConceptIndexResult:
    """Run pass-1 concept indexing and quality gating.

    The caller supplies concrete adapters so this service remains framework-agnostic
    and easy to test in isolation.
    """
    concepts, index_text = build_index(chapters)
    concepts, dropped_concepts = filter_concepts_with_evidence(concepts, all_chunks)
    concepts, deduped_pairs = dedupe_concepts_for_run(concepts)

    return ConceptIndexResult(
        concepts=concepts,
        index_text=index_text,
        dropped_concepts=dropped_concepts,
        deduped_pairs=deduped_pairs,
    )
