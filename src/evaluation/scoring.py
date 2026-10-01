"""Deterministic rubric scoring for a completed pipeline run.

The scorer deliberately evaluates observable contracts: source-ID preservation,
concept recovery, rejected concepts, and Markdown structure. It does not judge
whether two model responses use identical prose.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any

from extract.fact_extractor import Fact
from evaluation.fixtures import EvaluationCase, GoldConcept
from transform.filter import is_valid_concept


def _normalize_concept(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _matches_gold(concept: str, gold: GoldConcept) -> bool:
    normalized = _normalize_concept(concept)
    candidates = (gold.name, *gold.aliases)
    return normalized in {_normalize_concept(candidate) for candidate in candidates}


def _has_broken_wikilink(page: str) -> bool:
    complete = re.findall(r"\[\[[^\]\n]+\]\]", page)
    return page.count("[[") != len(complete) or page.count("]]") != len(complete)


def _page_has_heading(page: str) -> bool:
    return bool(re.search(r"^#\s+\S", page, re.MULTILINE))


def score_case(
    case: EvaluationCase,
    facts: Iterable[Fact],
    pages: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Score one run against one fixed rubric and return JSON-safe metrics."""
    fact_list = list(facts)
    pages = pages or {}
    allowed_source_ids = {chunk.id for chunk in case.chunks} | {
        f"{chunk.source}::{chunk.id}" for chunk in case.chunks
    }
    concepts = sorted({fact.concept.strip() for fact in fact_list if fact.concept.strip()})

    source_valid = sum(
        fact.source_chunk_id.strip() in allowed_source_ids
        for fact in fact_list
        if fact.source_chunk_id
    )
    source_missing = sum(not fact.source_chunk_id for fact in fact_list)
    source_total = len(fact_list)
    source_rate = source_valid / source_total if source_total else None

    matched_gold = [gold for gold in case.expected_concepts if any(_matches_gold(concept, gold) for concept in concepts)]
    gold_recall = len(matched_gold) / len(case.expected_concepts) if case.expected_concepts else None
    forbidden_normalized = {_normalize_concept(item) for item in case.forbidden_concepts}
    forbidden_found = [concept for concept in concepts if _normalize_concept(concept) in forbidden_normalized]
    confirmed_false_positives = concepts if case.noise_only else forbidden_found
    unreviewed_concepts = [
        concept for concept in concepts
        if not any(_matches_gold(concept, gold) for gold in case.expected_concepts)
        and concept not in confirmed_false_positives
    ]
    invalid_concepts = [concept for concept in concepts if not is_valid_concept(concept)]

    page_matches: dict[str, str] = {}
    for gold in case.expected_concepts:
        for title, page in pages.items():
            if _matches_gold(title, gold):
                page_matches[gold.name] = page
                break
    page_recall = len(page_matches) / len(case.expected_concepts) if case.expected_concepts else None
    missing_required_terms = {
        gold.name: [term for term in gold.required_terms if term not in page_matches.get(gold.name, "").lower()]
        for gold in case.expected_concepts
        if gold.name in page_matches
    }
    missing_required_terms = {name: terms for name, terms in missing_required_terms.items() if terms}

    page_values = list(pages.values())
    broken_wikilinks = sum(_has_broken_wikilink(page) for page in page_values)
    heading_pages = sum(_page_has_heading(page) for page in page_values)

    return {
        "case_id": case.id,
        "metrics": {
            "fact_count": source_total,
            "source_id_preservation_rate": source_rate,
            "missing_source_id_count": source_missing,
            "gold_concept_recall": gold_recall,
            "confirmed_false_positive_count": len(confirmed_false_positives),
            "unreviewed_concept_count": len(unreviewed_concepts),
            "invalid_concept_count": len(invalid_concepts),
            "forbidden_concept_count": len(forbidden_found),
            "noise_rejection_pass": not (fact_list or pages) if case.noise_only else None,
            "page_recall": page_recall,
            "page_heading_rate": heading_pages / len(page_values) if page_values else None,
            "broken_wikilink_count": broken_wikilinks,
            "missing_required_term_count": sum(len(terms) for terms in missing_required_terms.values()),
        },
        "details": {
            "matched_gold_concepts": [gold.name for gold in matched_gold],
            "confirmed_false_positive_concepts": confirmed_false_positives,
            "unreviewed_concepts": unreviewed_concepts,
            "invalid_concepts": invalid_concepts,
            "forbidden_concepts": forbidden_found,
            "missing_required_terms": missing_required_terms,
        },
    }


def score_suite(
    cases: Iterable[EvaluationCase],
    results: Mapping[str, tuple[Iterable[Fact], Mapping[str, str] | None]],
) -> dict[str, Any]:
    """Score a complete fixture corpus and return macro-average run metrics."""
    case_list = list(cases)
    missing = [case.id for case in case_list if case.id not in results]
    if missing:
        raise ValueError(f"Missing evaluation results for cases: {', '.join(missing)}")
    case_scores = [score_case(case, *results[case.id]) for case in case_list]
    if not case_scores:
        raise ValueError("Cannot score an empty evaluation suite")

    metric_names = (
        "source_id_preservation_rate",
        "gold_concept_recall",
        "page_recall",
        "page_heading_rate",
        "noise_rejection_pass",
    )
    summary = {}
    scored_case_counts = {}
    for name in metric_names:
        values = [score["metrics"][name] for score in case_scores if score["metrics"][name] is not None]
        summary[name] = sum(values) / len(values) if values else None
        scored_case_counts[name] = len(values)
    summary["scored_case_counts"] = scored_case_counts
    summary["case_count"] = len(case_scores)
    summary["confirmed_false_positive_count"] = sum(score["metrics"]["confirmed_false_positive_count"] for score in case_scores)
    summary["broken_wikilink_count"] = sum(score["metrics"]["broken_wikilink_count"] for score in case_scores)
    summary["missing_required_term_count"] = sum(score["metrics"]["missing_required_term_count"] for score in case_scores)
    return {"summary": summary, "cases": case_scores}
