"""Concept page workflow service.

This module contains pass-2 concept processing:
- per-concept retrieval and distillation,
- new page generation or merge,
- parallel execution and deterministic result collation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ConceptPageWorkflowSettings:
    context_max_chars: int
    retrieve_top_k: int
    extract_max_tokens: int
    write_max_tokens: int
    merge_max_tokens: int
    default_max_workers: int


@dataclass(frozen=True)
class ConceptPageWorkflowDeps:
    retrieve_context: Callable[[str], str]
    distill_context: Callable[[str, str], tuple[str, bool]]
    context_hash: Callable[[str], str]
    find_existing_page: Callable[[str], str | None]
    find_near_duplicate: Callable[[str, list[str]], str | None]
    extract_source_hash: Callable[[str], str | None]
    query_with_quality_retry: Callable[[str, int, str], str]
    parse_wiki_page: Callable[[str], tuple[str, str]]
    fix_wikilinks: Callable[[str, list[str], list[str]], str]
    find_related_concepts: Callable[[str, list[str]], list[str]]
    inject_active_wikilinks: Callable[[str, list[str], list[str]], str]
    postprocess_generated_content: Callable[[str], str]
    add_frontmatter: Callable[[str, str, list[str], str], str]
    upsert_source_hash_marker: Callable[[str, str], str]


@dataclass(frozen=True)
class ConceptPageWorkflowResult:
    wiki_pages: dict[str, str]
    merged_pages: dict[str, tuple[str, str]]
    skipped: list[str]
    concept_errors: list[tuple[int, str, str]]


def run_concept_page_workflow(
    *,
    concepts: list[str],
    subject: str,
    vault_state: dict,
    all_vault_pages: list[str],
    concept_graph: dict[str, set[str]],
    concept_names: str,
    wiki_prompt_template: str,
    merge_prompt_template: str,
    settings: ConceptPageWorkflowSettings,
    max_workers_arg: int | None,
    resolve_max_workers: Callable[[int | None, int], int],
    deps: ConceptPageWorkflowDeps,
) -> ConceptPageWorkflowResult:
    """Run pass-2 generation for all concepts with per-concept isolation."""

    def _process_single_concept(i: int, concept: str) -> dict:
        retrieved_context = deps.retrieve_context(concept)
        source_hash = deps.context_hash(retrieved_context)

        existing_path = deps.find_existing_page(concept)
        near_dup = None

        if existing_path is None and all_vault_pages:
            near_dup = deps.find_near_duplicate(concept, all_vault_pages)
            if near_dup:
                subject_pages = vault_state["pages"].get(subject, {})
                if near_dup in subject_pages:
                    existing_path = subject_pages[near_dup]

        existing_content = ""
        if existing_path is not None:
            existing_content = Path(existing_path).read_text(encoding="utf-8")
            existing_hash = deps.extract_source_hash(existing_content)
            if existing_hash == source_hash:
                return {
                    "index": i,
                    "concept": concept,
                    "kind": "skip",
                    "near_dup": near_dup,
                    "existing_path": existing_path,
                    "reason": "source-hash-match",
                }

        distilled_facts, extraction_skipped = deps.distill_context(concept, retrieved_context)

        if existing_path is None:
            page_prompt = (
                wiki_prompt_template
                .replace("{concept}", concept)
                .replace("{index}", concept_names)
                .replace("{concept_names}", concept_names)
                .replace("{facts}", distilled_facts)
                .replace("{text}", distilled_facts)
            )
            page_raw = deps.query_with_quality_retry(page_prompt, settings.write_max_tokens, "write")
            _, page_content = deps.parse_wiki_page(page_raw)
            page_content = deps.fix_wikilinks(page_content, concepts, all_vault_pages)
            graph_related = set(concept_graph.get(concept, set()))
            content_related = set(deps.find_related_concepts(page_content, concepts))
            active_related = sorted((graph_related | content_related) - {concept})
            page_content = deps.inject_active_wikilinks(page_content, active_related, concepts)
            page_content = deps.postprocess_generated_content(page_content)
            page_content = deps.fix_wikilinks(page_content, concepts, all_vault_pages)
            page_content = deps.add_frontmatter(concept, page_content, concepts, subject)
            return {
                "index": i,
                "concept": concept,
                "kind": "new",
                "near_dup": near_dup,
                "distilled_len": len(distilled_facts),
                "filename": concept,
                "content": page_content,
                "reason": "extract-skipped" if extraction_skipped else "generated",
            }

        merge_prompt = (
            merge_prompt_template
            .replace("{existing_content}", existing_content)
            .replace("{concept}", concept)
            .replace("{facts}", distilled_facts)
            .replace("{new_content}", distilled_facts)
            .replace("{source}", subject)
            .replace("{concept_names}", concept_names)
        )
        merge_raw = deps.query_with_quality_retry(merge_prompt, settings.merge_max_tokens, "write")

        if merge_raw.strip() == "NO_UPDATE":
            return {
                "index": i,
                "concept": concept,
                "kind": "skip",
                "near_dup": near_dup,
                "existing_path": existing_path,
                "reason": "no-update",
            }

        merge_raw = deps.fix_wikilinks(merge_raw, concepts, all_vault_pages)
        graph_related = set(concept_graph.get(concept, set()))
        content_related = set(deps.find_related_concepts(merge_raw, concepts))
        active_related = sorted((graph_related | content_related) - {concept})
        merge_raw = deps.inject_active_wikilinks(merge_raw, active_related, concepts)
        merge_raw = deps.postprocess_generated_content(merge_raw)
        merge_raw = deps.fix_wikilinks(merge_raw, concepts, all_vault_pages)
        merge_raw = deps.upsert_source_hash_marker(merge_raw, source_hash)
        return {
            "index": i,
            "concept": concept,
            "kind": "merge",
            "near_dup": near_dup,
            "existing_path": existing_path,
            "stem": Path(existing_path).stem,
            "content": merge_raw,
            "reason": "extract-skipped" if extraction_skipped else "merged",
        }

    concept_workers = resolve_max_workers(max_workers_arg, settings.default_max_workers)
    print(f"  Parallel concept workers: {concept_workers}", flush=True)

    results: list[dict] = []
    concept_errors: list[tuple[int, str, str]] = []
    total_concepts = len(concepts)
    completed = 0

    with ThreadPoolExecutor(max_workers=concept_workers) as executor:
        futures = {}
        for i, concept in enumerate(concepts):
            print(f"  [{i+1}/{total_concepts}] QUEUED: {concept}", flush=True)
            future = executor.submit(_process_single_concept, i, concept)
            futures[future] = (i, concept)

        for future in as_completed(futures):
            i, concept = futures[future]
            try:
                item = future.result()
                results.append(item)
                completed += 1
                kind = item.get("kind", "unknown").upper()
                near_dup = item.get("near_dup")
                if near_dup:
                    print(
                        f"  [{i+1}/{total_concepts}] DONE ({completed}/{total_concepts}) "
                        f"{kind}: {concept} (near-dup: {near_dup})",
                        flush=True,
                    )
                else:
                    print(
                        f"  [{i+1}/{total_concepts}] DONE ({completed}/{total_concepts}) "
                        f"{kind}: {concept}",
                        flush=True,
                    )
            except Exception as exc:  # pragma: no cover - runtime path
                concept_errors.append((i, concept, str(exc)))
                completed += 1
                print(f"  [{i+1}/{total_concepts}] FAILED ({completed}/{total_concepts}): {concept}", flush=True)

    wiki_pages: dict[str, str] = {}
    merged_pages: dict[str, tuple[str, str]] = {}
    skipped: list[str] = []

    for item in sorted(results, key=lambda x: x["index"]):
        kind = item["kind"]
        if kind == "new":
            wiki_pages[item["filename"]] = item["content"]
        elif kind == "merge":
            merged_pages[item["stem"]] = (item["existing_path"], item["content"])
        else:
            skipped.append(item["concept"])

    if concept_errors:
        print(f"  [warn] {len(concept_errors)} concepts failed during parallel processing:", flush=True)
        for i, concept, err in sorted(concept_errors, key=lambda x: x[0]):
            print(f"    - [{i+1}/{len(concepts)}] {concept}: {err}", flush=True)

    return ConceptPageWorkflowResult(
        wiki_pages=wiki_pages,
        merged_pages=merged_pages,
        skipped=skipped,
        concept_errors=concept_errors,
    )
