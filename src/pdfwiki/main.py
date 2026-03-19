"""pdf_to_notes main orchestration module.

This file contains the complete high-level workflow and most quality/safety
guardrails. The architecture is intentionally two-pass:

1) Pass 1 (index pass): ask the model for a concept list over broad content.
2) Pass 2 (concept pass): process each concept independently (parallel-safe).

Why two passes:
- Pass 1 creates structure from noisy source text.
- Pass 2 narrows each generation call to targeted context for better quality.

Why this module is large:
- Most production behavior is controlled here so profile tuning, dedupe,
    skip logic, and generation retries stay visible in one place.

Entry points:
- run_cli(...): CLI interface and argument validation.
- process_pdf(...): single-PDF pipeline execution.
"""

import re
import os
import argparse
import sys
import hashlib
from pathlib import Path
from difflib import get_close_matches

# Allow running this file directly: `python src/pdfwiki/main.py ...`
# by ensuring `src/` is on sys.path for `import pdfwiki...`.
if __package__ in (None, ""):
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from pdfwiki.extractor import extract_text, split_into_chapters, chunk_text, chunk_by_page, smart_chunk
from pdfwiki.retriever import retrieve_chunks, retrieve_ranked_chunks, retrieve_ranked_chunks_with_scores, limit_context, find_related_concepts, build_concept_graph, _compute_adaptive_context_size
from pdfwiki.ai_client import query, extract_facts, set_provider, get_provider
from pdfwiki.writer import write_wiki, write_flashcards, write_cheatsheet
from pdfwiki.vault import load_vault_state, find_existing_page
from pdfwiki.concept_indexing import run_concept_indexing
from pdfwiki.concept_page_workflow import (
    ConceptPageWorkflowDeps,
    ConceptPageWorkflowSettings,
    run_concept_page_workflow,
)
from pdfwiki.study_outputs import StudyOutputDeps, maybe_regenerate_moc, generate_study_aids
from pdfwiki.content_processing import (
    parse_index,
    parse_wiki_page,
    build_alias_map,
    fix_wikilinks,
    add_frontmatter,
    inject_active_wikilinks,
    detect_subject as cp_detect_subject,
    chapter_summary_text as cp_chapter_summary_text,
    context_hash as cp_context_hash,
    extract_source_hash as cp_extract_source_hash,
    upsert_source_hash_marker as cp_upsert_source_hash_marker,
    postprocess_generated_content as cp_postprocess_generated_content,
    query_with_quality_retry as cp_query_with_quality_retry,
)
from pdfwiki.concept_quality import (
    normalize_concept as cq_normalize_concept,
    concept_tokens as cq_concept_tokens,
    concept_aliases as cq_concept_aliases,
    has_modifier_conflict as cq_has_modifier_conflict,
    is_token_level_typo_variant as cq_is_token_level_typo_variant,
    is_safe_near_duplicate as cq_is_safe_near_duplicate,
    find_near_duplicate as cq_find_near_duplicate,
    concept_has_source_evidence as cq_concept_has_source_evidence,
    filter_concepts_with_evidence as cq_filter_concepts_with_evidence,
    dedupe_concepts_for_run as cq_dedupe_concepts_for_run,
)


PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
if not PROMPTS_DIR.exists():
    PROMPTS_DIR = Path(__file__).parent / "prompts"

RUN_PROFILE_ALIASES = {
    "balanced": "hybrid",
}

RUN_PROFILE_SETTINGS: dict[str, dict[str, int]] = {
    # Fastest local iteration for most machines.
    "speed": {
        "context_max_chars": 2200,
        "extract_max_tokens": 280,
        "write_max_tokens": 1000,
        "merge_max_tokens": 650,
        "retrieve_top_k": 2,
        "default_max_workers": 4,
    },
    # Default tradeoff profile.
    "hybrid": {
        "context_max_chars": 3000,
        "extract_max_tokens": 360,
        "write_max_tokens": 1000,
        "merge_max_tokens": 700,
        "retrieve_top_k": 2,
        "default_max_workers": 3,
    },
    # Highest quality, slower throughput.
    "quality": {
        "context_max_chars": 4800,
        "extract_max_tokens": 550,
        "write_max_tokens": 1400,
        "merge_max_tokens": 900,
        "retrieve_top_k": 3,
        "default_max_workers": 2,
    },
}

def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def detect_subject(raw_stem: str, concepts: list[str] | None = None, batch_mode: bool = False) -> str:
    return cp_detect_subject(raw_stem, concepts, batch_mode, query_fn=query)


def _chapter_summary_text(chapters: list[dict]) -> str:
    return cp_chapter_summary_text(chapters, chunk_text_fn=chunk_text)


def _concept_has_source_evidence(concept: str, chunks: list[str]) -> bool:
    return cq_concept_has_source_evidence(concept, chunks)


def _filter_concepts_with_evidence(concepts: list[str], chunks: list[str]) -> tuple[list[str], list[str]]:
    return cq_filter_concepts_with_evidence(concepts, chunks)


def _context_hash(text: str) -> str:
    return cp_context_hash(text)


def _extract_source_hash(content: str) -> str | None:
    return cp_extract_source_hash(content)


def _upsert_source_hash_marker(content: str, source_hash: str) -> str:
    return cp_upsert_source_hash_marker(content, source_hash)


def _postprocess_generated_content(content: str) -> str:
    return cp_postprocess_generated_content(content)


def _query_with_quality_retry(prompt: str, max_tokens: int, task: str = "write") -> str:
    return cp_query_with_quality_retry(prompt, max_tokens, task, query_fn=query)


def _normalize_concept(name: str) -> str:
    return cq_normalize_concept(name)


def find_near_duplicate(concept: str, vault_pages: list[str], cutoff: float = 0.82) -> str | None:
    return cq_find_near_duplicate(concept, vault_pages, cutoff=cutoff)


def _dedupe_concepts_for_run(concepts: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    return cq_dedupe_concepts_for_run(concepts)


def _build_index(chapters: list[dict]) -> tuple[list[str], str]:
    """Run Pass 1 indexing and return parsed concept names plus raw model output.

    Inputs are chapter-level text blocks joined into one index prompt. We keep the
    raw index text because downstream steps (for example MOC relationship hints)
    may still need unparsed context.

    Raises:
        ValueError: when no valid concepts survive parsing/cleanup.
    """
    index_text_input = "\n\n---\n\n".join(ch["content"] for ch in chapters)
    index_prompt = load_prompt("index").replace("{text}", index_text_input)
    index_raw = query(index_prompt, task="cheap", max_tokens=4000)
    concepts, index_text = parse_index(index_raw)
    if not concepts:
        raise ValueError(
            "No concepts were parsed from the index response. "
            "Check the prompt/output format before continuing."
        )
    return concepts, index_text


def _get_subject(raw_stem: str, concepts: list[str], subject_override: str, batch_mode: bool) -> str:
    if subject_override:
        print(f"  Subject (from --subject flag): \"{subject_override}\"")
        return subject_override
    return detect_subject(raw_stem, concepts, batch_mode=batch_mode)


def _collect_vault_pages(vault_state: dict) -> list[str]:
    return [
        page
        for pages in vault_state["pages"].values()
        for page in pages.keys()
    ]


def _retrieve_concept_context(
    all_chunks: list[str],
    concept: str,
    concepts: list[str],
    max_chars: int = 4000,
    retrieve_top_k: int = 2,
) -> str:
    """Build per-concept retrieval context used by generation.

    Pipeline inside this helper:
    1) rank chunks with lexical + heuristic scoring,
    2) adapt context size based on confidence score,
    3) enforce character budget with deterministic truncation.

    This helper is intentionally side-effect free so it is safe to call in
    parallel worker threads.
    """
    other_concepts = [c for c in concepts if c != concept]
    
    # Get ranked chunks with scores for intelligent context sizing
    ranked_with_scores = retrieve_ranked_chunks_with_scores(
        all_chunks,
        concept=concept,
        related_concepts=other_concepts[:5],
        top_k=retrieve_top_k,
    )
    
    # Calculate average relevance score to adjust context size
    if ranked_with_scores:
        avg_score = sum(score for _, score in ranked_with_scores) / len(ranked_with_scores)
        adaptive_max_chars = _compute_adaptive_context_size(
            avg_score=avg_score,
            base_max_chars=max_chars,
        )
    else:
        adaptive_max_chars = max_chars
    
    ranked_chunks = [chunk for chunk, _ in ranked_with_scores]
    return limit_context(ranked_chunks, max_chars=adaptive_max_chars)


def _distill_concept_context(
    concept: str,
    relevant_text: str,
    extract_max_tokens: int = 450,
    skip_extract_min_chars: int = 900,
) -> tuple[str, bool]:
    """Optionally distill facts from retrieved context.

    Returns:
        tuple[str, bool]:
            - distilled text to feed writer/merge prompts
            - whether extraction was skipped

    Skip behavior:
    - very short contexts skip extraction to save latency and avoid over-
      compressing already concise evidence.
    """

    # Fast path: short contexts don't benefit much from another model pass.
    if len(relevant_text) < skip_extract_min_chars:
        return relevant_text, True

    facts_text = extract_facts(concept, relevant_text, max_tokens=extract_max_tokens).strip()
    distilled = facts_text or relevant_text
    return distilled, not bool(facts_text)


# --- Main pipeline ---

def _resolve_run_profile(profile: str | None) -> tuple[str, dict[str, int]]:
    """Resolve run profile from CLI arg/env/default.

    Precedence:
    1) explicit function argument
    2) PDF_TO_NOTES_PROFILE env var
    3) hardcoded default ('hybrid')
    """
    raw = (profile or os.environ.get("PDF_TO_NOTES_PROFILE", "hybrid")).strip().lower()
    normalized = RUN_PROFILE_ALIASES.get(raw, raw)
    if normalized not in RUN_PROFILE_SETTINGS:
        print(f"  [warn] invalid profile {raw!r}; using 'hybrid'", flush=True)
        normalized = "hybrid"
    return normalized, RUN_PROFILE_SETTINGS[normalized]


def _resolve_max_workers(max_workers: int | None, default_workers: int) -> int:
    """Resolve concept-processing worker count from arg/env/default.

    Precedence:
    1) explicit CLI value
    2) PDF_TO_NOTES_MAX_WORKERS env var
    3) profile-derived default
    """
    if max_workers is not None:
        return max(1, int(max_workers))

    env_raw = os.environ.get("PDF_TO_NOTES_MAX_WORKERS", "").strip()
    if env_raw:
        try:
            return max(1, int(env_raw))
        except ValueError:
            print(f"  [warn] invalid PDF_TO_NOTES_MAX_WORKERS={env_raw!r}; using default", flush=True)

    return max(1, default_workers)


def process_pdf(
    pdf_path: str,
    output_dir: str,
    subject_override: str = "",
    batch_mode: bool = False,
    max_workers: int | None = None,
    profile: str | None = None,
):
    """Run the full PDF-to-vault pipeline for one PDF.

    High-level stages:
    1) text extraction and smart chunking
    2) concept indexing + quality filters
    3) per-concept generation/merge (parallel)
    4) MOC refresh (only when new pages are created)
    5) flashcards and cheatsheet generation

    Determinism and safety notes:
    - per-concept work runs in parallel but results are applied in original
      concept order to keep output stable across runs.
    - each concept is isolated; failures are recorded and do not stop others.
    - unchanged existing pages are skipped via source-context hash.
    """
    raw_stem = Path(pdf_path).stem
    print(f"\n{'='*50}", flush=True)
    print(f"Processing: {raw_stem}", flush=True)
    print(f"{'='*50}", flush=True)

    # 1. Extract text
    print("\n[1/6] Extracting text from PDF...", flush=True)
    full_text = extract_text(pdf_path)

    profile_name, profile_settings = _resolve_run_profile(profile)
    print(
        f"  Run profile: {profile_name} "
        f"(ctx={profile_settings['context_max_chars']}, "
        f"extract_tokens={profile_settings['extract_max_tokens']}, "
        f"write_tokens={profile_settings['write_max_tokens']}, "
        f"merge_tokens={profile_settings['merge_max_tokens']}, "
        f"top_k={profile_settings['retrieve_top_k']})",
        flush=True,
    )

    # 2. Split into chapters and chunk each one
    print("\n[2/6] Splitting into chapters and chunking...", flush=True)
    chapters = split_into_chapters(full_text)
    # Smart chunking auto-selects page/headings/paragraph/size strategy.
    all_chunks = smart_chunk(full_text, pages_per_chunk=2)
    print(f"  Total chunks: {len(all_chunks)}", flush=True)

    # For flashcards/cheatsheet: compressed summary (first chunk per chapter)
    summary_text = _chapter_summary_text(chapters)

    # 3. Build concept index — Pass 1
    print("\n[3/6] Building concept index (Pass 1)...", flush=True)
    index_result = run_concept_indexing(
        chapters=chapters,
        all_chunks=all_chunks,
        build_index=_build_index,
        filter_concepts_with_evidence=_filter_concepts_with_evidence,
        dedupe_concepts_for_run=_dedupe_concepts_for_run,
    )
    concepts = index_result.concepts
    index_text = index_result.index_text
    dropped_concepts = index_result.dropped_concepts
    if dropped_concepts:
        print(
            f"  [quality] dropped {len(dropped_concepts)} low-evidence concepts: "
            + ", ".join(dropped_concepts[:8])
            + ("..." if len(dropped_concepts) > 8 else ""),
            flush=True,
        )
    deduped_pairs = index_result.deduped_pairs
    if deduped_pairs:
        preview = ", ".join(f"{c} -> {k}" for c, k in deduped_pairs[:8])
        if len(deduped_pairs) > 8:
            preview += ", ..."
        print(f"  [quality] deduped {len(deduped_pairs)} near-duplicate concepts: {preview}", flush=True)
    print(f"  Found {len(concepts)} concepts: {', '.join(concepts)}", flush=True)

    subject = _get_subject(raw_stem, concepts, subject_override, batch_mode)

    # Build concept graph for active linking (system decides what should link)
    print("\n[Pass 1.5] Building concept relationship graph...", flush=True)
    concept_graph = build_concept_graph(concepts, all_chunks)
    print(f"  Concept relationships mapped", flush=True)

    # 4. Generate wiki pages — Pass 2 (incremental: new / merge / skip)
    print(f"\n[4/6] Processing {len(concepts)} concepts (Pass 2)...", flush=True)
    vault_state = load_vault_state(output_dir)
    all_vault_pages = _collect_vault_pages(vault_state)
    concept_names = "\n".join(f"- {c}" for c in concepts)
    wiki_prompt_template = load_prompt("wiki")
    merge_prompt_template = load_prompt("merge")
    if profile_name == "speed":
        wiki_prompt_template += "\n\nAdditional constraints for speed profile:\n- Do NOT generate Mermaid diagrams.\n- Prefer plain markdown sections and bullets."

    pass2_settings = ConceptPageWorkflowSettings(
        context_max_chars=profile_settings["context_max_chars"],
        retrieve_top_k=profile_settings["retrieve_top_k"],
        extract_max_tokens=profile_settings["extract_max_tokens"],
        write_max_tokens=profile_settings["write_max_tokens"],
        merge_max_tokens=profile_settings["merge_max_tokens"],
        default_max_workers=profile_settings["default_max_workers"],
    )
    pass2_deps = ConceptPageWorkflowDeps(
        retrieve_context=lambda concept: _retrieve_concept_context(
            all_chunks,
            concept,
            concepts,
            max_chars=profile_settings["context_max_chars"],
            retrieve_top_k=profile_settings["retrieve_top_k"],
        ),
        distill_context=lambda concept, text: _distill_concept_context(
            concept,
            text,
            extract_max_tokens=profile_settings["extract_max_tokens"],
        ),
        context_hash=_context_hash,
        find_existing_page=lambda concept: find_existing_page(concept, subject, vault_state),
        find_near_duplicate=find_near_duplicate,
        extract_source_hash=_extract_source_hash,
        query_with_quality_retry=_query_with_quality_retry,
        parse_wiki_page=parse_wiki_page,
        fix_wikilinks=fix_wikilinks,
        find_related_concepts=find_related_concepts,
        inject_active_wikilinks=inject_active_wikilinks,
        postprocess_generated_content=_postprocess_generated_content,
        add_frontmatter=add_frontmatter,
        upsert_source_hash_marker=_upsert_source_hash_marker,
    )
    pass2_result = run_concept_page_workflow(
        concepts=concepts,
        subject=subject,
        vault_state=vault_state,
        all_vault_pages=all_vault_pages,
        concept_graph=concept_graph,
        concept_names=concept_names,
        wiki_prompt_template=wiki_prompt_template,
        merge_prompt_template=merge_prompt_template,
        settings=pass2_settings,
        max_workers_arg=max_workers,
        resolve_max_workers=_resolve_max_workers,
        deps=pass2_deps,
    )

    wiki_pages = pass2_result.wiki_pages
    merged_pages = pass2_result.merged_pages
    skipped = pass2_result.skipped

    # Write new pages
    if wiki_pages:
        write_wiki(output_dir, wiki_pages, subject=subject)

    # Write merged pages (overwrite existing files with updated content)
    for stem, (path, content_) in merged_pages.items():
        Path(path).write_text(content_, encoding="utf-8")
        print(f"  Updated: {Path(path).name}")

    # Summary
    print(f"\n  Summary: {len(wiki_pages)} new, "
          f"{len(merged_pages)} merged, {len(skipped)} skipped")
    added_new = len(wiki_pages) > 0

    output_deps = StudyOutputDeps(
        load_vault_state=load_vault_state,
        load_prompt=load_prompt,
        query=query,
        parse_wiki_page=parse_wiki_page,
        add_frontmatter=add_frontmatter,
        write_flashcards=write_flashcards,
        write_cheatsheet=write_cheatsheet,
    )
    maybe_regenerate_moc(
        added_new=added_new,
        output_dir=output_dir,
        subject=subject,
        concepts=concepts,
        index_text=index_text,
        deps=output_deps,
    )
    generate_study_aids(
        output_dir=output_dir,
        subject=subject,
        summary_text=summary_text,
        deps=output_deps,
    )

    print(f"\nDone! Output written to: {output_dir}/")
    print(f"Wiki pages: {len(wiki_pages)}")


def run_cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint used by both direct execution and tests.

    Returns process-style exit code:
    - 0 on success
    - nonzero on argument/runtime errors
    """
    parser = argparse.ArgumentParser(
        description="Generate Obsidian wiki, flashcards, and cheatsheet from PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single PDF, auto-detect subject
  python main.py Cryptography_pptx.pdf --vault ./vault

  # Single PDF, override subject
  python main.py Lecture_3.pdf --vault ./vault --subject Cryptography

  # Multiple PDFs (auto batch mode)
  python main.py Cryptography.pdf AccessControl.pdf --vault ./vault

  # Glob (shell expands *.pdf before passing to argparse)
  python main.py *.pdf --vault ./vault
        """
    )
    parser.add_argument("pdfs", nargs="+", help="One or more PDF files to process")
    parser.add_argument("--vault", "-v", default=None,
                        help="Obsidian vault root. Defaults to a 'vault/' folder "
                             "next to the first PDF if not specified.")
    parser.add_argument("--subject", "-s", default=None,
                        help="Subject override. Single PDF only.")
    parser.add_argument("--batch", action="store_true",
                        help="Never prompt. Unknown subjects go to Unsorted/.")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "ollama"],
        default=None,
        help=(
            "Model provider override for this run. "
            "Defaults to PDF_TO_NOTES_PROVIDER from environment."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Maximum parallel workers for concept pass (Pass 2). "
            "Defaults to PDF_TO_NOTES_MAX_WORKERS or an auto value."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=["speed", "quality", "hybrid"],
        default=None,
        help=(
            "Performance/quality profile. "
            "Also configurable via PDF_TO_NOTES_PROFILE."
            "'balanced' is an alias for 'hybrid'."
        ),
    )
    args = parser.parse_args(argv)

    if args.subject and len(args.pdfs) > 1:
        parser.error("--subject can only be used with a single PDF")

    if args.provider:
        set_provider(args.provider)
        print(f"Provider override: {get_provider()}")

    # Default vault: sibling folder next to the first PDF
    # e.g. ~/Documents/Cryptography.pdf → ~/Documents/vault/
    vault = args.vault or str(Path(args.pdfs[0]).parent / "vault")
    if not args.vault:
        print(f"No --vault specified. Using: {vault}")

    batch = args.batch or len(args.pdfs) > 1
    failed = []

    for pdf_path in args.pdfs:
        try:
            process_pdf(pdf_path, vault,
                        subject_override=args.subject if len(args.pdfs) == 1 else "",
                        batch_mode=batch,
                        max_workers=args.max_workers,
                        profile=args.profile)
        except Exception as e:
            print(f"\nERROR processing {pdf_path}: {e}")
            failed.append((pdf_path, str(e)))

    if failed:
        print(f"\n{'='*50}")
        print(f"FAILED ({len(failed)}/{len(args.pdfs)} PDFs):")
        for path, err in failed:
            print(f"  {path}: {err}")
        return 1
    elif len(args.pdfs) > 1:
        print(f"\n{'='*50}")
        print(f"All {len(args.pdfs)} PDFs processed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())