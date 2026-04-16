from pathlib import Path
import os
import time
import uuid
import re

from backend import create_backend, create_pass_backends
from backend.factory import create_pass_backends_from_config, backends_config_path, warn_deprecated_env_vars
from backend.config import _load_dotenv
from cli import build_parser, apply_args_to_env
from generate.renderers import generate_pages, generate_pages_wiki, render_pages_preview
from generate.synthesize import synthesize_pages
from pipeline import validate_pipeline_inputs, write_vault, run_pipeline_two_pass, run_pipeline_streaming, run_pipeline_parallel
from postprocess import (
    apply_canonical_map,
    check_evaluation_assertions,
    consolidate_concepts_llm,
    enrich_thin_concepts,
    evaluate_concepts,
    load_previous_evaluation,
    prune_low_signal_concepts,
    save_evaluation_snapshot,
)
from transform.cluster import cluster_related_concepts
from transform.canonicalize import canonicalize_concepts
from transform.fact_hygiene import apply_fact_hygiene
from transform.filter import (
    filter_concepts,
    filter_publishable_grouped_concepts,
)
from transform.grouping import group_facts_by_concept
from transform.merge import merge_similar_concepts
from transform.normalize import normalize_group_keys
from extract.fact_extractor import Fact


def _resynthesize_vault(vault_dir: str) -> None:
    """
    Re-run Pass 3 synthesis on pages in vault_dir that lack 'generated_by_backend:' frontmatter.

    These are typically enriched-only pages that never made it through synthesis,
    or pages from a prior partial run in the same directory.
    """
    _load_dotenv()

    vault_path = Path(vault_dir)
    if not vault_path.is_dir():
        print(f"Error: Vault directory not found: {vault_dir}")
        return

    # Find synthesis backend
    json_path = backends_config_path()
    if json_path:
        _, _, pass3_backend = create_pass_backends_from_config(json_path)
    else:
        pass3_backend = create_backend(provider="openai_compat", label="local")

    synthesis_workers = int(os.getenv("PIPELINE_SYNTHESIS_WORKERS", "1"))

    # Scan vault for unsynthesized pages
    unsynthesized_pages = {}
    markdown_files = sorted(vault_path.glob("*.md"))

    for md_file in markdown_files:
        content = md_file.read_text(encoding="utf-8")
        # Check if page was already synthesized
        if "generated_by_backend:" in content:
            continue

        # Parse title from filename (reverse the safe-name encoding)
        title = md_file.stem  # e.g., "Accounts Payable"
        unsynthesized_pages[title] = content

    if not unsynthesized_pages:
        print(f"No unsynthesized pages found in {vault_dir}")
        return

    print(f"Found {len(unsynthesized_pages)} unsynthesized pages. Starting synthesis...")

    # Extract pseudo-facts from enriched pages to reconstruct grouped dict
    grouped = {}
    for title, content in unsynthesized_pages.items():
        # Extract the body (after frontmatter and before Related Concepts)
        body_start = content.find("\n---\n")
        if body_start == -1:
            body_start = 0
        else:
            body_start += 4

        body_end = content.find("## Related Concepts")
        if body_end == -1:
            body_end = content.find("## See Also")
        if body_end == -1:
            body_end = len(content)

        body = content[body_start:body_end].strip()

        # Extract bullet points and paragraphs as fact-like content
        # Strip "Fact N.:" labels that may have leaked through
        fact_pattern = re.compile(r"\bFact\s+\w+\.:\s*(.+?)(?=\n|$)", re.MULTILINE)
        facts_from_labels = fact_pattern.findall(body)

        # Also extract bullet points
        bullet_pattern = re.compile(r"^\s*[-•*]\s+(.+?)$", re.MULTILINE)
        bullets = bullet_pattern.findall(body)

        # Also grab the intro paragraph (first non-heading paragraph)
        intro_lines = []
        for line in body.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and not stripped.startswith("-") and not stripped.startswith("•"):
                intro_lines.append(stripped)
                if len(intro_lines) >= 2:  # Take first 2 sentences roughly
                    break

        # Combine all extracted content as fact strings
        fact_contents = facts_from_labels + bullets + intro_lines
        fact_contents = [f.strip() for f in fact_contents if f.strip()]

        # Create pseudo-Fact objects
        # Fact dataclass requires: id (UUID), concept, content, source_chunk_id
        grouped[title] = [
            Fact(id=str(uuid.uuid4()), concept=title, content=fact_str, source_chunk_id="")
            for fact_str in fact_contents
        ]

    # Run synthesis on the grouped concepts
    pages_streaming = synthesize_pages(
        grouped=grouped,
        backend=pass3_backend,
        wiki_pages=unsynthesized_pages,
        workers=synthesis_workers,
        streaming=True,
    )

    # Write results back to vault (overwriting in place)
    write_vault(pages_streaming, vault_path)
    print(f"Resynthesis complete. {len(unsynthesized_pages)} pages updated in {vault_dir}")


def run_application(args) -> None:
    _load_dotenv()   # must run before any os.getenv() call

    # Handle --resynthesize mode (re-run Pass 3 only on unsynthesized pages)
    if args.resynthesize:
        return _resynthesize_vault(args.resynthesize)

    demo_pdf_path = args.input or "./sample_accounting_openstax.pdf"
    output_path = Path(args.output)
    batch_size = int(os.getenv("PIPELINE_BATCH_SIZE", "4"))
    pass1_batch_size = int(os.getenv("PIPELINE_PASS1_BATCH_SIZE", str(batch_size)))
    pass2_batch_size = int(os.getenv("PIPELINE_PASS2_BATCH_SIZE", str(batch_size)))
    max_workers = int(os.getenv("PIPELINE_MAX_WORKERS", "5"))
    _p1w_env = os.getenv("PASS1_MAX_WORKERS", "").strip()
    _p2w_env = os.getenv("PASS2_MAX_WORKERS", "").strip()
    _synth_env = os.getenv("PIPELINE_SYNTHESIS_WORKERS", "").strip()
    pass1_max_workers = int(_p1w_env) if _p1w_env else max_workers
    pass2_max_workers = int(_p2w_env) if _p2w_env else max_workers
    render_workers = int(os.getenv("PIPELINE_RENDER_WORKERS", "1"))
    enrich_workers = int(os.getenv("PIPELINE_ENRICH_WORKERS", "1"))
    use_synthesis = os.getenv("PIPELINE_SYNTHESIS", "0").strip().lower() in {"1", "true", "yes"}
    synthesis_workers = int(_synth_env) if _synth_env else 1
    max_chunks_env = os.getenv("PIPELINE_MAX_CHUNKS", "").strip()
    max_chunks = int(max_chunks_env) if max_chunks_env.isdigit() else None

    use_two_pass = os.getenv("TWO_PASS", "1").strip().lower() in {"1", "true", "yes"}
    use_streaming = os.getenv("PIPELINE_STREAMING", "0").strip().lower() in {"1", "true", "yes"}

    json_path = backends_config_path()
    if json_path:
        pass1_backend, pass2_backend, pass3_backend = create_pass_backends_from_config(json_path)
        canonicalize_backend = pass2_backend
    elif use_two_pass:
        print("=== LLM BACKEND CONFIGURATION ===")
        warn_deprecated_env_vars()
        pass1_backend, pass2_backend, pass3_backend = create_pass_backends()
        canonicalize_backend = pass2_backend
    else:
        print("=== LLM BACKEND CONFIGURATION ===")
        warn_deprecated_env_vars()
        backend = create_backend(label="single-pass")
        pass1_backend = backend
        pass2_backend = backend
        pass3_backend = backend
        canonicalize_backend = backend

    # Auto-derive max_workers from pool capacity when not explicitly overridden.
    # BackendPool.total_workers = sum of member workers, so the outer executor
    # has exactly enough threads to keep every backend slot busy at once.
    if not _p1w_env and hasattr(pass1_backend, "total_workers"):
        pass1_max_workers = pass1_backend.total_workers
        print(f"  [auto] pass1_max_workers = {pass1_max_workers} (pool capacity)")
    if not _p2w_env and hasattr(pass2_backend, "total_workers"):
        pass2_max_workers = pass2_backend.total_workers
        print(f"  [auto] pass2_max_workers = {pass2_max_workers} (pool capacity)")
    if not _synth_env and hasattr(pass3_backend, "total_workers"):
        synthesis_workers = pass3_backend.total_workers
        print(f"  [auto] synthesis_workers = {synthesis_workers} (pool capacity)")

    validate_pipeline_inputs(demo_pdf_path, output_path, seeds_file=args.seeds)

    pipeline_start = time.time()

    _two_pass_kwargs = dict(
        pass1_backend=pass1_backend,
        pass2_backend=pass2_backend,
        batch_size=batch_size,
        max_workers=max_workers,
        max_chunks=max_chunks,
        seeds_file=args.seeds,
        pass1_batch_size=pass1_batch_size,
        pass2_batch_size=pass2_batch_size,
        pass1_max_workers=pass1_max_workers,
        pass2_max_workers=pass2_max_workers,
    )
    if use_two_pass and use_streaming:
        print("\n=== USING STREAMING TWO-PASS PIPELINE ===")
        all_facts = run_pipeline_streaming(demo_pdf_path, **_two_pass_kwargs)
    elif use_two_pass:
        print("\n=== USING TWO-PASS PIPELINE ===")
        all_facts = run_pipeline_two_pass(demo_pdf_path, **_two_pass_kwargs)
    else:
        print("\n=== USING LEGACY SINGLE-PASS PIPELINE ===")
        all_facts = run_pipeline_parallel(
            demo_pdf_path,
            backend=pass1_backend,
            batch_size=batch_size,
            max_workers=max_workers,
            max_chunks=max_chunks,
        )

    print(f"Extracted {len(all_facts)} facts")
    for fact in all_facts[:5]:
        print(f"{fact.id} | {fact.concept} | {fact.content} | chunk={fact.source_chunk_id}")

    all_facts, dropped_noise = apply_fact_hygiene(all_facts)
    if dropped_noise:
        print(f"Dropped {dropped_noise} noisy pass-1/pass-2 artifacts before concept filtering")

    all_facts = filter_concepts(all_facts)
    print(f"After filtering: {len(all_facts)} valid concept facts")

    grouped = group_facts_by_concept(all_facts)
    rule_normalized_grouped = normalize_group_keys(grouped)

    concept_names = list(rule_normalized_grouped.keys())
    _t = time.perf_counter()
    canonical_map = canonicalize_concepts(concept_names, backend=canonicalize_backend)
    print(f"Canonicalization complete ({time.perf_counter() - _t:.0f}s)")

    final_grouped = apply_canonical_map(rule_normalized_grouped, canonical_map)
    final_grouped = merge_similar_concepts(final_grouped)
    final_grouped = cluster_related_concepts(final_grouped)

    if use_two_pass:
        _t = time.perf_counter()
        final_grouped = consolidate_concepts_llm(final_grouped, backend=canonicalize_backend)
        print(f"Consolidation complete ({time.perf_counter() - _t:.0f}s)")
        # Second dedup pass: LLM consolidation can rename concepts in ways that re-introduce near-duplicates
        final_grouped = merge_similar_concepts(final_grouped)

    enrich_threshold = int(os.getenv("PIPELINE_ENRICH_THRESHOLD", "6"))

    # OPTIMIZATION: Split concepts into ready (already have enough facts) vs thin (need enrichment).
    # Render ready concepts immediately while enrichment runs on thin concepts in parallel.
    ready_grouped = {}
    thin_grouped = {}
    if enrich_threshold > 0:
        for concept, facts in final_grouped.items():
            if len(facts) >= enrich_threshold:
                ready_grouped[concept] = facts
            else:
                thin_grouped[concept] = facts
    else:
        ready_grouped = final_grouped

    # Render ready concepts now (while enrichment runs)
    enhanced_mode = os.getenv("ENHANCED_PAGE_MODE", "1").strip().lower() in {"1", "true", "yes"}
    mode_label = "wiki" if enhanced_mode else "standard"
    render_fn = generate_pages_wiki if enhanced_mode else generate_pages

    if ready_grouped:
        print(f"\nRendering {len(ready_grouped)} ready concepts (>= {enrich_threshold} facts)...")
        _t_render = time.perf_counter()
        ready_pages = render_fn(ready_grouped, workers=render_workers)
        print(f"  Ready concepts rendered ({time.perf_counter() - _t_render:.0f}s)")
    else:
        ready_pages = {}

    # Enrich thin concepts in parallel while ready rendering just happened
    if thin_grouped and enrich_threshold > 0:
        print(f"\nEnrichment pass: filling {len(thin_grouped)} concepts with < {enrich_threshold} facts...")
        _t = time.perf_counter()
        thin_grouped = enrich_thin_concepts(
            thin_grouped, backend=canonicalize_backend, min_facts=enrich_threshold,
            workers=enrich_workers,
        )
        print(f"Enrichment complete ({time.perf_counter() - _t:.0f}s)")

    # Combine ready and enriched thin concepts
    final_grouped = {**ready_grouped, **thin_grouped}

    min_publishable = int(os.getenv("PIPELINE_MIN_PUBLISHABLE_FACTS", "2"))
    if min_publishable > 1:
        before = len(final_grouped)
        final_grouped = {k: v for k, v in final_grouped.items() if len(v) >= min_publishable}
        dropped = before - len(final_grouped)
        if dropped:
            print(f"  [prune] Dropped {dropped} stub concepts (< {min_publishable} facts after enrichment)")

    min_facts = int(os.getenv("PIPELINE_MIN_FACTS_PER_CONCEPT", "1"))
    pre_prune_count = len(final_grouped)
    final_grouped = prune_low_signal_concepts(final_grouped, min_facts_per_concept=min_facts)
    pruned_count = max(0, pre_prune_count - len(final_grouped))
    if pruned_count:
        print(f"Pruned {pruned_count} low-signal concepts (< {min_facts} facts)")

    before_publish = len(final_grouped)
    final_grouped = filter_publishable_grouped_concepts(final_grouped)
    publish_pruned = before_publish - len(final_grouped)
    if publish_pruned:
        print(f"Pruned {publish_pruned} non-publishable concepts before rendering")

    print("\n=== CONCEPT GROUPS ===")
    for concept, facts in sorted(final_grouped.items(), key=lambda x: -len(x[1])):
        print(f"  {concept} -> {len(facts)} facts")

    # Render thin concepts (after enrichment + pruning)
    if thin_grouped:
        thin_grouped_final = {c: facts for c, facts in final_grouped.items() if c in thin_grouped}
        if thin_grouped_final:
            print(f"\nRendering {len(thin_grouped_final)} enriched concepts...")
            _t_render = time.perf_counter()
            thin_pages = render_fn(thin_grouped_final, workers=render_workers)
            print(f"  Enriched concepts rendered ({time.perf_counter() - _t_render:.0f}s)")
        else:
            thin_pages = {}
    else:
        thin_pages = {}

    # Combine all rendered pages
    pages = {**ready_pages, **thin_pages}

    # Render any remaining concepts that weren't in ready or thin (shouldn't happen, but be safe)
    rendered_concepts = set(pages.keys())
    final_concepts_set = {concept for concept in final_grouped.keys()}
    missing_concepts = final_concepts_set - rendered_concepts
    if missing_concepts:
        missing_grouped = {c: final_grouped[c] for c in missing_concepts}
        print(f"\nRendering {len(missing_grouped)} remaining concepts...")
        missing_pages = render_fn(missing_grouped, workers=render_workers)
        pages.update(missing_pages)

    skipped_pages = max(0, len(final_grouped) - len(pages))
    print(f"\nGenerated {len(pages)} concept pages ({mode_label} mode)")
    if skipped_pages:
        print(f"Skipped {skipped_pages} empty/low-signal pages")

    if use_synthesis:
        print(
            f"\nPass 3 [synthesis]: Rewriting {len(final_grouped)} concepts "
            f"[{pass3_backend.label}:{pass3_backend.model}]..."
        )
        _t = time.perf_counter()
        # Stream synthesis results directly to vault (incremental writes)
        pages_streaming = synthesize_pages(
            final_grouped,
            backend=pass3_backend,
            wiki_pages=pages,
            workers=synthesis_workers,
            streaming=True,  # Returns list of (title, page) tuples as completed
        )

        # Preview before writing (sample first result if streaming)
        preview_page = None
        if isinstance(pages_streaming, list) and pages_streaming:
            _, preview_page = pages_streaming[0]

        write_vault(pages_streaming, output_path)
        print(f"Pass 3 [synthesis]: completed ({time.perf_counter() - _t:.0f}s)")

        # Update pages dict for preview below
        pages = {title: page for title, page in pages_streaming}
    else:
        preview_page = None

    preview = render_pages_preview(pages, max_pages=2) if not use_synthesis else None
    if preview:
        print("\n=== PAGE PREVIEW (FIRST 1-2) ===")
        print(preview)
    elif use_synthesis and preview_page:
        print("\n=== PAGE PREVIEW (FIRST SYNTHESIZED) ===")
        print(preview_page[:500] + "..." if len(preview_page) > 500 else preview_page)

    eval_result = evaluate_concepts(final_grouped)

    print("\n=== EVALUATION ===")
    for k, v in eval_result.items():
        print(k, ":", v)
    print(f"total_pipeline_time : {time.time() - pipeline_start:.0f}s")

    # Generate redirect stubs for near-duplicate pairs
    from postprocess import generate_redirect_pages
    near_dups = eval_result.get("near_duplicates", [])
    if near_dups:
        redirect_pages = generate_redirect_pages(near_dups, final_grouped)
        if redirect_pages:
            print(f"\n[dedup] Writing {len(redirect_pages)} redirect stubs for near-duplicate concepts...")
            write_vault(redirect_pages, output_path)

    previous_eval = load_previous_evaluation()
    check_evaluation_assertions(eval_result, previous_eval)
    save_evaluation_snapshot(eval_result)


if __name__ == "__main__":
    args = build_parser().parse_args()
    apply_args_to_env(args)
    run_application(args)
