from pathlib import Path
import os
import time

from backend import create_backend, create_pass_backends
from backend.config import _load_dotenv
from cli import build_parser, apply_args_to_env
from generate.renderers import generate_pages, generate_pages_wiki, render_pages_preview
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


def run_application(args) -> None:
    _load_dotenv()   # must run before any os.getenv() call
    demo_pdf_path = args.input or "./sample_accounting_openstax.pdf"
    output_path = Path(args.output)
    batch_size = int(os.getenv("PIPELINE_BATCH_SIZE", "4"))
    pass1_batch_size = int(os.getenv("PIPELINE_PASS1_BATCH_SIZE", str(batch_size)))
    pass2_batch_size = int(os.getenv("PIPELINE_PASS2_BATCH_SIZE", str(batch_size)))
    max_workers = int(os.getenv("PIPELINE_MAX_WORKERS", "5"))
    pass1_max_workers = int(os.getenv("PASS1_MAX_WORKERS", str(max_workers)))
    pass2_max_workers = int(os.getenv("PASS2_MAX_WORKERS", str(max_workers)))
    render_workers = int(os.getenv("PIPELINE_RENDER_WORKERS", "1"))
    enrich_workers = int(os.getenv("PIPELINE_ENRICH_WORKERS", "1"))
    max_chunks_env = os.getenv("PIPELINE_MAX_CHUNKS", "").strip()
    max_chunks = int(max_chunks_env) if max_chunks_env.isdigit() else None

    use_two_pass = os.getenv("TWO_PASS", "1").strip().lower() in {"1", "true", "yes"}
    use_streaming = os.getenv("PIPELINE_STREAMING", "0").strip().lower() in {"1", "true", "yes"}

    print("=== LLM BACKEND CONFIGURATION ===")
    if use_two_pass:
        pass1_backend, pass2_backend = create_pass_backends()
        canonicalize_backend = pass2_backend
    else:
        backend = create_backend(label="single-pass")
        pass1_backend = backend
        pass2_backend = backend
        canonicalize_backend = backend

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
    canonical_map = canonicalize_concepts(concept_names, backend=canonicalize_backend)

    final_grouped = apply_canonical_map(rule_normalized_grouped, canonical_map)
    final_grouped = merge_similar_concepts(final_grouped)
    final_grouped = cluster_related_concepts(final_grouped)

    if use_two_pass:
        final_grouped = consolidate_concepts_llm(final_grouped, backend=canonicalize_backend)

    enrich_threshold = int(os.getenv("PIPELINE_ENRICH_THRESHOLD", "6"))
    if enrich_threshold > 0:
        print(f"\nEnrichment pass: filling concepts with < {enrich_threshold} facts...")
        final_grouped = enrich_thin_concepts(
            final_grouped, backend=canonicalize_backend, min_facts=enrich_threshold,
            workers=enrich_workers,
        )

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

    enhanced_mode = os.getenv("ENHANCED_PAGE_MODE", "1").strip().lower() in {"1", "true", "yes"}
    if enhanced_mode:
        pages = generate_pages_wiki(final_grouped, workers=render_workers)
        mode_label = "wiki"
    else:
        pages = generate_pages(final_grouped, workers=render_workers)
        mode_label = "standard"
    skipped_pages = max(0, len(final_grouped) - len(pages))
    print(f"\nGenerated {len(pages)} concept pages ({mode_label} mode)")
    if skipped_pages:
        print(f"Skipped {skipped_pages} empty/low-signal pages")

    preview = render_pages_preview(pages, max_pages=2)
    if preview:
        print("\n=== PAGE PREVIEW (FIRST 1-2) ===")
        print(preview)

    write_vault(pages, output_path)

    eval_result = evaluate_concepts(final_grouped)

    print("\n=== EVALUATION ===")
    for k, v in eval_result.items():
        print(k, ":", v)
    print(f"total_pipeline_time : {time.time() - pipeline_start:.0f}s")

    previous_eval = load_previous_evaluation()
    check_evaluation_assertions(eval_result, previous_eval)
    save_evaluation_snapshot(eval_result)


if __name__ == "__main__":
    args = build_parser().parse_args()
    apply_args_to_env(args)
    run_application(args)
