from pathlib import Path
from typing import Iterator, List, Optional
import json
import os
import re
import sys

from backend import create_backend, create_pass_backends, LLMBackend, LLMBackendError
from extract.fact_extractor import (
    extract_facts_batched,
    extract_raw_statements_batched,
    assign_concepts_to_statements,
    derive_seed_concepts,
    load_seeds_from_file,
    load_builtin_seeds,
    Fact,
    _parse_json_object,
)
from ingest.pdf_loader import Chunk, load_pdf_chunks
from transform.cluster import cluster_related_concepts
from transform.grouping import group_facts_by_concept
from transform.canonicalize import canonicalize_concepts
from transform.merge import merge_similar_concepts
from transform.normalize import normalize_group_keys
from transform.filter import filter_concepts
from generate.renderers import (
    generate_pages,
    generate_pages_wiki,
    render_pages_preview,
)
from transform.matching import has_antonym_conflict, is_sibling, tokenize_for_matching
from concurrent.futures import ThreadPoolExecutor, as_completed

EVALUATION_CACHE_PATH = Path(__file__).with_name("evaluation_metrics.json")

# Characters that are invalid in filenames on Windows / macOS / Linux.
_INVALID_FILENAME_CHARS = str.maketrans({c: "" for c in r'\/:*?"<>|'})


def write_vault(pages: dict[str, str], output_dir: Path) -> None:
    """
    Write each concept page as an individual .md file into output_dir.

    - If output_dir does not exist, it is created.
    - If output_dir exists and is non-empty, a warning is printed and writing continues.
    - If the directory cannot be created or a file cannot be written due to a
      permission error or OS error, the process exits with a helpful message.
    """
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            print(f"Error: Cannot create output directory '{output_dir}': {exc}")
            print("Check that you have write permission to the parent directory.")
            sys.exit(1)
        except OSError as exc:
            print(f"Error: Failed to create output directory '{output_dir}': {exc}")
            sys.exit(1)
    else:
        existing = list(output_dir.iterdir())
        if existing:
            print(
                f"Warning: Output directory '{output_dir}' already exists and "
                f"contains {len(existing)} file(s). Adding pages to existing directory."
            )

    written = 0
    for title, content in pages.items():
        safe_name = title.translate(_INVALID_FILENAME_CHARS).strip() or "Unnamed Concept"
        file_path = output_dir / f"{safe_name}.md"
        try:
            file_path.write_text(content, encoding="utf-8")
            written += 1
        except PermissionError:
            print(f"Error: Cannot write '{file_path}': permission denied.")
            print("Check that you have write access to the output directory.")
            sys.exit(1)
        except OSError as exc:
            print(f"Error: Failed to write '{file_path}': {exc}")
            sys.exit(1)

    print(f"Vault written: {written} pages -> {output_dir}/")


def generate_chunk_batches(chunks: List[Chunk], batch_size: int = 2) -> Iterator[List[Chunk]]:
    if batch_size < 1:
        batch_size = 1
    for i in range(0, len(chunks), batch_size):
        yield chunks[i : i + batch_size]


# ===================================================================
# Two-pass pipeline
# ===================================================================

def resolve_seed_concepts(
    statements: List[dict],
    pass2_backend: LLMBackend,
    seeds_file: Optional[str] = None,
) -> List[str]:
    """
    Resolve the seed concept list for Pass 2, in priority order:
      1. User-supplied JSON file (--seeds / seeds_file)
      2. Auto-generated from statements via Pass 1.5
      3. Hardcoded SEED_CONCEPTS fallback

    Returns the seed list to use.
    """
    if seeds_file:
        try:
            seeds = load_seeds_from_file(seeds_file)
            print(f"Pass 1.5: Loaded {len(seeds)} seed concepts from {seeds_file}")
            return seeds
        except Exception as exc:
            print(f"Pass 1.5: Failed to load seeds file ({exc}), falling back to auto-generation")

    print(f"Pass 1.5: Deriving seed concepts from statements "
          f"[{pass2_backend.label}:{pass2_backend.model}]...")
    seeds = derive_seed_concepts(statements, pass2_backend)
    if seeds:
        print(f"Pass 1.5 complete: {len(seeds)} seed concepts derived")
        return seeds

    builtin = load_builtin_seeds()
    print(f"Pass 1.5: Auto-generation failed, using built-in seed list "
          f"({len(builtin)} concepts)")
    return builtin


def run_pipeline_two_pass(
    pdf_path: str,
    pass1_backend: LLMBackend,
    pass2_backend: LLMBackend,
    batch_size: int = 2,
    max_workers: int = 5,
    max_chunks: Optional[int] = None,
    seeds_file: Optional[str] = None,
) -> List[Fact]:
    """
    Two-pass extraction pipeline:
      Pass 1:   Extract raw factual statements (no concept naming)
      Pass 1.5: Derive seed concept names from statements (or load from file)
      Pass 2:   Assign concept names using the seed list

    Each pass can use a different LLM backend (hybrid mode).
    """
    chunks = load_pdf_chunks(pdf_path=pdf_path)
    if isinstance(max_chunks, int) and max_chunks > 0:
        chunks = chunks[:max_chunks]

    chunk_batches = list(generate_chunk_batches(chunks, batch_size=batch_size))
    all_statements: List[dict] = []

    print(f"Pass 1: Extracting raw statements from {len(chunks)} chunks "
          f"[{pass1_backend.label}:{pass1_backend.model}]...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(extract_raw_statements_batched, batch, pass1_backend, batch_size)
            for batch in chunk_batches
        ]
        for future in as_completed(futures):
            try:
                batch_statements = future.result()
                all_statements.extend(batch_statements)
            except Exception:
                continue

    print(f"Pass 1 complete: {len(all_statements)} raw statements extracted")

    seeds = resolve_seed_concepts(all_statements, pass2_backend, seeds_file=seeds_file)

    print(f"Pass 2: Assigning concept names to {len(all_statements)} statements "
          f"[{pass2_backend.label}:{pass2_backend.model}]...")
    all_facts = assign_concepts_to_statements(all_statements, pass2_backend, seed_concepts=seeds)
    print(f"Pass 2 complete: {len(all_facts)} facts with concept names")

    return all_facts


# ===================================================================
# Legacy single-pass pipeline
# ===================================================================

def run_pipeline_parallel(
    pdf_path: str,
    backend: LLMBackend,
    batch_size: int = 2,
    max_workers: int = 5,
    max_chunks: Optional[int] = None,
) -> List[Fact]:
    """Legacy single-pass pipeline: PDF -> Chunks -> Facts."""
    chunks = load_pdf_chunks(pdf_path=pdf_path)
    if isinstance(max_chunks, int) and max_chunks > 0:
        chunks = chunks[:max_chunks]
    chunk_batches = list(generate_chunk_batches(chunks, batch_size=batch_size))

    all_facts: List[Fact] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(extract_facts_batched, batch, backend, batch_size)
            for batch in chunk_batches
        ]

        for future in as_completed(futures):
            try:
                batch_facts = future.result()
                all_facts.extend(batch_facts)
            except Exception:
                continue

    return all_facts


# ===================================================================
# Post-extraction consolidation pass
# ===================================================================

def consolidate_concepts_llm(
    grouped: dict[str, List[Fact]],
    backend: LLMBackend,
) -> dict[str, List[Fact]]:
    """
    Ask the LLM which concept groups should be merged.

    This is a simpler task than extraction — the model just needs to
    identify duplicates in a flat list of names.
    """
    concept_names = list(grouped.keys())
    if len(concept_names) <= 3:
        return grouped

    concept_list = "\n".join(f"  - {c}" for c in concept_names)

    prompt = f"""These are concept names extracted from a textbook chapter.
Some may refer to the same concept under different names.

Concept list:
{concept_list}

Rules:
- Only merge concepts that are TRULY the same thing (just named differently).
- Do NOT merge related-but-distinct concepts.
- "Inventory Fraud" and "Inventory Shrinkage" are DIFFERENT — do not merge.
- "Gross Profit" and "Gross Margin" may be the same — merge if so.
- "FIFO" and "First In First Out" are the same — merge.

Output a JSON object mapping duplicate names to their canonical name.
Only include concepts that need merging. Use the more standard name as canonical.
If no merges needed, output: {{}}

Example: {{"FIFO": "First In First Out", "Gross Margin": "Gross Profit"}}"""

    try:
        raw_content = backend.generate(prompt, max_tokens=400)
    except Exception:
        return grouped

    parsed = _parse_json_object(raw_content)
    if not isinstance(parsed, dict):
        return grouped

    merge_map: dict[str, str] = {}
    for source, target in parsed.items():
        if (
            isinstance(source, str)
            and isinstance(target, str)
            and source in grouped
            and target in grouped
            and source != target
        ):
            if has_antonym_conflict(source, target):
                continue
            if is_sibling(source, target):
                continue
            merge_map[source] = target

    if not merge_map:
        return grouped

    result: dict[str, List[Fact]] = {}
    for concept, facts in grouped.items():
        target = merge_map.get(concept, concept)
        result.setdefault(target, []).extend(facts)

    merged_count = len(grouped) - len(result)
    if merged_count > 0:
        print(f"Consolidation: merged {merged_count} duplicate concept groups")

    return result


def apply_canonical_map(
    grouped: dict[str, List[Fact]],
    canonical_map: dict[str, Optional[str]],
) -> dict[str, List[Fact]]:
    remapped: dict[str, List[Fact]] = {}
    for concept, facts in grouped.items():
        canonical_name = canonical_map.get(concept)
        target_name = canonical_name if canonical_name is not None else concept
        remapped.setdefault(target_name, []).extend(facts)
    return remapped


def prune_low_signal_concepts(
    grouped: dict[str, List[Fact]],
    min_facts_per_concept: int = 1,
) -> dict[str, List[Fact]]:
    if min_facts_per_concept <= 1:
        return grouped

    return {
        concept: facts
        for concept, facts in grouped.items()
        if len(facts) >= min_facts_per_concept
    }


def evaluate_concepts(grouped: dict[str, list]) -> dict:
    concepts = list(grouped.keys())
    total_concepts = len(concepts)
    total_facts = sum(len(grouped[concept]) for concept in concepts)
    avg_facts_per_concept = (total_facts / total_concepts) if total_concepts else 0.0

    singleton_count = sum(1 for concept in concepts if len(grouped[concept]) == 1)
    singleton_ratio = (singleton_count / total_concepts) if total_concepts else 0.0

    suspicious_concepts: List[str] = []
    filler_words = {"goals", "impact", "effects"}
    lowercase_acronyms = {"Aes", "Des", "Tls"}

    for concept in concepts:
        words = re.findall(r"[A-Za-z]+", concept)
        words_lower = [w.lower() for w in words]

        has_filler = any(w in filler_words for w in words_lower)
        has_plural = (
            len(words) == 1
            and words_lower[0].endswith("s")
            and len(words_lower[0]) > 3
            and not words[0].isupper()
        ) if words else False
        has_lowercase_acronym = any(w in lowercase_acronyms for w in words)

        if has_filler or has_plural or has_lowercase_acronym:
            suspicious_concepts.append(concept)

    near_duplicates: List[tuple[str, str]] = []

    def differs_by_one_word(left: str, right: str) -> bool:
        left_words = tokenize_for_matching(left)
        right_words = tokenize_for_matching(right)

        if min(len(left_words), len(right_words)) < 2:
            return False

        if abs(len(left_words) - len(right_words)) == 1:
            shorter = left_words if len(left_words) < len(right_words) else right_words
            longer = right_words if len(right_words) > len(left_words) else left_words
            return all(word in longer for word in shorter)

        if len(left_words) == len(right_words):
            mismatches = sum(1 for a, b in zip(left_words, right_words) if a != b)
            return mismatches == 1

        return False

    for i, left in enumerate(concepts):
        for right in concepts[i + 1 :]:
            if has_antonym_conflict(left, right):
                continue
            if is_sibling(left, right):
                continue

            left_words = tokenize_for_matching(left)
            right_words = tokenize_for_matching(right)
            shorter_words = left_words if len(left_words) <= len(right_words) else right_words
            longer_words = right_words if len(right_words) >= len(left_words) else left_words
            is_substring_match = (
                len(shorter_words) >= 2
                and " ".join(shorter_words) in " ".join(longer_words)
            )
            if is_substring_match or differs_by_one_word(left, right):
                near_duplicates.append((left, right))

    return {
        "total_concepts": total_concepts,
        "avg_facts_per_concept": avg_facts_per_concept,
        "singleton_ratio": singleton_ratio,
        "suspicious_concepts": suspicious_concepts,
        "near_duplicates": near_duplicates,
    }


def load_previous_evaluation() -> Optional[dict]:
    if not EVALUATION_CACHE_PATH.exists():
        return None

    try:
        with EVALUATION_CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    return data if isinstance(data, dict) else None


def save_evaluation_snapshot(evaluation: dict) -> None:
    with EVALUATION_CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)


def check_evaluation_assertions(current: dict, previous: Optional[dict]) -> None:
    warnings: List[str] = []

    singleton_value = current.get("singleton_ratio", 0.0)
    if isinstance(singleton_value, (int, float)):
        singleton_ratio = singleton_value / 100.0 if singleton_value > 1 else singleton_value
        if singleton_ratio >= 0.8:
            warnings.append(
                f"singleton_ratio is high ({singleton_ratio:.2f}); expected < 0.8"
            )

    if previous is not None:
        current_suspicious = current.get("suspicious_concepts", [])
        previous_suspicious = previous.get("suspicious_concepts", [])
        current_count = len(current_suspicious) if isinstance(current_suspicious, list) else 0
        previous_count = len(previous_suspicious) if isinstance(previous_suspicious, list) else 0

        if current_count >= previous_count and previous_count > 0:
            warnings.append(
                f"suspicious_concepts did not decrease ({previous_count} -> {current_count})"
            )

        current_total = current.get("total_concepts", 0)
        previous_total = previous.get("total_concepts", 0)
        if (
            isinstance(current_total, int)
            and isinstance(previous_total, int)
            and previous_total > 0
            and current_total < previous_total * 0.7
        ):
            warnings.append(
                f"total_concepts dropped too much ({previous_total} -> {current_total})"
            )

    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    from cli import build_parser, apply_args_to_env
    args = build_parser().parse_args()
    apply_args_to_env(args)

    demo_pdf_path = args.input or "./sample_accounting_openstax.pdf"
    batch_size = int(os.getenv("PIPELINE_BATCH_SIZE", "2"))
    max_workers = int(os.getenv("PIPELINE_MAX_WORKERS", "5"))
    max_chunks_env = os.getenv("PIPELINE_MAX_CHUNKS", "").strip()
    max_chunks = int(max_chunks_env) if max_chunks_env.isdigit() else None

    use_two_pass = os.getenv("TWO_PASS", "1").strip().lower() in {"1", "true", "yes"}

    # Initialize LLM backends
    print("=== LLM BACKEND CONFIGURATION ===")
    if use_two_pass:
        pass1_backend, pass2_backend = create_pass_backends()
        # Use pass2 backend for canonicalization and consolidation too
        canonicalize_backend = pass2_backend
    else:
        backend = create_backend(label="single-pass")
        pass1_backend = backend
        pass2_backend = backend
        canonicalize_backend = backend

    if use_two_pass:
        print("\n=== USING TWO-PASS PIPELINE ===")
        all_facts = run_pipeline_two_pass(
            demo_pdf_path,
            pass1_backend=pass1_backend,
            pass2_backend=pass2_backend,
            batch_size=batch_size,
            max_workers=max_workers,
            max_chunks=max_chunks,
            seeds_file=args.seeds,
        )
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
        print(f"{fact.id} | {fact.concept} | {fact.content} "
              f"| chunk={fact.source_chunk_id}")

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

    min_facts = int(os.getenv("PIPELINE_MIN_FACTS_PER_CONCEPT", "1"))
    pre_prune_count = len(final_grouped)
    final_grouped = prune_low_signal_concepts(final_grouped, min_facts_per_concept=min_facts)
    pruned_count = max(0, pre_prune_count - len(final_grouped))
    if pruned_count:
        print(f"Pruned {pruned_count} low-signal concepts (< {min_facts} facts)")

    print("\n=== CONCEPT GROUPS ===")
    for concept, facts in sorted(final_grouped.items(), key=lambda x: -len(x[1])):
        print(f"  {concept} -> {len(facts)} facts")

    enhanced_mode = os.getenv("ENHANCED_PAGE_MODE", "0").strip().lower() in {"1", "true", "yes"}
    if enhanced_mode:
        pages = generate_pages_wiki(final_grouped)
        mode_label = "wiki"
    else:
        pages = generate_pages(final_grouped)
        mode_label = "standard"
    skipped_pages = max(0, len(final_grouped) - len(pages))
    print(f"\nGenerated {len(pages)} concept pages ({mode_label} mode)")
    if skipped_pages:
        print(f"Skipped {skipped_pages} empty/low-signal pages")

    preview = render_pages_preview(pages, max_pages=2)
    if preview:
        print("\n=== PAGE PREVIEW (FIRST 1-2) ===")
        print(preview)

    write_vault(pages, Path(args.output))

    eval_result = evaluate_concepts(final_grouped)

    print("\n=== EVALUATION ===")
    for k, v in eval_result.items():
        print(k, ":", v)

    previous_eval = load_previous_evaluation()
    check_evaluation_assertions(eval_result, previous_eval)
    save_evaluation_snapshot(eval_result)