from pathlib import Path
from typing import Iterator, List, Optional
import json
import re

from extract.fact_extractor import extract_facts_batched, Fact
from ingest.pdf_loader import Chunk, load_pdf_chunks
from transform.cluster import cluster_related_concepts
from transform.grouping import group_facts_by_concept
from transform.canonicalize import canonicalize_concepts
from transform.merge import merge_similar_concepts
from transform.normalize import normalize_group_keys
from transform.filter import filter_concepts
from generate.pages import generate_pages, render_pages_document, render_pages_preview
from concurrent.futures import ThreadPoolExecutor, as_completed

EVALUATION_CACHE_PATH = Path(__file__).with_name("evaluation_metrics.json")

def generate_chunk_batches(chunks: List[Chunk], batch_size: int = 2) -> Iterator[List[Chunk]]:
    if batch_size < 1:
        batch_size = 1
    for i in range(0, len(chunks), batch_size):
        yield chunks[i : i + batch_size]

def run_pipeline_parallel(pdf_path: str, batch_size: int = 2, max_workers: int = 5) -> List[Fact]:
    # Step 3: End-to-end pipeline: PDF -> Chunks -> Facts.
    chunks = load_pdf_chunks(pdf_path=pdf_path)
    chunk_batches = list(generate_chunk_batches(chunks, batch_size=batch_size))

    all_facts: List[Fact] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(extract_facts_batched, batch, batch_size)
            for batch in chunk_batches
        ]

        for future in as_completed(futures):
            try:
                batch_facts = future.result()
                all_facts.extend(batch_facts)
            except Exception:
                continue

    return all_facts


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
        has_plural = any(
            w.endswith("s") and len(w) > 3 and not original.isupper()
            for w, original in zip(words_lower, words)
        )
        has_lowercase_acronym = any(w in lowercase_acronyms for w in words)

        if has_filler or has_plural or has_lowercase_acronym:
            suspicious_concepts.append(concept)

    near_duplicates: List[tuple[str, str]] = []

    def differs_by_one_word(left: str, right: str) -> bool:
        left_words = left.lower().split()
        right_words = right.lower().split()

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
            left_words = left.lower().split()
            right_words = right.lower().split()
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

    # Assertion 1: singleton_ratio should be < 0.8.
    singleton_value = current.get("singleton_ratio", 0.0)
    if isinstance(singleton_value, (int, float)):
        singleton_ratio = singleton_value / 100.0 if singleton_value > 1 else singleton_value
        if singleton_ratio >= 0.8:
            warnings.append(
                f"singleton_ratio is high ({singleton_ratio:.2f}); expected < 0.8"
            )

    if previous is not None:
        # Assertion 2: suspicious_concepts count should decrease over time.
        current_suspicious = current.get("suspicious_concepts", [])
        previous_suspicious = previous.get("suspicious_concepts", [])
        current_count = len(current_suspicious) if isinstance(current_suspicious, list) else 0
        previous_count = len(previous_suspicious) if isinstance(previous_suspicious, list) else 0

        if current_count >= previous_count and previous_count > 0:
            warnings.append(
                f"suspicious_concepts did not decrease ({previous_count} -> {current_count})"
            )

        # Assertion 3: total_concepts should not drop by more than 30%.
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
    # Step 4: Demo run and print first 5 facts.
    demo_pdf_path = "./sample-accounting-openstax.pdf"
    all_facts = run_pipeline_parallel(demo_pdf_path, batch_size=2, max_workers=5)
    print(f"Extracted {len(all_facts)} facts")
    for fact in all_facts[:5]:
        print(f"{fact.id} | {fact.concept} | {fact.content} "
              f"| chunk={fact.source_chunk_id}")
    
    # Step 4.5: Filter concepts by eligibility (reject examples, proper nouns, etc.)
    all_facts = filter_concepts(all_facts)
    print(f"After filtering: {len(all_facts)} valid concept facts")
    
    grouped = group_facts_by_concept(all_facts)

    # Step 5: Deterministic rule-based normalization before LLM canonicalization.
    rule_normalized_grouped = normalize_group_keys(grouped)

    # Step 6: Canonicalize normalized concepts (cache-backed to avoid redundant LLM calls).
    concept_names = list(rule_normalized_grouped.keys())
    canonical_map = canonicalize_concepts(concept_names)

    # Step 7: Build final grouped map with canonical names.
    final_grouped = apply_canonical_map(rule_normalized_grouped, canonical_map)

    final_grouped = merge_similar_concepts(final_grouped)
    final_grouped = cluster_related_concepts(final_grouped)

    # Step 8: Print final grouped concept counts.
    for concept, facts in final_grouped.items():
        print(concept, "->", len(facts))

    # Step 9: Generate simple concept pages before evaluation output.
    pages = generate_pages(final_grouped)
    skipped_pages = max(0, len(final_grouped) - len(pages))
    print(f"Generated {len(pages)} concept pages")
    if skipped_pages:
        print(f"Skipped {skipped_pages} empty/low-signal pages")

    preview = render_pages_preview(pages, max_pages=2)
    if preview:
        print("\n=== PAGE PREVIEW (FIRST 1-2) ===")
        print(preview)

    pages_doc = render_pages_document(pages)
    pages_output_path = Path("output") / "concept_pages_sample.md"
    pages_output_path.parent.mkdir(parents=True, exist_ok=True)
    pages_output_path.write_text(pages_doc, encoding="utf-8")
    print(f"Saved combined pages to {pages_output_path}")

    eval_result = evaluate_concepts(final_grouped)

    # Evaluation
    print("\n=== EVALUATION ===")
    for k, v in eval_result.items():
        print(k, ":", v)

    previous_eval = load_previous_evaluation()
    check_evaluation_assertions(eval_result, previous_eval)
    save_evaluation_snapshot(eval_result)