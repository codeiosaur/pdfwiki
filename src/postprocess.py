from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import uuid as _uuid

from backend import LLMBackend
from extract.fact_extractor import Fact, _parse_json_object, _parse_json_array
from generate.titles import concept_tokens
from transform.matching import has_antonym_conflict, is_cousin, is_sibling, tokenize_for_matching


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


def enrich_thin_concepts(
    grouped: dict[str, List[Fact]],
    backend: LLMBackend,
    min_facts: int = 4,
    workers: int = 1,
) -> dict[str, List[Fact]]:
    enriched: dict[str, List[Fact]] = {c: list(f) for c, f in grouped.items()}
    thin = sorted(c for c, f in grouped.items() if len(f) < min_facts)
    if not thin:
        return enriched

    def _enrich_one(concept: str) -> tuple[str, list[Fact]]:
        facts = grouped[concept]
        need = min_facts - len(facts) + 1
        existing_block = "\n".join(f"- {f.content}" for f in facts)
        prompt = f"""The following facts about "{concept}" were extracted from a textbook.
Expand on them by writing {need} additional factual statements.

Existing facts:
{existing_block}

Rules:
- Each new statement must be a direct elaboration, consequence, or clarification
  of the existing facts above.
- Do NOT introduce information that is not already implied by the existing facts.
- Do NOT repeat or rephrase the existing facts.
- Each statement must be ONE complete, self-contained factual claim.
- Write clear sentences a student could study from.

Output ONLY a JSON array of strings:
["Fact one.", "Fact two.", ...]"""

        try:
            raw = backend.generate(prompt, max_tokens=600)
        except Exception as exc:
            print(f"  [enrich] {concept}: LLM call failed: {exc}")
            return concept, []

        parsed = _parse_json_array(raw)
        if not isinstance(parsed, list):
            return concept, []

        new_facts: list[Fact] = []
        for item in parsed:
            if not isinstance(item, str) or not item.strip():
                continue
            new_facts.append(Fact(
                id=str(_uuid.uuid4()),
                concept=concept,
                content=item.strip(),
                source_chunk_id="",
            ))
        return concept, new_facts

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results: list = list(executor.map(_enrich_one, thin))
    else:
        results = [_enrich_one(c) for c in thin]

    total_added = 0
    enriched_count = 0
    for concept, new_facts in results:
        if new_facts:
            enriched[concept].extend(new_facts)
            total_added += len(new_facts)
            enriched_count += 1

    if total_added:
        print(f"  [enrich] Added {total_added} facts across {enriched_count} thin concepts")

    return enriched


def consolidate_concepts_llm(
    grouped: dict[str, List[Fact]],
    backend: LLMBackend,
) -> dict[str, List[Fact]]:
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
        words = tokenize_for_matching(concept)
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

    def _is_contiguous_token_subsequence(shorter: list[str], longer: list[str]) -> bool:
        """True if `shorter` appears as a contiguous run of tokens inside `longer`."""
        if not shorter or len(shorter) > len(longer):
            return False
        for start in range(len(longer) - len(shorter) + 1):
            if longer[start : start + len(shorter)] == shorter:
                return True
        return False

    for i, left in enumerate(concepts):
        for right in concepts[i + 1 :]:
            if has_antonym_conflict(left, right):
                continue
            if is_sibling(left, right):
                continue
            if is_cousin(left, right):
                continue

            left_words = tokenize_for_matching(left)
            right_words = tokenize_for_matching(right)
            shorter_words = left_words if len(left_words) <= len(right_words) else right_words
            longer_words = right_words if len(right_words) >= len(left_words) else left_words
            # Token-level contiguous subsequence — prevents string-level false
            # positives like "direct material" matching inside "indirect
            # materials expense".
            is_substring_match = (
                len(shorter_words) >= 2
                and _is_contiguous_token_subsequence(shorter_words, longer_words)
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
    from pathlib import Path
    import json

    evaluation_cache_path = Path(__file__).with_name("evaluation_metrics.json")
    if not evaluation_cache_path.exists():
        return None

    try:
        with evaluation_cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    return data if isinstance(data, dict) else None


def save_evaluation_snapshot(evaluation: dict) -> None:
    from pathlib import Path
    import json

    evaluation_cache_path = Path(__file__).with_name("evaluation_metrics.json")
    with evaluation_cache_path.open("w", encoding="utf-8") as f:
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


def generate_redirect_pages(
    near_duplicates: List[tuple[str, str]],
    final_grouped: dict[str, List[Fact]],
) -> dict[str, str]:
    """
    For near-duplicate pairs, generate a redirect stub for the concept
    with fewer facts pointing to the concept with more facts.

    Returns a dict of {display_title: page_content} redirect stubs.
    Skips pairs where both concepts have the same fact count (ambiguous).

    Applies the same antonym / sibling / cousin safety filters as
    `evaluate_concepts` to prevent semantically-distinct pairs (e.g.
    Accounts Payable vs Accounts Receivable, Fixed Cost vs Fixed Asset)
    from being collapsed into a redirect.
    """
    from generate.titles import normalize_page_title

    redirects: dict[str, str] = {}
    for left, right in near_duplicates:
        # Re-verify semantic safety — the near-duplicates list may have been
        # computed by a caller with looser rules.
        if has_antonym_conflict(left, right):
            continue
        if is_sibling(left, right):
            continue
        if is_cousin(left, right):
            continue

        left_count = len(final_grouped.get(left, []))
        right_count = len(final_grouped.get(right, []))

        # Skip if both have same fact count (ambiguous which is canonical)
        if left_count == right_count:
            continue

        # Determine source (fewer facts) and target (more facts)
        source = left if left_count < right_count else right
        target = right if left_count < right_count else left

        # Skip if source doesn't exist in final_grouped (already merged away)
        if source not in final_grouped:
            continue

        source_title = normalize_page_title(source)
        target_title = normalize_page_title(target)

        # Obsidian-compatible redirect stub with backend attribution so the
        # quality tool buckets these under "redirect" rather than "unknown".
        page = (
            "---\n"
            "generated_by_backend: redirect\n"
            "generated_by_model: none\n"
            "---\n\n"
            f"# {source_title}\n\n"
            f"> This page redirects to [[{target_title}]].\n\n"
            "## Redirect\n"
            f"[[{target_title}]]\n"
        )
        redirects[source_title] = page

    return redirects
