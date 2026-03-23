from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import re
import uuid

from pypdf import PdfReader
from extract.fact_extractor import extract_facts, Fact, ollama_client, OLLAMA_MODEL


CANONICAL_CACHE_PATH = Path(__file__).with_name("canonical_cache.json")
EVALUATION_CACHE_PATH = Path(__file__).with_name("evaluation_metrics.json")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chapter: Optional[str]


def load_pdf_chunks(pdf_path: str, chunk_size_words: int = 1000) -> List[Chunk]:
    # Step 1: Load PDF and collect text from all pages.
    reader = PdfReader(pdf_path)
    page_texts: List[str] = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")

    full_text = "\n".join(page_texts)

    # Step 2: Split into ~1000-word chunks.
    words = full_text.split()
    chunks: List[Chunk] = []
    source_name = Path(pdf_path).name

    for i in range(0, len(words), chunk_size_words):
        chunk_words = words[i : i + chunk_size_words]
        chunk_text = " ".join(chunk_words)
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=chunk_text,
                source=source_name,
                chapter=None,
            )
        )

    return chunks


def run_pipeline(pdf_path: str) -> List[Fact]:
    # Step 3: End-to-end pipeline: PDF -> Chunks -> Facts.
    chunks = load_pdf_chunks(pdf_path=pdf_path, chunk_size_words=1000)

    all_facts: List[Fact] = []
    for chunk in chunks:
        all_facts.extend(extract_facts(chunk_text=chunk.text, chunk_id=chunk.id))

    return all_facts


def normalize_concept(concept: str) -> str:
    cleaned = concept.lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    tokens = cleaned.split()
    return " ".join(token.title() for token in tokens)


def load_canonical_cache() -> dict[str, Optional[str]]:
    if not CANONICAL_CACHE_PATH.exists():
        return {}

    try:
        with CANONICAL_CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    cache: dict[str, Optional[str]] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        cache[key] = value if isinstance(value, str) else None
    return cache


def save_canonical_cache(cache: dict[str, Optional[str]]) -> None:
    with CANONICAL_CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def canonicalize_concepts(concepts: List[str]) -> dict[str, Optional[str]]:
    if not concepts:
        return {}

    # Step 0: Load cache and only request uncached concepts.
    cache = load_canonical_cache()
    missing = [concept for concept in concepts if concept not in cache]

    if not missing:
        return {name: cache.get(name) for name in concepts}

    # Step 1: Send uncached concepts in one batched prompt.
    prompt = (f"""
    You are given concept names extracted from academic material.

    Your task:
    - Fix spelling errors
    - Fix possessives (e.g., Shannon, Euler, Kerckhoffs)
    - Normalize to standard terminology
    - Remove invalid or vague concepts (return null)

    Rules:
    - Do NOT merge different concepts
    - Do NOT invent new concepts
    - Preserve meaning exactly
    - Keep names concise (1–4 words)

    Return ONLY valid JSON mapping original → fixed (or null):

    {{
      "Concept A": "Fixed Name",
      "Concept B": null
    }}

    Concepts:
    {chr(10).join("- " + c for c in missing)}
    """)

    try:
        response = ollama_client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            max_tokens=600,
        )
        raw_content = response.choices[0].message.content or ""
    except Exception:
        for name in missing:
            cache[name] = None
        save_canonical_cache(cache)
        return {name: cache.get(name) for name in concepts}

    # Step 2: Parse JSON safely.
    try:
        parsed = json.loads(raw_content)
    except Exception:
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            for name in missing:
                cache[name] = None
            save_canonical_cache(cache)
            return {name: cache.get(name) for name in concepts}
        try:
            parsed = json.loads(raw_content[start : end + 1])
        except Exception:
            for name in missing:
                cache[name] = None
            save_canonical_cache(cache)
            return {name: cache.get(name) for name in concepts}

    if not isinstance(parsed, dict):
        for name in missing:
            cache[name] = None
        save_canonical_cache(cache)
        return {name: cache.get(name) for name in concepts}

    # Step 3: Update cache with LLM results and save.
    for name in missing:
        value = parsed.get(name)
        cache[name] = value if isinstance(value, str) else None

    save_canonical_cache(cache)
    return {name: cache.get(name) for name in concepts}


def needs_canonicalization(concept: str) -> bool:
    # Rule 1: single-letter token (e.g., "S").
    if any(len(token) == 1 for token in re.findall(r"[A-Za-z]+", concept)):
        return True

    # Rule 2: unusual spacing or punctuation artifacts.
    if concept != concept.strip():
        return True
    if re.search(r"\s{2,}", concept):
        return True
    if re.search(r"[\-_/]{2,}|[()]{2,}|[,:;.]\s*[,:;.]", concept):
        return True

    # Rule 3: repeated words (case-insensitive).
    words = re.findall(r"[A-Za-z]+", concept.lower())
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            return True

    # Rule 4: mixed casing inside a word (e.g., eCb, iNd).
    for token in re.findall(r"[A-Za-z]+", concept):
        if any(c.islower() for c in token) and any(c.isupper() for c in token):
            if not token.istitle():
                return True

    return False


def group_facts_by_concept(facts: List[Fact]) -> dict[str, List[Fact]]:
    grouped: dict[str, List[Fact]] = {}
    for fact in facts:
        concept_key = normalize_concept(fact.concept)
        grouped.setdefault(concept_key, []).append(fact)
    return grouped


def evaluate_concepts(grouped: dict[str, list]) -> dict:
    concepts = list(grouped.keys())
    total_concepts = len(concepts)
    total_facts = sum(len(grouped[concept]) for concept in concepts)
    avg_facts_per_concept = (total_facts / total_concepts) if total_concepts else 0.0

    singleton_count = sum(1 for concept in concepts if len(grouped[concept]) == 1)
    singleton_ratio = (singleton_count / total_concepts * 100.0) if total_concepts else 0.0

    suspicious_concepts: List[str] = []
    filler_words = {"goals", "impact", "effects"}
    lowercase_acronyms = {"Aes", "Des", "Tls"}

    for concept in concepts:
        words = re.findall(r"[A-Za-z]+", concept)
        words_lower = [w.lower() for w in words]

        has_filler = any(w in filler_words for w in words_lower)
        has_plural = any(w.endswith("s") and len(w) > 1 for w in words_lower)
        has_lowercase_acronym = any(w in lowercase_acronyms for w in words)

        if has_filler or has_plural or has_lowercase_acronym:
            suspicious_concepts.append(concept)

    near_duplicates: List[tuple[str, str]] = []

    def differs_by_one_word(left: str, right: str) -> bool:
        left_words = left.lower().split()
        right_words = right.lower().split()

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
            left_l = left.lower()
            right_l = right.lower()
            is_substring_match = left_l in right_l or right_l in left_l
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


def _concepts_are_similar(left: str, right: str) -> bool:
    left_l = left.lower().strip()
    right_l = right.lower().strip()

    # Rule 1: one concept name is a substring of the other.
    if left_l in right_l or right_l in left_l:
        return True

    left_words = left_l.split()
    right_words = right_l.split()

    # Rule 2a: names differ by one added/removed word.
    if abs(len(left_words) - len(right_words)) == 1:
        shorter = left_words if len(left_words) < len(right_words) else right_words
        longer = right_words if len(right_words) > len(left_words) else left_words
        if all(word in longer for word in shorter):
            return True

    # Rule 2b: same length, exactly one different word.
    if len(left_words) == len(right_words):
        mismatches = sum(1 for a, b in zip(left_words, right_words) if a != b)
        if mismatches == 1:
            return True

    return False


def merge_similar_concepts(grouped: dict[str, List[Fact]]) -> dict[str, List[Fact]]:
    merged: dict[str, List[Fact]] = {}

    for concept, facts in grouped.items():
        target_key = concept
        for existing_key in list(merged.keys()):
            if not _concepts_are_similar(concept, existing_key):
                continue

            # Keep more common name; if tied, keep shorter name.
            if len(merged[existing_key]) > len(facts):
                target_key = existing_key
            elif len(merged[existing_key]) < len(facts):
                target_key = concept
            else:
                target_key = concept if len(concept) < len(existing_key) else existing_key

            break

        if target_key in merged:
            merged[target_key].extend(facts)
        else:
            merged[target_key] = list(facts)

    return merged


if __name__ == "__main__":
    # Step 4: Demo run and print first 5 facts.
    demo_pdf_path = "./example.pdf"
    all_facts = run_pipeline(demo_pdf_path)
    print(f"Extracted {len(all_facts)} facts")
    for fact in all_facts[:5]:
        print(f"{fact.id} | {fact.concept} | {fact.content} "
              f"| chunk={fact.source_chunk_id}")
    
    grouped = group_facts_by_concept(all_facts)

    # Step 5: Canonicalize only concepts that need fixing.
    concept_names = list(grouped.keys())
    concepts_to_fix = [name for name in concept_names if needs_canonicalization(name)]
    canonical_map = canonicalize_concepts(concepts_to_fix)

    # Step 6: Build final grouped map with canonical names.
    final_grouped: dict[str, List[Fact]] = {}
    for concept, facts in grouped.items():
        if concept in canonical_map:
            canonical_name = canonical_map[concept]
            if canonical_name is None:
                continue
            target_name = canonical_name
        else:
            target_name = concept

        final_grouped.setdefault(target_name, []).extend(facts)

    # Step 7: Print final grouped concept counts.
    for concept, facts in final_grouped.items():
        print(concept, "->", len(facts))

    eval_result = evaluate_concepts(final_grouped)

    # Evaluation
    print("\n=== EVALUATION ===")
    for k, v in eval_result.items():
        print(k, ":", v)

    previous_eval = load_previous_evaluation()
    check_evaluation_assertions(eval_result, previous_eval)
    save_evaluation_snapshot(eval_result)