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
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", concept.strip())
    raw_tokens = cleaned.split()

    normalized_tokens: List[str] = []
    for token in raw_tokens:
        if token.isalpha() and token.isupper() and len(token) <= 6:
            normalized_tokens.append(token)
        else:
            normalized_tokens.append(token.lower().title())

    return " ".join(normalized_tokens)


def tokenize_for_matching(name: str) -> list[str]:
    """
    Convert a concept name into normalized tokens for comparison ONLY.
    Do NOT return a string. Do NOT modify original input.

    Rules:
    - lowercase
    - remove punctuation EXCEPT apostrophes inside words
    - split into tokens
    - collapse whitespace
    - singularize ONLY the last token if it ends with 's' and length > 3

    Examples:
    - "Certificate Authorities" -> ["certificate", "authority"]
    - "Anastasia's Revenge" -> ["anastasia's", "revenge"]
    """
    lowered = name.lower()

    # Keep apostrophes only when they are inside a word.
    no_edge_apostrophes = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", " ", lowered)
    cleaned = re.sub(r"[^a-z0-9'\s]", " ", no_edge_apostrophes)

    tokens = cleaned.split()
    if not tokens:
        return tokens

    last = tokens[-1]
    if last.endswith("s") and len(last) > 3 and not last.endswith("'s"):
        if last.endswith("ies") and len(last) > 4:
            tokens[-1] = last[:-3] + "y"
        else:
            tokens[-1] = last[:-1]

    return tokens


def edit_distance_1(left: str, right: str) -> bool:
    if left == right:
        return False
    if abs(len(left) - len(right)) > 1:
        return False

    if len(left) == len(right):
        mismatches = sum(1 for a, b in zip(left, right) if a != b)
        return mismatches == 1

    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    i = 0
    j = 0
    skipped = False
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
            continue
        if skipped:
            return False
        skipped = True
        j += 1
    return True


def is_duplicate(a: str, b: str) -> bool:
    """
    Return True ONLY if concepts are effectively identical.

    Rules:
    - Tokenize using tokenize_for_matching
    - Same number of tokens
    - All tokens match exactly OR
      exactly one token differs by edit distance 1
    """
    left_tokens = tokenize_for_matching(a)
    right_tokens = tokenize_for_matching(b)

    if len(left_tokens) != len(right_tokens):
        return False

    mismatches = [
        (left, right)
        for left, right in zip(left_tokens, right_tokens)
        if left != right
    ]

    if not mismatches:
        return True
    if len(mismatches) != 1:
        return False

    left_token, right_token = mismatches[0]
    return edit_distance_1(left_token, right_token)


def is_sibling(a: str, b: str) -> bool:
    """
    True if concepts share the same head word (last token)
    but have different modifiers.

    Example:
    - "ECB Mode" vs "CBC Mode" -> True
    """
    left_tokens = tokenize_for_matching(a)
    right_tokens = tokenize_for_matching(b)

    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False
    if left_tokens[-1] != right_tokens[-1]:
        return False

    return left_tokens[:-1] != right_tokens[:-1]


def has_strong_overlap(a: str, b: str) -> bool:
    """
    True if concepts share at least 2 tokens in order.

    This replaces naive substring matching.

    Example:
    - "Public Key Cryptography" vs "Key Cryptography" -> True
    - "Key Length" vs "Key Generation" -> False
    """
    left_tokens = tokenize_for_matching(a)
    right_tokens = tokenize_for_matching(b)

    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False

    best_run = 0
    for i in range(len(left_tokens)):
        for j in range(len(right_tokens)):
            run = 0
            while (
                i + run < len(left_tokens)
                and j + run < len(right_tokens)
                and left_tokens[i + run] == right_tokens[j + run]
            ):
                run += 1
            if run > best_run:
                best_run = run
            if best_run >= 2:
                return True

    return False


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


def _concepts_are_similar(left: str, right: str) -> bool:
    if is_duplicate(left, right):
        return True
    if is_sibling(left, right):
        return False
    if has_strong_overlap(left, right):
        return True
    return False


def merge_similar_concepts(grouped: dict[str, List[Fact]]) -> dict[str, List[Fact]]:
    merged: dict[str, List[Fact]] = {}

    for concept, facts in grouped.items():
        merged_with_existing = False
        for existing_key in list(merged.keys()):
            if not _concepts_are_similar(concept, existing_key):
                continue

            # Keep more common name; if tied, keep shorter name.
            if len(merged[existing_key]) > len(facts):
                winner = existing_key
            elif len(merged[existing_key]) < len(facts):
                winner = concept
            else:
                winner = concept if len(concept) < len(existing_key) else existing_key

            if winner == existing_key:
                merged[existing_key].extend(facts)
            else:
                existing_facts = merged.pop(existing_key)
                merged[concept] = list(existing_facts) + list(facts)

            merged_with_existing = True
            break

        if not merged_with_existing:
            merged[concept] = list(facts)

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
            target_name = canonical_name if canonical_name is not None else concept
        else:
            target_name = concept

        final_grouped.setdefault(target_name, []).extend(facts)

    final_grouped = merge_similar_concepts(final_grouped)

    # Step 7: Second-pass canonicalization for suspicious concept labels.
    initial_eval = evaluate_concepts(final_grouped)
    suspicious_to_fix = initial_eval.get("suspicious_concepts", [])
    suspicious_map = canonicalize_concepts(suspicious_to_fix) if suspicious_to_fix else {}

    if suspicious_map:
        refined_grouped: dict[str, List[Fact]] = {}
        for concept, facts in final_grouped.items():
            canonical_name = suspicious_map.get(concept)
            target_name = canonical_name if canonical_name is not None else concept
            refined_grouped.setdefault(target_name, []).extend(facts)
        final_grouped = merge_similar_concepts(refined_grouped)

    # Step 8: Print final grouped concept counts.
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