from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import re
import uuid

from pypdf import PdfReader
from extract.fact_extractor import extract_facts, Fact, ollama_client, OLLAMA_MODEL


CANONICAL_CACHE_PATH = Path(__file__).with_name("canonical_cache.json")


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
    normalized = concept.strip()
    lowered = normalized.lower()

    # Rule 1a: single-letter tokens (e.g., "S").
    if any(len(token) == 1 for token in re.findall(r"[A-Za-z]+", normalized)):
        return True

    # Rule 1b: unusual capitalization in alphabetic tokens.
    alpha_tokens = re.findall(r"[A-Za-z]+", normalized)
    for token in alpha_tokens:
        if not (token.islower() or token.istitle() or token.isupper()):
            return True

    # Rule 2: concept has more than 3 words.
    if len(lowered.split()) > 3:
        return True

    # Rule 3: concept contains filler phrases.
    filler_phrases = {
        "goal of",
        "impact",
        "effects of",
        "importance of",
        "role of",
    }
    if any(phrase in lowered for phrase in filler_phrases):
        return True

    # Rule 1c: likely misspellings via uncommon-word heuristic.
    common_words = {
        "aes", "rsa", "tls", "ocsp", "ssl", "md5", "sha", "sha1", "sha2", "sha3",
        "cryptography", "crypto", "encryption", "decryption", "cipher", "hash", "signature",
        "certificate", "protocol", "attack", "mode", "padding", "nonce", "key", "keys",
        "symmetric", "asymmetric", "public", "private", "one", "time", "pad", "block",
        "stream", "diffie", "hellman", "authority", "revocation", "list", "security",
        "transport", "layer", "standard", "algorithm", "method", "system", "process",
        "mechanism", "function",
    }
    for token in re.findall(r"[a-z]+", lowered):
        if len(token) >= 5 and token not in common_words:
            return True

    return False


def group_facts_by_concept(facts: List[Fact]) -> dict[str, List[Fact]]:
    grouped: dict[str, List[Fact]] = {}
    for fact in facts:
        concept_key = normalize_concept(fact.concept)
        grouped.setdefault(concept_key, []).append(fact)
    return grouped


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