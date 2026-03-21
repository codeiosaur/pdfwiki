from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re
import uuid

from pypdf import PdfReader
from extract.fact_extractor import extract_facts, Fact


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

    # Remove punctuation and normalize separators.
    cleaned = re.sub(r"[-()]", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)

    acronym_map = {
        "aes": "AES",
        "tls": "TLS",
        "rsa": "RSA",
        "ocsp": "OCSP",
    }
    filler_words = {"the", "a", "an", "algorithm", "method", "system", "process", "mechanism"}

    normalized_tokens: List[str] = []
    for token in cleaned.split():
        # Basic plural-to-singular normalization.
        singular = token[:-1] if token.endswith("s") and len(token) > 1 else token

        # Preserve acronym canonical forms.
        if token in acronym_map:
            normalized_tokens.append(acronym_map[token])
            continue
        if singular in acronym_map:
            normalized_tokens.append(acronym_map[singular])
            continue

        if singular in filler_words:
            continue

        normalized_tokens.append(singular.title())

    return " ".join(normalized_tokens)


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

    for concept, facts in list(grouped.items()):
        print(concept, "->", len(facts))