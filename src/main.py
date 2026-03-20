from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import uuid

from pypdf import PdfReader


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chapter: Optional[str]


@dataclass
class Fact:
    id: str
    concept: str
    content: str
    source_chunk_id: str


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


def stub_llm_extract_atomic_facts(chunk: Chunk) -> List[dict]:
    # Step 3: Stub LLM call (replace with a real LLM call later).
    return [
        {
            "concept": "Dummy Concept",
            "content": f"Dummy fact extracted from chunk {chunk.id[:8]}",
        }
    ]


def extract_facts_from_chunk(chunk: Chunk) -> List[Fact]:
    # Step 4: Convert stub LLM output into Fact objects.
    raw_facts = stub_llm_extract_atomic_facts(chunk)
    facts: List[Fact] = []

    for raw in raw_facts:
        facts.append(
            Fact(
                id=str(uuid.uuid4()),
                concept=raw["concept"],
                content=raw["content"],
                source_chunk_id=chunk.id,
            )
        )

    return facts


def run_pipeline(pdf_path: str) -> List[Fact]:
    # Step 5: End-to-end pipeline: PDF -> Chunks -> Facts.
    chunks = load_pdf_chunks(pdf_path=pdf_path, chunk_size_words=1000)

    all_facts: List[Fact] = []
    for chunk in chunks:
        all_facts.extend(extract_facts_from_chunk(chunk))

    return all_facts


if __name__ == "__main__":
    # Step 6: Demo run and print first 5 facts.
    demo_pdf_path = "./example.pdf"
    facts = run_pipeline(demo_pdf_path)
    print(f"Extracted {len(facts)} facts")
    for fact in facts[:5]:
        print(f"{fact.id} | {fact.concept} | {fact.content} | chunk={fact.source_chunk_id}")