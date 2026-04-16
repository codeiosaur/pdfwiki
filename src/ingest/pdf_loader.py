# src/ingest/pdf_loader.py
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import re
import statistics
import uuid

# Typographic Unicode → ASCII replacements common in academic PDFs.
# Keeps structure intact (bullets, dashes) while avoiding codec errors in
# HTTP clients that default to ASCII encoding.
_UNICODE_REPLACEMENTS = str.maketrans({
    "\u2014": "--",   # em dash
    "\u2013": "-",    # en dash
    "\u2012": "-",    # figure dash
    "\u2015": "--",   # horizontal bar
    "\u2018": "'",    # left single quotation mark
    "\u2019": "'",    # right single quotation mark
    "\u201c": '"',    # left double quotation mark
    "\u201d": '"',    # right double quotation mark
    "\u2026": "...",  # horizontal ellipsis
    "\u00a0": " ",    # non-breaking space
    "\u00ad": "",     # soft hyphen (invisible, safe to drop)
    "\u2022": "*",    # bullet
})


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chapter: str | None


def _chunk_text(
    full_text: str,
    source_name: str,
    min_chunk_words: int = 800,
    max_chunk_words: int = 1200,
) -> Tuple[List[Chunk], int]:
    """
    Split *full_text* into sentence-packed Chunks.

    Returns ``(chunks, fallback_slices)`` where *fallback_slices* counts how
    many chunks were produced by word-level fallback splitting (oversized
    single sentences, or the whole-document fallback for structure-free text).
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full_text) if s.strip()]
    chunks: List[Chunk] = []
    fallback_slices = 0

    current_sentences: List[str] = []
    current_word_count = 0

    def flush_current() -> None:
        nonlocal current_sentences, current_word_count
        if not current_sentences:
            return
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=" ".join(current_sentences),
                source=source_name,
                chapter=None,
            )
        )
        current_sentences = []
        current_word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        if sentence_word_count == 0:
            continue

        # Fallback: if a single sentence is too large, split it by words.
        if sentence_word_count > max_chunk_words:
            flush_current()
            for i in range(0, sentence_word_count, max_chunk_words):
                part_words = sentence_words[i : i + max_chunk_words]
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        text=" ".join(part_words),
                        source=source_name,
                        chapter=None,
                    )
                )
                fallback_slices += 1
            continue

        # Start a new chunk when current one is already in-range and next sentence would overflow.
        if (
            current_sentences
            and current_word_count >= min_chunk_words
            and current_word_count + sentence_word_count > max_chunk_words
        ):
            flush_current()

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    # Emit any remaining text as the final chunk.
    flush_current()

    # If sentence splitting produced no chunks (e.g., unusual formatting), fall back to word slicing.
    if not chunks:
        words = full_text.split()
        for i in range(0, len(words), max_chunk_words):
            chunk_words = words[i : i + max_chunk_words]
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=" ".join(chunk_words),
                    source=source_name,
                    chapter=None,
                )
            )
            fallback_slices += 1

    return chunks, fallback_slices


def load_pdf_chunks(
    pdf_path: str,
    min_chunk_words: int = 800,
    max_chunk_words: int = 1200,
) -> List[Chunk]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "The 'pdfplumber' package is required to load PDF files. Install it with: pip install pdfplumber"
        ) from exc

    if min_chunk_words < 1:
        min_chunk_words = 1
    if max_chunk_words < min_chunk_words:
        max_chunk_words = min_chunk_words

    page_texts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_texts.append(page.extract_text() or "")

    full_text = "\n".join(page_texts)
    full_text = full_text.translate(_UNICODE_REPLACEMENTS)

    source_name = Path(pdf_path).name
    chunks, fallback_slices = _chunk_text(full_text, source_name, min_chunk_words, max_chunk_words)

    if chunks:
        word_counts = [len(c.text.split()) for c in chunks]
        med = statistics.median(word_counts)
        print(
            f"Loaded {len(chunks)} chunks from \"{source_name}\" | "
            f"words min/med/max={min(word_counts)}/{med:.0f}/{max(word_counts)} | "
            f"fallback_slices={fallback_slices}"
        )

    return chunks
