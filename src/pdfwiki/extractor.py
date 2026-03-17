"""
PDF text extraction, chapter splitting, and semantic chunking.
Format-agnostic: works on slide decks, textbooks, papers, and notes.
"""

import re
import pdfplumber
from pathlib import Path


# --- Text extraction ---

def extract_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    Returns a single string with page separators.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"--- Page {i + 1} ---\n{text.strip()}")

    if not pages:
        raise ValueError(f"No text could be extracted from: {pdf_path}")

    full_text = "\n\n".join(pages)
    print(f"  Extracted {len(pages)} pages, {len(full_text):,} characters")
    return full_text


# --- Chapter / section detection ---

# Patterns that indicate a new chapter or major section boundary
CHAPTER_PATTERNS = [
    r'^chapter\s+\d+',                    # "Chapter 1", "Chapter 12"
    r'^chapter\s+[ivxlcdm]+',             # "Chapter IV" (Roman numerals)
    r'^\d{1,2}\.\s+[A-Z][a-zA-Z\s]{3,}', # "1. Introduction to..."
    r'^part\s+\d+',                        # "Part 1"
    r'^part\s+[ivxlcdm]+',                # "Part II"
    r'^section\s+\d+',                    # "Section 3"
    r'^[A-Z\s]{8,}$',                     # ALL CAPS HEADINGS (common in textbooks)
    r'^--- Page \d+ ---$',                # Slide deck page breaks (fallback)
]

CHAPTER_REGEX = re.compile(
    '|'.join(CHAPTER_PATTERNS),
    re.IGNORECASE | re.MULTILINE
)


def chunk_by_page(text: str, pages_per_chunk: int = 3) -> list[str]:
    """
    Split text into chunks by PDF page boundary.
    Better than character-based chunking for slide PDFs where each page
    covers a distinct topic — keeps concept content together.

    pages_per_chunk: group N pages per chunk (1 = most granular,
                     3 = good balance of context vs separation)
    """
    # Split on page markers inserted by extract_text()
    pages = re.split(r'--- Page \d+ ---', text)
    pages = [p.strip() for p in pages if p.strip()]

    if not pages:
        return [text]

    chunks = []
    for i in range(0, len(pages), pages_per_chunk):
        group = pages[i:i + pages_per_chunk]
        chunks.append("\n\n".join(group))

    return chunks


def split_into_chapters(text: str, min_chapter_length: int = 500) -> list[dict]:
    """
    Split extracted text into chapters or major sections.
    Works on textbooks (Chapter headings), slide decks (page breaks),
    and papers (section headings).

    Returns list of dicts: [{title, content, start_page}]
    """
    lines = text.splitlines()
    sections = []
    current_title = "Introduction"
    current_lines = []
    current_start = 1

    for line in lines:
        # Check if this line looks like a chapter/section header
        stripped = line.strip()
        if stripped and CHAPTER_REGEX.match(stripped):
            # Save previous section if it has enough content
            content = "\n".join(current_lines).strip()
            if len(content) >= min_chapter_length:
                sections.append({
                    "title": current_title,
                    "content": content,
                    "start_page": current_start
                })
            # Start new section
            current_title = stripped
            current_lines = []
            # Try to extract page number from "--- Page N ---" markers nearby
            page_match = re.search(r'Page (\d+)', stripped)
            current_start = int(page_match.group(1)) if page_match else current_start + 1
        else:
            current_lines.append(line)

    # Don't forget the last section
    content = "\n".join(current_lines).strip()
    if content:
        sections.append({
            "title": current_title,
            "content": content,
            "start_page": current_start
        })

    # If no chapters detected, treat whole document as one section
    if len(sections) <= 1:
        print("  No chapter boundaries detected — treating as single document")
        return [{"title": "Full Document", "content": text, "start_page": 1}]

    print(f"  Detected {len(sections)} chapters/sections")
    return sections


# --- Semantic chunking ---

def chunk_text(text: str, max_chars: int = 6000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks that fit within the model context.
    Tries to split on paragraph boundaries where possible.
    Overlap ensures concepts that span chunk boundaries aren't lost.
    """
    # Try to split on paragraph breaks first
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Overlap: carry last N chars into next chunk for context continuity
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                # Single paragraph too long — hard split
                chunks.append(para[:max_chars])
                current_chunk = para[max_chars - overlap:]
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# --- Relevance retrieval ---

def get_relevant_chunks(text: str, concept: str, max_chunks: int = 2, max_chars: int = 6000) -> str:
    """
    Return the chunks of text most relevant to a given concept.
    Uses keyword matching against concept name and its likely synonyms.
    Falls back to first chunk if no strong match found.

    max_chunks: how many chunks to include (2 keeps cost low while
                catching concepts that span a page boundary)
    """
    chunks = chunk_text(text, max_chars=max_chars)

    if len(chunks) == 1:
        return chunks[0]

    # Build keyword list from concept name
    # e.g. "Kasisky Test" → ["kasisky", "test", "kasisky test"]
    keywords = [w.lower() for w in concept.split() if len(w) > 2]
    keywords.append(concept.lower())

    # Score each chunk by keyword hit count
    scored = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = sum(chunk_lower.count(kw) for kw in keywords)
        scored.append((score, i, chunk))

    # Sort by score descending, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, _, chunk in scored[:max_chunks]]

    # If top score is 0, nothing matched — use first chunk as fallback
    if scored[0][0] == 0:
        return chunks[0]

    return "\n\n---\n\n".join(top_chunks)