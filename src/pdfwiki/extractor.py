"""
PDF text extraction, chapter splitting, and semantic chunking.
Format-agnostic: works on slide decks, textbooks, papers, and notes.
"""

import re
import os
import importlib
import pdfplumber
from pathlib import Path
from pdfwiki.retriever import deduplicate_chunks


# --- Text conversion to Markdown ---

def extract_text_with_markitdown(file_path: str) -> str:
    """
    Extract text from any file format (PDF, PPTX, DOCX, etc.) using markitdown.
    Falls back to pdfplumber for PDFs if markitdown is unavailable.
    Returns markdown-formatted text with page separators.
    
    Requires: pip install markitdown[office] (for Office formats)
              or pip install markitdown[pdf] (for enhanced PDF support)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        markitdown_mod = importlib.import_module("markitdown")
        MarkItDown = getattr(markitdown_mod, "MarkItDown")
    except Exception:
        print(f"  [warn] markitdown not installed, falling back to pdfplumber for {path.suffix} files")
        if path.suffix.lower() == '.pdf':
            return extract_text(file_path, use_markdown=False)
        raise ImportError(
            "markitdown required for non-PDF formats. "
            "Install with: pip install 'markitdown[office]'"
        )
    
    mmd = MarkItDown()
    
    try:
        # Extract markdown directly from the file
        result = mmd.convert(file_path)
        markdown_text = result.text_content
        
        # If no markdown content extracted, fall back to plain extract_text for PDFs.
        # Keep this explicit so users know extraction mode changed.
        if not markdown_text.strip() and path.suffix.lower() == '.pdf':
            print(f"  [warn] markitdown returned empty text for {file_path}; falling back to pdfplumber")
            return extract_text(file_path, use_markdown=False)
        
        # Estimate page count from content size and line count for page markers
        lines = markdown_text.splitlines()
        # Rough estimate: ~50 lines per page in markdown format
        estimated_pages = max(1, len(lines) // 50)
        
        print(f"  Extracted {estimated_pages} estimated pages, {len(markdown_text):,} characters (markdown)")
        return markdown_text
        
    except Exception as e:
        print(f"  [warn] markitdown extraction failed: {e}")
        if path.suffix.lower() == '.pdf':
            print(f"  Falling back to pdfplumber for {file_path}")
            return extract_text(file_path, use_markdown=False)
        raise


# --- Text extraction ---

def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a bool-like environment variable with a safe default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_use_markitdown(use_markdown: bool | None) -> bool:
    """
    Resolve whether MarkItDown should be used.

    Precedence:
    1) explicit function argument
    2) USE_MARKITDOWN env var
    3) PDF_TO_NOTES_USE_MARKITDOWN env var (compat alias)
    4) default False
    """
    if use_markdown is not None:
        return use_markdown

    if "USE_MARKITDOWN" in os.environ:
        return _env_bool("USE_MARKITDOWN", default=False)

    return _env_bool("PDF_TO_NOTES_USE_MARKITDOWN", default=False)

def extract_text(pdf_path: str, use_markdown: bool | None = None) -> str:
    """
    Extract all text from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        use_markdown: If True, use markitdown for extraction (with markdown formatting).
                      If False, use pdfplumber (plain text, faster).
                      If None, resolve from env vars USE_MARKITDOWN /
                      PDF_TO_NOTES_USE_MARKITDOWN.
    
    Returns a single string with page separators.
    """
    use_markitdown = resolve_use_markitdown(use_markdown)
    if use_markitdown:
        print("  Extraction backend: markitdown")
        return extract_text_with_markitdown(pdf_path)
    else:
        print("  Extraction backend: pdfplumber")
    
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

PAGE_MARKER_REGEX = re.compile(r'^--- Page \d+ ---$', re.MULTILINE)
HEADING_REGEX = re.compile(r'(?m)^#{1,6}\s+.+$')


def has_page_markers(text: str) -> bool:
    """Return True if text contains page markers like '--- Page N ---'."""
    return bool(PAGE_MARKER_REGEX.search(text))


def has_headings(text: str) -> bool:
    """Return True if text contains markdown-style headings."""
    return bool(HEADING_REGEX.search(text))


def has_paragraphs(text: str) -> bool:
    """Return True if text appears to contain multiple paragraph breaks."""
    return len(re.findall(r'\n\s*\n', text)) >= 2


def _enforce_max_chunk_size(chunks: list[str], max_chars: int, overlap: int) -> list[str]:
    """Ensure no output chunk exceeds max_chars by re-splitting oversized chunks."""
    bounded: list[str] = []
    for chunk in chunks:
        cleaned = chunk.strip()
        if not cleaned:
            continue
        if len(cleaned) <= max_chars:
            bounded.append(cleaned)
            continue
        bounded.extend(chunk_by_size(cleaned, max_chars=max_chars, overlap=overlap))
    return bounded


def _finalize_chunks(chunks: list[str], max_chars: int, overlap: int) -> list[str]:
    """Normalize, enforce max size, and deduplicate chunk output."""
    normalized = [c.strip() for c in chunks if c and c.strip()]
    bounded = _enforce_max_chunk_size(normalized, max_chars=max_chars, overlap=overlap)
    deduped = deduplicate_chunks(bounded)
    return deduped


def _adaptive_semantic_chunk_config(text: str) -> tuple[int, int]:
    """
    Pick semantic chunk settings based on document shape.
    This keeps chunk counts useful for retrieval across slides and textbooks.
    """
    text_len = len(text.strip())
    if text_len <= 0:
        return 1800, 120

    line_count = max(text.count("\n") + 1, 1)
    avg_line_len = text_len / line_count
    slide_like = line_count >= 180 and avg_line_len <= 90

    # Target a chunk count range that is retrieval-friendly for BM25.
    target_size = 1800 if slide_like else 2600
    target_chunks = max(15, min(100, text_len // target_size))

    max_chars = text_len // max(target_chunks, 1)
    max_chars = max(1200, min(3200, max_chars))
    if slide_like:
        max_chars = min(max_chars, 1800)

    overlap = max(80, min(220, max_chars // 10))
    return max_chars, overlap


def chunk_by_size(text: str, max_chars: int = 6000, overlap: int = 200) -> list[str]:
    """Chunk text by fixed size with overlap for continuity."""
    cleaned = text.strip()
    if not cleaned:
        return []

    step = max(1, max_chars - overlap)
    chunks = []
    for start in range(0, len(cleaned), step):
        piece = cleaned[start:start + max_chars].strip()
        if piece:
            chunks.append(piece)
        if start + max_chars >= len(cleaned):
            break
    return chunks


def chunk_by_headings(
    text: str,
    max_chars: int = 6000,
    overlap: int = 200,
    min_section_chars: int = 1000,
) -> list[str]:
    """
    Split text on markdown headings and group undersized sections.
    Oversized groups are constrained to max_chars in post-processing.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    heading_matches = list(HEADING_REGEX.finditer(cleaned))
    if not heading_matches:
        return []

    sections: list[str] = []
    first_heading_start = heading_matches[0].start()
    if first_heading_start > 0:
        lead = cleaned[:first_heading_start].strip()
        if lead:
            sections.append(lead)

    for i, match in enumerate(heading_matches):
        start = match.start()
        end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(cleaned)
        section = cleaned[start:end].strip()
        if section:
            sections.append(section)

    grouped: list[str] = []
    buffer = ""
    for section in sections:
        if not buffer:
            buffer = section
        elif len(buffer) < min_section_chars:
            buffer = f"{buffer}\n\n{section}".strip()
        else:
            grouped.append(buffer)
            buffer = section

    if buffer:
        grouped.append(buffer)

    return _finalize_chunks(grouped, max_chars=max_chars, overlap=overlap)


def chunk_by_paragraphs(text: str, max_chars: int = 6000, overlap: int = 200) -> list[str]:
    """Chunk text by paragraph boundaries using semantic chunking logic."""
    chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
    return _finalize_chunks(chunks, max_chars=max_chars, overlap=overlap)


def smart_chunk(
    text: str,
    max_chars: int = 6000,
    overlap: int = 200,
    pages_per_chunk: int = 3,
    min_section_chars: int = 1000,
) -> list[str]:
    """
    Select the best chunking strategy based on document structure.

    Priority:
    1) page markers
    2) markdown headings
    3) paragraph boundaries
    4) fixed-size chunking fallback
    """
    if has_page_markers(text):
        strategy = "page-based"
        chunks = chunk_by_page(text, pages_per_chunk=pages_per_chunk)
    elif has_headings(text):
        strategy = "heading-based"
        chunks = chunk_by_headings(
            text,
            max_chars=max_chars,
            overlap=overlap,
            min_section_chars=min_section_chars,
        )
    elif has_paragraphs(text):
        strategy = "paragraph-based"
        adaptive_max, adaptive_overlap = _adaptive_semantic_chunk_config(text)
        chunks = chunk_by_paragraphs(text, max_chars=adaptive_max, overlap=adaptive_overlap)
    else:
        strategy = "size-based"
        adaptive_max, adaptive_overlap = _adaptive_semantic_chunk_config(text)
        chunks = chunk_by_size(text, max_chars=adaptive_max, overlap=adaptive_overlap)

    finalized = _finalize_chunks(chunks, max_chars=max_chars, overlap=overlap)
    print(f"  Chunking strategy: {strategy}")
    print(f"  Total chunks: {len(finalized)}")
    return finalized


def chunk_by_page(text: str, pages_per_chunk: int = 3) -> list[str]:
    """
    Split text into chunks by PDF page boundary.
    Better than character-based chunking for slide PDFs where each page
    covers a distinct topic — keeps concept content together.

    pages_per_chunk: group N pages per chunk (1 = most granular,
                     3 = good balance of context vs separation)
    """
    if not has_page_markers(text):
        return [text.strip()] if text.strip() else []

    # Split on page markers inserted by extract_text()
    pages = re.split(r'--- Page \d+ ---', text)
    pages = [p.strip() for p in pages if p.strip()]

    if not pages:
        return [text]

    chunks = []
    for i in range(0, len(pages), pages_per_chunk):
        group = pages[i:i + pages_per_chunk]
        chunks.append("\n\n".join(group).strip())

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
                # Single paragraph too long — hard split repeatedly.
                start = 0
                step = max(1, max_chars - overlap)
                while start + max_chars < len(para):
                    chunks.append(para[start:start + max_chars].strip())
                    start += step
                current_chunk = para[start:]

            # If carry-over is still oversized, keep splitting until it fits.
            while len(current_chunk) > max_chars:
                chunks.append(current_chunk[:max_chars].strip())
                current_chunk = current_chunk[max(1, max_chars - overlap):]
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