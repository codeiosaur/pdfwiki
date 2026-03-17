"""
Chunk retrieval module.
Scores text chunks by relevance to a concept using keyword matching.
No embeddings or vector DB needed — simple and fast for this use case.
"""

import re


def _keywords_from_concept(concept: str) -> list[str]:
    """
    Extract searchable keywords from a concept name.
    Filters out short stop words that would match too broadly.
    e.g. "Kasisky Test" → ["kasisky", "test", "kasisky test"]
    e.g. "One Time Pad (OTP)" → ["one", "time", "pad", "otp", "one time pad"]
    """
    # Strip parenthetical abbreviations like "(OTP)" into their own keyword
    abbrevs = re.findall(r'\(([^)]+)\)', concept)
    clean = re.sub(r'\([^)]+\)', '', concept).strip()

    words = [w.lower() for w in clean.split() if len(w) > 2]
    keywords = words + [w.lower() for w in abbrevs]
    keywords.append(clean.lower())  # full concept name as a phrase

    return list(set(keywords))


def score_chunk(chunk: str, concept: str, related_concepts: list[str]) -> float:
    """
    Score a chunk's relevance to a concept.

    Scoring:
    - Primary concept keywords: 3 points each hit
    - Full concept name phrase: 5 bonus points per occurrence
    - Related concept keywords: 0.5 points each (context signal, not primary)
    """
    chunk_lower = chunk.lower()
    score = 0.0

    # Primary concept scoring
    primary_keywords = _keywords_from_concept(concept)
    for kw in primary_keywords:
        count = chunk_lower.count(kw)
        score += count * 3

    # Bonus for exact concept name match
    score += chunk_lower.count(concept.lower()) * 5

    # Related concept context scoring
    for related in related_concepts:
        related_keywords = _keywords_from_concept(related)
        for kw in related_keywords:
            score += chunk_lower.count(kw) * 0.5

    return score


def retrieve_chunks(
    chunks: list[str],
    concept: str,
    related_concepts: list[str] = [],
    top_k: int = 3,
    max_chars: int = 4000
) -> str:
    """
    Return the most relevant chunks for a concept, up to max_chars total.

    chunks: all text chunks from the PDF
    concept: the concept we're generating a wiki page for
    related_concepts: other concepts in the index (for context scoring)
    top_k: max number of chunks to include
    max_chars: hard character limit on returned text

    Returns concatenated relevant chunks as a single string.
    """
    if not chunks:
        return ""

    if len(chunks) == 1:
        return chunks[0][:max_chars]

    related = related_concepts or []

    # Score all chunks
    scored = [
        (score_chunk(chunk, concept, related), i, chunk)
        for i, chunk in enumerate(chunks)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    # If nothing scored, fall back to first chunk
    if scored[0][0] == 0:
        return chunks[0][:max_chars]

    # Take top_k chunks, respecting max_chars budget
    selected = []
    total_chars = 0

    for score, idx, chunk in scored[:top_k]:
        if total_chars + len(chunk) > max_chars:
            # Add truncated version if we have room for at least 500 chars
            remaining = max_chars - total_chars
            if remaining > 500:
                selected.append(chunk[:remaining])
            break
        selected.append(chunk)
        total_chars += len(chunk)

    return "\n\n---\n\n".join(selected)