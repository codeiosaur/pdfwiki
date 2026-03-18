"""
Chunk retrieval module.
Scores text chunks by relevance to a concept using keyword matching.
"""

import re
import math
import hashlib


# Cache for deduplicated chunks to avoid recomputing expensive Jaccard comparisons
_DEDUP_CACHE: dict[str, list[str]] = {}


def _chunks_cache_key(chunks: list[str]) -> str:
    """Generate a cache key based on chunk hashes."""
    content_hash = hashlib.md5(
        "\n".join(chunks[:100]).encode()  # Hash first 100 chunks for speed
    ).hexdigest()
    return f"{len(chunks)}_{content_hash}"


def _normalize_chunk(text: str) -> str:
    """Normalize chunk text for overlap and duplicate detection."""
    return re.sub(r"\s+", " ", text.strip().lower())


def deduplicate_chunks(
    chunks: list[str],
    similarity_threshold: float = 0.9,
    containment_threshold: float = 0.9,
) -> list[str]:
    """
    Remove repeated/near-identical chunks while preserving order.

    This primarily catches overlap artifacts introduced by chunking,
    where adjacent chunks can share large repeated spans.
    
    Uses a cache to avoid recomputing expensive Jaccard comparisons.
    """
    # Check cache first
    cache_key = _chunks_cache_key(chunks)
    if cache_key in _DEDUP_CACHE:
        return _DEDUP_CACHE[cache_key]
    
    unique: list[str] = []
    normalized_seen: list[str] = []

    for chunk in chunks:
        norm = _normalize_chunk(chunk)
        if not norm:
            continue

        is_duplicate = False
        norm_words = set(norm.split())

        for existing in normalized_seen:
            if norm == existing:
                is_duplicate = True
                break

            short, long_ = (norm, existing) if len(norm) <= len(existing) else (existing, norm)
            if len(short) >= 80 and short in long_:
                overlap_ratio = len(short) / max(len(long_), 1)
                if overlap_ratio >= containment_threshold:
                    is_duplicate = True
                    break

            existing_words = set(existing.split())
            union = norm_words | existing_words
            if union:
                jaccard = len(norm_words & existing_words) / len(union)
                if jaccard >= similarity_threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique.append(chunk)
            normalized_seen.append(norm)

    # Cache the result
    _DEDUP_CACHE[cache_key] = unique
    return unique


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


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word-like tokens for lexical scoring."""
    return re.findall(r"[a-z0-9][a-z0-9-]*", text.lower())


def bm25_scores(
    chunks: list[str],
    query_terms: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """
    Compute BM25 relevance score for each chunk against query terms.

    Returns a score list aligned with input chunk order.
    """
    if not chunks:
        return []

    tokenized_docs = [_tokenize(chunk) for chunk in chunks]
    doc_count = len(tokenized_docs)
    doc_lens = [len(doc) for doc in tokenized_docs]
    avg_doc_len = sum(doc_lens) / max(doc_count, 1)

    # Term document frequencies and per-document term frequencies.
    dfs: dict[str, int] = {}
    tfs_per_doc: list[dict[str, int]] = []
    for doc_tokens in tokenized_docs:
        term_freqs: dict[str, int] = {}
        for token in doc_tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1
        tfs_per_doc.append(term_freqs)
        for token in term_freqs.keys():
            dfs[token] = dfs.get(token, 0) + 1

    # Expand multi-word query terms into component tokens.
    query_tokens: list[str] = []
    for term in query_terms:
        query_tokens.extend(_tokenize(term))
    if not query_tokens:
        return [0.0] * doc_count

    unique_query_tokens = list(dict.fromkeys(query_tokens))

    scores: list[float] = []
    for idx, tf_doc in enumerate(tfs_per_doc):
        doc_len = max(doc_lens[idx], 1)
        norm = k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1)))
        score = 0.0

        for token in unique_query_tokens:
            term_freq = tf_doc.get(token, 0)
            if term_freq == 0:
                continue
            df = dfs.get(token, 0)
            # BM25 IDF variant with +1 for numerical stability.
            idf = math.log(1 + ((doc_count - df + 0.5) / (df + 0.5)))
            score += idf * ((term_freq * (k1 + 1)) / (term_freq + norm))

        scores.append(score)

    return scores


def score_chunk(chunk: str, concept: str, related_concepts: list[str]) -> float:
    """
    Score a chunk's relevance to a concept.

    Scoring:
    - Primary concept keywords: 3 points each hit
    - Full concept name phrase: 5 bonus points per occurrence
    - Related concept keywords: 0.5 points each (context signal, not primary)
    """
    chunk_lower = chunk.lower()

    primary_keywords = _keywords_from_concept(concept)
    primary_hits = sum(chunk_lower.count(kw) for kw in primary_keywords)
    exact_phrase_hits = chunk_lower.count(concept.lower())

    related_hits = 0
    for related in related_concepts:
        related_keywords = _keywords_from_concept(related)
        related_hits += sum(chunk_lower.count(kw) for kw in related_keywords)

    # Slightly favor dense matches over very long, weakly related chunks.
    # This keeps retrieval quality high without requiring embeddings.
    chunk_words = max(len(re.findall(r"\w+", chunk_lower)), 1)
    density_boost = (primary_hits + exact_phrase_hits * 2) / chunk_words

    score = 0.0
    score += primary_hits * 3.0
    score += exact_phrase_hits * 5.0
    score += related_hits * 0.5
    score += density_boost * 50.0
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
    ranked_chunks = retrieve_ranked_chunks(
        chunks=chunks,
        concept=concept,
        related_concepts=related_concepts,
        top_k=top_k,
    )
    return limit_context(ranked_chunks, max_chars=max_chars)


def retrieve_ranked_chunks(
    chunks: list[str],
    concept: str,
    related_concepts: list[str] = [],
    top_k: int = 3,
) -> list[str]:
    """Return top-ranked relevant chunks before context-size limiting."""
    ranked_with_scores = retrieve_ranked_chunks_with_scores(
        chunks=chunks,
        concept=concept,
        related_concepts=related_concepts,
        top_k=top_k,
    )
    return [chunk for chunk, _ in ranked_with_scores]


def retrieve_ranked_chunks_with_scores(
    chunks: list[str],
    concept: str,
    related_concepts: list[str] = [],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Return top-ranked relevant chunks with their relevance scores.
    
    Returns list of (chunk, score) tuples sorted by score descending.
    """
    if not chunks:
        return []

    deduped_chunks = deduplicate_chunks(chunks)
    if not deduped_chunks:
        return []

    if len(deduped_chunks) == 1:
        return [(deduped_chunks[0], 1.0)]

    related = related_concepts or []

    # BM25 query terms prioritize the target concept and then nearby concepts.
    query_terms = _keywords_from_concept(concept)
    for related_concept in related:
        query_terms.extend(_keywords_from_concept(related_concept))
    lexical_scores = bm25_scores(deduped_chunks, query_terms)

    scored = [
        (
            lexical_scores[i] * 4.0 + score_chunk(chunk, concept, related),
            i,
            chunk,
        )
        for i, chunk in enumerate(deduped_chunks)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    if scored[0][0] == 0:
        return [(deduped_chunks[0], 0.0)]

    # Return (chunk, score) tuples
    return [(chunk, score) for score, _, chunk in scored[:top_k]]


def _compute_adaptive_context_size(
    avg_score: float,
    base_max_chars: int,
    min_factor: float = 0.75,
    max_factor: float = 1.25,
) -> int:
    """
    Scale context size based on BM25 relevance score.
    
    High score (tight signal) → smaller context needed
    Low score (weak signal) → larger context for quality
    
    avg_score: average BM25 relevance score (higher = better match)
    base_max_chars: default context size from profile
    """
    base = max(1200, int(base_max_chars))
    min_chars = max(1000, int(base * min_factor))
    max_chars = max(min_chars + 300, int(base * max_factor))

    # Normalize score to [0, 1]. Higher score means stronger match.
    normalized = min(max(avg_score, 0.0) / 20.0, 1.0)

    # Inverse scaling around the profile budget:
    # strong signal -> smaller context, weak signal -> larger context.
    context_size = max_chars - (normalized * (max_chars - min_chars))
    return max(min_chars, min(int(context_size), max_chars))


def limit_context(chunks: list[str], max_chars: int = 4000) -> str:
    """Join chunks into a single context string under a max char budget."""
    if not chunks:
        return ""

    separator = "\n\n---\n\n"
    selected: list[str] = []
    total_chars = 0

    for chunk in chunks:
        sep_len = len(separator) if selected else 0
        required = sep_len + len(chunk)

        if total_chars + required > max_chars:
            remaining = max_chars - total_chars
            # Only append a truncated chunk when there is still meaningful
            # payload room left after accounting for a separator (if needed).
            payload_room = remaining - sep_len
            if payload_room > 500:
                selected.append(chunk[:payload_room])
            break

        selected.append(chunk)
        total_chars += required

    return separator.join(selected)

def find_related_concepts(text: str, concept_names: list[str]) -> list[str]:
    found = set()
    text_lower = text.lower()

    for concept in concept_names:
        if concept.lower() in text_lower:
            found.add(concept)

    return list(found)

def build_concept_graph(concepts: list[str], chunks: list[str]) -> dict:
    graph = {c: set() for c in concepts}

    for chunk in chunks:
        present = [c for c in concepts if c.lower() in chunk.lower()]

        for c1 in present:
            for c2 in present:
                if c1 != c2:
                    graph[c1].add(c2)

    return graph