"""Concept quality and deduplication utilities.

This module isolates concept-level heuristics from orchestration so the main
pipeline can stay focused on control flow.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher, get_close_matches

REGEX_PAREN = re.compile(r"^(.+?)\s*\((.+?)\)")
TAIL_EQUIVALENTS = {
    frozenset(("encryption", "cryptography")),
    frozenset(("encryption", "cipher")),
    frozenset(("cryptography", "cipher")),
}


def _tail_equivalent_with_shared_stem(norm_candidate: str, norm_existing: str) -> bool:
    c_tokens = norm_candidate.split()
    e_tokens = norm_existing.split()
    if len(c_tokens) < 2 or len(e_tokens) < 2:
        return False

    # If stems align and final taxonomy term differs by known equivalent,
    # treat as near-duplicate: "symmetric key cryptography" ~= "symmetric key encryption".
    if c_tokens[:-1] == e_tokens[:-1] and c_tokens[-1] != e_tokens[-1]:
        return frozenset((c_tokens[-1], e_tokens[-1])) in TAIL_EQUIVALENTS

    return False


def normalize_concept(name: str) -> str:
    """Normalize concept names for fuzzy comparison.

    Examples:
    - "IND-CPA (Indistinguishability...)" -> "ind-cpa"
    - "AES_(Advanced_Encryption_Standard)" -> "aes"
    """
    normalized = name.replace("_", " ")
    normalized = re.sub(r"%20", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s*\([^)]+\)", "", normalized).strip().lower()
    normalized = re.sub(r"[\-–—/:]+", " ", normalized)
    normalized = re.sub(r"\s*(?:->|=>|→|⇒|<-|←)\s*", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def concept_tokens(name: str) -> set[str]:
    """Tokenize a normalized concept for overlap checks."""
    return set(re.findall(r"[a-z0-9]+", normalize_concept(name)))


def concept_aliases(name: str) -> set[str]:
    """Generate likely aliases (short name, abbreviation, normalized form)."""
    aliases = {normalize_concept(name)}
    paren_match = REGEX_PAREN.match(name)
    if paren_match:
        short = normalize_concept(paren_match.group(1).strip())
        expanded = normalize_concept(paren_match.group(2).strip())
        aliases.add(short)
        aliases.add(expanded)

    # Add acronym alias from phrase words (e.g. "one time pad" -> "otp").
    norm_name = normalize_concept(name)
    words = [w for w in re.findall(r"[a-z0-9]+", norm_name) if w]
    if len(words) >= 2:
        acronym = "".join(w[0] for w in words if w[0].isalnum())
        if len(acronym) >= 2:
            aliases.add(acronym)
    return {a for a in aliases if a}


def has_modifier_conflict(norm_candidate: str, norm_existing: str) -> bool:
    """Guard against collapsing semantically opposite modifiers.

    Example:
    - symmetric encryption
    - asymmetric encryption
    """
    c_tokens = norm_candidate.split()
    e_tokens = norm_existing.split()
    if len(c_tokens) < 2 or len(e_tokens) < 2:
        return False

    if c_tokens[-1] != e_tokens[-1]:
        return False

    c_first = c_tokens[0]
    e_first = e_tokens[0]
    if c_first == e_first:
        return False

    if c_first == f"a{e_first}" or e_first == f"a{c_first}":
        return True
    if c_first == f"non{e_first}" or e_first == f"non{c_first}":
        return True
    return False


def is_token_level_typo_variant(norm_candidate: str, norm_existing: str) -> bool:
    """Detect one-token spelling variants (e.g., Kasisky vs Kasiski)."""
    c_tokens = norm_candidate.split()
    e_tokens = norm_existing.split()
    if len(c_tokens) != len(e_tokens):
        return False

    equal = 0
    typo_pairs: list[tuple[str, str]] = []
    for c_tok, e_tok in zip(c_tokens, e_tokens):
        if c_tok == e_tok:
            equal += 1
            continue
        typo_pairs.append((c_tok, e_tok))

    if equal < max(1, len(c_tokens) - 1):
        return False
    if not typo_pairs:
        return False

    for c_tok, e_tok in typo_pairs:
        sim = SequenceMatcher(None, c_tok, e_tok).ratio()
        if sim < 0.84:
            return False
    return True


def is_safe_near_duplicate(candidate: str, existing: str) -> bool:
    """Conservative guard to prevent merging distinct concepts."""
    norm_candidate = normalize_concept(candidate)
    norm_existing = normalize_concept(existing)

    if not norm_candidate or not norm_existing:
        return False
    if norm_candidate == norm_existing:
        return True

    if has_modifier_conflict(norm_candidate, norm_existing):
        return False

    if is_token_level_typo_variant(norm_candidate, norm_existing):
        return True

    if _tail_equivalent_with_shared_stem(norm_candidate, norm_existing):
        return True

    alias_overlap = concept_aliases(candidate) & concept_aliases(existing)
    if alias_overlap:
        return True

    candidate_tokens = concept_tokens(candidate)
    existing_tokens = concept_tokens(existing)
    overlap = candidate_tokens & existing_tokens
    if not overlap:
        return False

    token_union = candidate_tokens | existing_tokens
    token_jaccard = len(overlap) / max(len(token_union), 1)
    norm_similarity = SequenceMatcher(None, norm_candidate, norm_existing).ratio()

    if norm_candidate.startswith(norm_existing) or norm_existing.startswith(norm_candidate):
        return len(min(norm_candidate, norm_existing, key=len)) >= 6

    return norm_similarity >= 0.9 and token_jaccard >= 0.4


def find_near_duplicate(concept: str, existing_names: list[str], cutoff: float = 0.82) -> str | None:
    """Find a near-duplicate concept name in an existing name set."""
    for existing in existing_names:
        if is_safe_near_duplicate(concept, existing):
            return existing

    close = get_close_matches(concept, existing_names, n=1, cutoff=cutoff)
    if close and is_safe_near_duplicate(concept, close[0]):
        return close[0]

    norm_concept = normalize_concept(concept)
    norm_pages = {normalize_concept(p): p for p in existing_names}
    close_norm = get_close_matches(norm_concept, list(norm_pages.keys()), n=1, cutoff=0.75)
    if close_norm:
        candidate = norm_pages[close_norm[0]]
        if is_safe_near_duplicate(concept, candidate):
            return candidate

    for norm_page, original_page in norm_pages.items():
        shorter = min(norm_concept, norm_page, key=len)
        longer = max(norm_concept, norm_page, key=len)
        if len(shorter) >= 6 and longer.startswith(shorter) and is_safe_near_duplicate(concept, original_page):
            return original_page

    return None


def concept_has_source_evidence(concept: str, chunks: list[str]) -> bool:
    """Require concept-like evidence in extracted text to reduce hallucinations."""
    if not chunks:
        return False

    aliases = {concept.lower()}
    paren_match = REGEX_PAREN.match(concept)
    if paren_match:
        aliases.add(paren_match.group(1).strip().lower())
        aliases.add(paren_match.group(2).strip().lower())

    for chunk in chunks:
        chunk_l = chunk.lower()
        if any(alias and alias in chunk_l for alias in aliases):
            return True

    tokens = [t for t in re.findall(r"[a-z0-9]+", concept.lower()) if len(t) >= 4]
    if not tokens:
        return False

    for chunk in chunks:
        chunk_l = chunk.lower()
        hits = sum(1 for t in tokens if t in chunk_l)
        if hits >= min(2, len(tokens)):
            return True

    return False


def filter_concepts_with_evidence(concepts: list[str], chunks: list[str]) -> tuple[list[str], list[str]]:
    """Filter concepts lacking evidence, with low-signal fallback."""
    kept: list[str] = []
    dropped: list[str] = []
    for concept in concepts:
        if concept_has_source_evidence(concept, chunks):
            kept.append(concept)
        else:
            dropped.append(concept)

    if concepts and (not kept or (len(kept) / len(concepts)) < 0.5):
        return concepts, []

    return kept, dropped


def dedupe_concepts_for_run(
    concepts: list[str],
    existing_concepts: list[str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Dedupe near-identical concept names within the same run and against existing concepts.
    
    Args:
        concepts: Concepts extracted from current PDF
        existing_concepts: Concepts from previous PDFs in batch (optional)
    
    Returns:
        (deduplicated_concepts, dropped_pairs)
    """
    # Start with existing concepts from earlier PDFs (master list)
    all_kept = list(existing_concepts) if existing_concepts else []
    dropped_pairs: list[tuple[str, str]] = []

    for concept in concepts:
        duplicate_of = find_near_duplicate(concept, all_kept, cutoff=0.9) if all_kept else None
        if duplicate_of and is_safe_near_duplicate(concept, duplicate_of):
            dropped_pairs.append((concept, duplicate_of))
            continue
        all_kept.append(concept)

    # Return only the newly-kept concepts (not existing ones)
    # Caller will maintain the master list separately
    new_concepts = all_kept[len(existing_concepts) if existing_concepts else 0:]
    return new_concepts, dropped_pairs
