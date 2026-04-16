from typing import List
from extract.fact_extractor import Fact
from transform.matching import (
    has_antonym_conflict,
    has_strong_overlap,
    is_cousin,
    is_duplicate,
    is_sibling,
    tokenize_for_matching,
)

def cluster_related_concepts(grouped: dict[str, List[Fact]]) -> dict[str, List[Fact]]:
    """
    Merge related concepts based on shared FIRST token.

        Rules:
        - Only consider concepts with 2+ words
        - If multiple concepts share the same FIRST token (case-insensitive),
            merge into ONE concept
        - Use an existing concept name as the cluster label:
            most facts, then longest name
        - Concepts not in a merge group remain unchanged
    """
    clustered: dict[str, List[Fact]] = {}

    def longest_common_suffix(concepts: List[str]) -> List[str]:
        token_lists = [tokenize_for_matching(concept) for concept in concepts]
        if not token_lists:
            return []

        reversed_lists = [list(reversed(tokens)) for tokens in token_lists]
        min_len = min(len(tokens) for tokens in reversed_lists)
        shared_reversed: List[str] = []

        for i in range(min_len):
            token = reversed_lists[0][i]
            if all(tokens[i] == token for tokens in reversed_lists[1:]):
                shared_reversed.append(token)
            else:
                break

        return list(reversed(shared_reversed))

    concepts = list(grouped.keys())
    visited: set[str] = set()

    for concept in concepts:
        if concept in visited:
            continue

        cluster_members = [concept]
        visited.add(concept)

        for candidate in concepts:
            if candidate in visited:
                continue
            if has_antonym_conflict(concept, candidate):
                continue
            if is_clusterable(concept, candidate):
                cluster_members.append(candidate)
                visited.add(candidate)

        if len(cluster_members) == 1:
            clustered.setdefault(concept, []).extend(grouped[concept])
            continue

        suffix_tokens = longest_common_suffix(cluster_members)
        if suffix_tokens:
            target = " ".join(token.title() for token in suffix_tokens)
        else:
            target = max(cluster_members, key=lambda c: (len(grouped[c]), len(c)))

        bucket = clustered.setdefault(target, [])
        for member in cluster_members:
            bucket.extend(grouped[member])

    return clustered

def find_head_word(concept: str) -> str:
    """
    Return the last token (normalized).
    """
    tokens = tokenize_for_matching(concept)
    return tokens[-1] if tokens else ""

def is_clusterable(a: str, b: str) -> bool:
    """
    Return True if concepts belong to same cluster.

    Rules:
    - Same head word (last token)
    - NOT siblings (same head but different modifiers — these are
      intentionally distinct concepts like "Inventory Fraud" vs
      "Inventory Valuation")
    - Must share the same first AND last tokens (for multi-word)
    - Do NOT cluster if first tokens are clearly distinct acronyms
    """
    if has_antonym_conflict(a, b):
        return False

    # Siblings are distinct concepts that share a head word but differ
    # in their modifier — e.g., "Inventory Fraud" vs "Inventory Shrinkage".
    # These must NEVER be merged.
    if is_sibling(a, b):
        return False

    # Cousins share all modifiers but differ on the head — e.g., "Fixed Cost"
    # vs "Fixed Asset". These are distinct concepts and must never merge.
    if is_cousin(a, b):
        return False

    head_a = find_head_word(a)
    head_b = find_head_word(b)
    if not head_a or head_a != head_b:
        return False

    tokens_a = tokenize_for_matching(a)
    tokens_b = tokenize_for_matching(b)

    # Only cluster multi-word concepts that share BOTH first and last tokens.
    # This prevents "Inventory Fraud" + "Inventory Valuation" from merging
    # (same first token but different last tokens).
    if (
        len(tokens_a) >= 2
        and len(tokens_b) >= 2
        and tokens_a[0] == tokens_b[0]
        and tokens_a[-1] == tokens_b[-1]
    ):
        return True

    first_a = a.split()[0] if a.split() else ""
    first_b = b.split()[0] if b.split() else ""
    if (
        first_a.isalpha()
        and first_b.isalpha()
        and first_a.isupper()
        and first_b.isupper()
        and first_a != first_b
    ):
        return False

    # Require substantial overlap: shared non-stopword tokens must be
    # at least 50% of the shorter concept's non-stopword tokens.
    stopwords = {"a", "an", "the", "of", "and", "or", "to", "for", "in", "on", "with", "by"}
    core_a = {t for t in tokens_a[:-1] if t not in stopwords}
    core_b = {t for t in tokens_b[:-1] if t not in stopwords}

    overlap = core_a.intersection(core_b)
    shorter_core = min(len(core_a), len(core_b))
    if shorter_core > 0 and len(overlap) >= max(1, shorter_core * 0.5):
        return True

    return False

def _concepts_are_similar(left: str, right: str) -> bool:
    if has_antonym_conflict(left, right):
        return False
    if is_duplicate(left, right):
        return True
    if is_sibling(left, right):
        return False
    if is_cousin(left, right):
        return False

    # Only merge via strong overlap if the shared tokens represent
    # at least 50% of the shorter concept's tokens.  This prevents
    # "Inventory Fraud" from merging with "Inventory Valuation" just
    # because they share the word "inventory".
    left_tokens = tokenize_for_matching(left)
    right_tokens = tokenize_for_matching(right)
    shorter_len = min(len(left_tokens), len(right_tokens))

    if has_strong_overlap(left, right) and shorter_len >= 2:
        # Count total shared tokens (not just the contiguous run)
        shared = len(set(left_tokens).intersection(set(right_tokens)))
        if shared >= max(2, shorter_len * 0.5):
            return True

    return False