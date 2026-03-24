from typing import List
from extract.fact_extractor import Fact
from transform.matching import (
    has_antonym_conflict,
    has_strong_overlap,
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
    groups_by_first: dict[str, List[str]] = {}
    for concept in grouped.keys():
        words = concept.split()
        if len(words) < 2:
            continue
        first_token = words[0].lower()
        groups_by_first.setdefault(first_token, []).append(concept)

    merged_keys: set[str] = set()
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

    for _, concepts in groups_by_first.items():
        if len(concepts) < 2:
            continue

        suffix_tokens = longest_common_suffix(concepts)
        if suffix_tokens:
            target = " ".join(token.title() for token in suffix_tokens)
        else:
            target = max(concepts, key=lambda c: (len(grouped[c]), len(c)))

        bucket = clustered.setdefault(target, [])
        for concept in concepts:
            bucket.extend(grouped[concept])
            merged_keys.add(concept)

    for concept, facts in grouped.items():
        if concept in merged_keys:
            continue
        clustered.setdefault(concept, []).extend(facts)

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
    - At least one shared non-stopword token OR both are 2-3 words
    - Do NOT cluster if first tokens are clearly distinct acronyms
      (e.g., ECB vs CBC)
    """
    if has_antonym_conflict(a, b):
        return False

    head_a = find_head_word(a)
    head_b = find_head_word(b)
    if not head_a or head_a != head_b:
        return False

    tokens_a = tokenize_for_matching(a)
    tokens_b = tokenize_for_matching(b)

    # Cluster multi-word concepts that share the same first token.
    if (
        len(tokens_a) >= 2
        and len(tokens_b) >= 2
        and tokens_a[0] == tokens_b[0]
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

    stopwords = {"a", "an", "the", "of", "and", "or", "to", "for", "in", "on", "with", "by"}
    core_a = {t for t in tokens_a[:-1] if t not in stopwords}
    core_b = {t for t in tokens_b[:-1] if t not in stopwords}

    if core_a.intersection(core_b):
        return True

    return 2 <= len(tokens_a) <= 3 and 2 <= len(tokens_b) <= 3

def _concepts_are_similar(left: str, right: str) -> bool:
    if has_antonym_conflict(left, right):
        return False
    if is_duplicate(left, right):
        return True
    if is_sibling(left, right):
        return False
    if has_strong_overlap(left, right):
        return True
    return False