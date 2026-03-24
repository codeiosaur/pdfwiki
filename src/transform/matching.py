import re

ANTONYM_TOKEN_PAIRS = {
    ("first", "last"),
    ("periodic", "perpetual"),
}

def tokenize_for_matching(name: str) -> list[str]:
    """
    Convert a concept name into normalized tokens for comparison ONLY.
    Do NOT return a string. Do NOT modify original input.

    Rules:
    - lowercase
    - remove punctuation EXCEPT apostrophes inside words
    - split into tokens
    - collapse whitespace
    - singularize ONLY the last token if it ends with 's' and length > 3

    Examples:
    - "Certificate Authorities" -> ["certificate", "authority"]
    - "Anastasia's Revenge" -> ["anastasia's", "revenge"]
    """
    lowered = name.lower()

    # Keep apostrophes only when they are inside a word.
    no_edge_apostrophes = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", " ", lowered)
    cleaned = re.sub(r"[^a-z0-9'\s]", " ", no_edge_apostrophes)

    tokens = cleaned.split()
    if not tokens:
        return tokens

    last = tokens[-1]
    if last.endswith("s") and len(last) > 3 and not last.endswith("'s"):
        if last.endswith("ies") and len(last) > 4:
            tokens[-1] = last[:-3] + "y"
        else:
            tokens[-1] = last[:-1]

    return tokens

def edit_distance_1(left: str, right: str) -> bool:
    """ 
    Function used that returns True if two tokens differ
    by an edit distance of 1, meaning they are identical
    except for one of the following operations:
    """
    if left == right: # Identical strings
        return False
    if abs(len(left) - len(right)) > 1: # One string longer than other
        return False

    # Are they the same length with exactly one character different?
    if len(left) == len(right):
        mismatches = sum(1 for a, b in zip(left, right) if a != b)
        return mismatches == 1

    # Are they different lengths with one extra character in the longer string?
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    i = 0
    j = 0
    skipped = False
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
            continue
        if skipped:
            return False
        skipped = True
        j += 1
    return True


def is_duplicate(a: str, b: str) -> bool:
    """
    Return True ONLY if concepts are effectively identical.

    Rules:
    - Tokenize using tokenize_for_matching
    - Same number of tokens
    - All tokens match exactly OR
      exactly one token differs by edit distance 1
    """
    left_tokens = tokenize_for_matching(a)
    right_tokens = tokenize_for_matching(b)

    if len(left_tokens) != len(right_tokens):
        return False

    mismatches = [
        (left, right)
        for left, right in zip(left_tokens, right_tokens)
        if left != right
    ]

    if not mismatches:
        return True
    if len(mismatches) != 1:
        return False

    left_token, right_token = mismatches[0]
    return edit_distance_1(left_token, right_token)


def is_sibling(a: str, b: str) -> bool:
    """
    True if concepts share the same head word (last token)
    but have different modifiers.

    Example:
    - "ECB Mode" vs "CBC Mode" -> True
    """
    left_tokens = tokenize_for_matching(a)
    right_tokens = tokenize_for_matching(b)

    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False
    if left_tokens[-1] != right_tokens[-1]:
        return False

    return left_tokens[:-1] != right_tokens[:-1]


def has_strong_overlap(a: str, b: str) -> bool:
    """
    True if concepts share at least 2 tokens in order.

    This replaces naive substring matching.

    Example:
    - "Public Key Cryptography" vs "Key Cryptography" -> True
    - "Key Length" vs "Key Generation" -> False
    """
    left_tokens = tokenize_for_matching(a)
    right_tokens = tokenize_for_matching(b)

    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False

    best_run = 0
    for i in range(len(left_tokens)):
        for j in range(len(right_tokens)):
            run = 0
            while (
                i + run < len(left_tokens)
                and j + run < len(right_tokens)
                and left_tokens[i + run] == right_tokens[j + run]
            ):
                run += 1
            if run > best_run:
                best_run = run
            if best_run >= 2:
                return True

    return False


def has_antonym_conflict(a: str, b: str) -> bool:
    """
    True if concept pair contains known antonym token patterns.

    Example:
    - "First In First Out" vs "Last In First Out" -> True
    - "Periodic System" vs "Perpetual System" -> True
    """
    left_tokens = set(tokenize_for_matching(a))
    right_tokens = set(tokenize_for_matching(b))

    for token_a, token_b in ANTONYM_TOKEN_PAIRS:
        if (token_a in left_tokens and token_b in right_tokens) or (
            token_b in left_tokens and token_a in right_tokens
        ):
            return True

    return False