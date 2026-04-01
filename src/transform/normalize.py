import re
from typing import List

from extract.fact_extractor import Fact


GENERIC_SUFFIXES = {"method", "system", "approach", "model", "technique", "process"}
LEADING_FILLERS = ("number of ", "type of ", "kind of ")
CANONICAL_PHRASE_RULES = [
	(r"\bfirst\s+in\s+first\s+out\b|\bfifo\b", "First In First Out"),
	(r"\blast\s+in\s+first\s+out\b|\blifo\b", "Last In First Out"),
	(r"\blower\s+of\s+cost\s+or\s+market\b|\blcm\b", "Lower of Cost or Market"),
	(r"\bcost\s+of\s+goods\s+sold\b|\bcogs\b", "Cost of Goods Sold"),
	(r"\bdays\s+sales\s+in\s+inventory\b|\bdsi\b", "Days Sales in Inventory (DSI)"),
	(r"\belectronic\s+product\s+code\b|\bepcs?\b", "EPC"),
	(r"\bupc\s*bar\s*code\b|\bupc\s+barcode\b", "UPC Barcode"),
]


def _split_words(text: str) -> List[str]:
	return [token for token in re.split(r"\s+", text.strip()) if token]


_SINGULARIZE_EXCEPTIONS = {
    "analysis", "basis", "axis", "thesis", "crisis", "diagnosis",
    "emphasis", "hypothesis", "oasis", "parenthesis", "synopsis",
    "status", "apparatus", "campus", "virus", "census", "bonus",
    "class", "grass", "glass", "mass", "pass", "brass",
}


def _singularize_last_word(words: List[str]) -> List[str]:
	if not words:
		return words

	last = words[-1]
	lower = last.lower()
	if lower in _SINGULARIZE_EXCEPTIONS:
		return words
	if len(lower) > 3 and lower.endswith("s"):
		if lower.endswith("ies") and len(lower) > 4:
			words[-1] = last[:-3] + "y"
		elif not lower.endswith("ss"):
			words[-1] = last[:-1]
	return words


def _dedupe_repeated_words(words: List[str]) -> List[str]:
	if not words:
		return words

	deduped = [words[0]]
	for token in words[1:]:
		if token.lower() != deduped[-1].lower():
			deduped.append(token)
	return deduped


def _normalize_parentheses(concept: str) -> str:
	# Normalize patterns like "Full Form (ABC)" by keeping one consistent form.
	match = re.match(r"^\s*(.*?)\s*\(([^()]+)\)\s*$", concept)
	if not match:
		return concept

	left = re.sub(r"\s+", " ", match.group(1)).strip()
	inside = re.sub(r"\s+", " ", match.group(2)).strip()
	if not left:
		return inside
	if not inside:
		return left

	left_words = [w for w in re.findall(r"[A-Za-z0-9]+", left) if w]
	initials = "".join(word[0].upper() for word in left_words if word)
	inside_compact = re.sub(r"[^A-Za-z0-9]", "", inside).upper()

	# If parenthetical is an acronym of the left phrase, keep full phrase only.
	if inside_compact and inside_compact == initials and 2 <= len(inside_compact) <= 6:
		return left

	# If both are essentially the same token, keep a single copy.
	if left.lower() == inside.lower():
		return left

	return f"{left} {inside}"


def _title_case_preserve_acronyms(words: List[str]) -> List[str]:
	out: List[str] = []
	for word in words:
		clean = re.sub(r"[^A-Za-z0-9]", "", word)
		if clean.isupper() and 2 <= len(clean) <= 6:
			out.append(clean)
		else:
			out.append(word.lower().capitalize())
	return out


def normalize_concept_rules(concept: str) -> str:
	"""
	Deterministic, domain-agnostic concept normalization.
	"""
	if not concept:
		return concept

	text = concept.strip()
	if not text:
		return text

	# Rule 2: collapse parenthetical duplication to one consistent form.
	text = _normalize_parentheses(text)

	# Rule 3: punctuation normalization.
	text = text.replace("-", " ")
	text = re.sub(r"\s+", " ", text).strip()

	# Rule 3.5: deterministic canonical phrase mapping for common variants.
	lower_text = text.lower()
	for pattern, replacement in CANONICAL_PHRASE_RULES:
		if re.search(pattern, lower_text):
			text = replacement
			break

	# Rule 6: remove leading filler phrases.
	lowered = text.lower()
	for filler in LEADING_FILLERS:
		if lowered.startswith(filler):
			text = text[len(filler):].strip()
			break

	words = _split_words(text)
	if not words:
		return ""

	# Rule 7: dedupe repeated words.
	words = _dedupe_repeated_words(words)

	# Rule 4: remove generic trailing suffix when remaining phrase is meaningful.
	if words and words[-1].lower() in GENERIC_SUFFIXES:
		stem = words[:-1]
		if len(stem) >= 2 or (len(stem) == 1 and len(stem[0]) > 3):
			words = stem

	# Rule 5: singularize last word.
	words = _singularize_last_word(words)

	# Rule 1: title case with acronym preservation.
	words = _title_case_preserve_acronyms(words)

	return " ".join(words)


def normalize_group_keys(grouped: dict[str, List[Fact]]) -> dict[str, List[Fact]]:
	"""
	Normalize grouped concept keys and merge groups with equal normalized keys.
	"""
	normalized_grouped: dict[str, List[Fact]] = {}
	for concept, facts in grouped.items():
		normalized = normalize_concept_rules(concept)
		target = normalized if normalized else concept
		normalized_grouped.setdefault(target, []).extend(facts)
	return normalized_grouped
