import re
from typing import List

from extract.fact_extractor import Fact


_LOWERCASE_CONNECTORS = {"a", "an", "and", "for", "in", "of", "or", "the", "to"}


def _format_word(word: str, first: bool) -> str:
	"""Format ordinary words without changing acronyms or mixed-case names."""
	if not first and word.lower() in _LOWERCASE_CONNECTORS:
		return word.lower()
	if "-" in word or "/" in word:
		# capitalize() would turn "First-In" into "First-in".
		return word[:1].upper() + word[1:] if word.islower() else word
	if word.islower() or word.istitle():
		return word.capitalize()
	return word


def normalize_concept_rules(concept: str) -> str:
	"""
	Normalize presentation only; do not infer that names mean the same thing.
	"""
	if not concept:
		return concept

	text = concept.strip()
	if not text:
		return text

	text = re.sub(r"\s+", " ", text)
	text = re.sub(r"\s+([,.;:])", r"\1", text)
	return " ".join(_format_word(word, i == 0) for i, word in enumerate(text.split()))


def normalize_group_keys(grouped: dict[str, List[Fact]]) -> dict[str, List[Fact]]:
	"""
	Normalize grouped concept keys and merge groups with equal normalized keys.
	"""
	normalized_grouped: dict[str, List[Fact]] = {}
	merge_key_to_title: dict[str, str] = {}

	for concept, facts in grouped.items():
		normalized = normalize_concept_rules(concept)
		target = normalized if normalized else concept
		key = target.casefold()
		if key:
			existing = merge_key_to_title.get(key)
			if existing is None:
				merge_key_to_title[key] = target
			else:
				# The first spelling is stable; later case variants share its group.
				target = existing
		normalized_grouped.setdefault(target, []).extend(facts)
	return normalized_grouped
