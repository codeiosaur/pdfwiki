"""
Text processing utilities for page generation.

Normalization, deduplication, summary building, and intro formatting.
"""

import re
from typing import Optional

from extract.fact_extractor import Fact
from generate.classify import _normalize_text_for_compare


def unique_fact_contents(facts: list[Fact]) -> list[str]:
    """Extract unique fact content strings, preserving order."""
    seen: set[str] = set()
    items: list[str] = []
    for fact in facts:
        text = fact.content.strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(text)
    return items


def build_summary(fact_contents: list[str], max_items: int = 2, max_chars: int = 320) -> str:
    if not fact_contents:
        return "No summary available."
    summary = " ".join(fact_contents[:max_items]).strip()
    if len(summary) <= max_chars:
        return summary
    return summary[: max_chars - 3].rstrip() + "..."


def normalize_text_for_compare(text: str) -> str:
    """Public alias for the comparison normalizer."""
    return _normalize_text_for_compare(text)


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = _normalize_text_for_compare(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def trim_section(items: list[str], limit: int) -> list[str]:
    cleaned = dedupe_preserve_order(items)
    return cleaned[:limit] if limit > 0 else cleaned


def emphasize_concept_once(text: str, display_title: str) -> str:
    pattern = re.compile(re.escape(display_title), re.IGNORECASE)
    if pattern.search(text):
        return pattern.sub(f"**{display_title}**", text, count=1)
    return f"**{display_title}**: {text}"


def build_lead(definition: str, key_points: list[str]) -> str:
    if definition and definition != "No definition available.":
        parts = re.split(r"(?<=[.!?])\s+", definition.strip())
        return parts[0].strip() if parts and parts[0].strip() else definition
    if key_points:
        return key_points[0]
    return "No lead available."


def build_enhanced_intro(
    display_title: str,
    definition: str,
    interpretations: list[str],
    key_points: list[str],
) -> str:
    primary = definition if definition and definition != "No definition available." else ""
    if not primary and key_points:
        primary = key_points[0]
    if not primary and interpretations:
        primary = interpretations[0]
    if not primary:
        return f"**{display_title}** is a concept in this source material."

    primary = emphasize_concept_once(primary.strip(), display_title)

    secondary_candidates = interpretations + key_points
    secondary = ""
    norm_primary = _normalize_text_for_compare(primary)
    for item in secondary_candidates:
        if _normalize_text_for_compare(item) != norm_primary:
            secondary = item.strip()
            break

    if not secondary:
        combined = primary
    else:
        secondary = secondary[0].upper() + secondary[1:] if secondary else secondary
        combined = f"{primary} {secondary}"

    sentences = re.split(r"(?<=[.!?])\s+", combined.strip())
    return " ".join(sentences[:2])