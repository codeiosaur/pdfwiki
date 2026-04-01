"""
Page renderers.

Three rendering modes:
  generate_pages          — standard (Lead/Definition/Key Points with footnotes)
  generate_pages_enhanced — semantic sections with footnotes
  generate_pages_wiki     — wiki-style with templates, wikilinks, clean sources
"""

import re

from extract.fact_extractor import Fact
from transform.matching import has_antonym_conflict, is_sibling

from generate.classify import (
    classify_fact,
    classify_semantic_fact,
    select_fallback_definition,
    _has_template_markers,
    _is_low_signal_key_point,
    _normalize_text_for_compare,
)
from generate.util import (
    unique_fact_contents,
    trim_section,
    build_lead,
    build_enhanced_intro,
    emphasize_concept_once,
)
from generate.titles import (
    build_acronym_map,
    normalize_page_title,
    ACRONYM_CANONICAL,
)
from generate.related import (
    build_related_concepts,
    build_related_concepts_by_chunks,
    fact_sources,
    citation_suffixes,
    looks_like_uuid,
    all_sources_are_uuids,
)
from generate.wiki_helpers import (
    classify_concept_type,
    inject_wikilinks,
    promote_all_facts_to_content,
)

import generate.titles as _titles_module


def _init_acronyms(concept_names: list[str]) -> None:
    """Populate the global acronym map from concept names."""
    _titles_module.ACRONYM_CANONICAL = build_acronym_map(concept_names)


# ===================================================================
# Standard renderer
# ===================================================================

def generate_pages(grouped: dict[str, list[Fact]], include_empty_pages: bool = False) -> dict[str, str]:
    """
    Generate concept pages from grouped facts using deterministic fact types.

    Each page contains:
    - Definition (best definition fact)
    - Key Points (only key_point facts)
    - Excludes examples and instructions
    """
    pages: dict[str, str] = {}
    concept_names = list(grouped.keys())
    _init_acronyms(concept_names)
    related_map = build_related_concepts(concept_names)

    for concept, facts in grouped.items():
        display_title = normalize_page_title(concept)
        fact_contents = unique_fact_contents(facts)
        text_to_sources = fact_sources(facts)

        definitions: list[str] = []
        key_points: list[str] = []

        for item in fact_contents:
            label = classify_fact(item)
            if label == "definition":
                definitions.append(item)
            elif label == "key_point":
                key_points.append(item)

        key_points = [
            item
            for item in key_points
            if not _is_low_signal_key_point(item)
            and not has_antonym_conflict(concept, item)
        ]

        selected_definition = select_fallback_definition(
            concept=concept,
            definitions=definitions,
            key_points=key_points,
            fact_contents=fact_contents,
        )
        definition = selected_definition if selected_definition is not None else "No definition available."
        definition_norm = _normalize_text_for_compare(definition)

        key_points = [
            item for item in key_points
            if _normalize_text_for_compare(item) != definition_norm
        ]

        lead = build_lead(definition, key_points)

        hide_sources = all_sources_are_uuids(text_to_sources)

        if hide_sources:
            definition_rendered = [definition]
            key_point_rendered = key_points
        else:
            citation_index = 1
            citation_map: dict[tuple[str, ...], int] = {}
            definition_rendered, definition_notes, citation_index = citation_suffixes(
                [definition], text_to_sources, citation_map, citation_index,
            )
            key_point_rendered, key_point_notes, citation_index = citation_suffixes(
                key_points, text_to_sources, citation_map, citation_index,
            )
            combined_notes = definition_notes + key_point_notes

        if definition == "No definition available." and not key_points and not include_empty_pages:
            continue

        lines: list[str] = [
            f"# {display_title}", "",
            "## Lead", lead, "",
            "## Definition",
            definition_rendered[0] if definition_rendered else definition, "",
            "## Key Points",
        ]

        if key_point_rendered:
            for item in key_point_rendered:
                lines.append(f"- {item}")
        else:
            lines.append("- No key points available.")

        lines.extend(["", "## Related Concepts"])
        related = related_map.get(concept, [])
        if related:
            for item in related:
                lines.append(f"- [[{normalize_page_title(item)}]]")
        else:
            lines.append("- None")

        if not hide_sources:
            lines.extend(["", "## Sources"])
            if combined_notes:
                lines.extend(combined_notes)
            else:
                lines.append("- None")

        pages[display_title] = "\n".join(lines)

    return pages


# ===================================================================
# Enhanced renderer (semantic sections with footnotes)
# ===================================================================

def generate_pages_enhanced(grouped: dict[str, list[Fact]], include_empty_pages: bool = False) -> dict[str, str]:
    """
    Generate richer wiki-style pages with semantic sections.

    This renderer keeps deterministic logic and source citations while structuring
    content into narrative sections similar to curated study notes.
    """
    pages: dict[str, str] = {}
    concept_names = list(grouped.keys())
    _init_acronyms(concept_names)
    related_map = build_related_concepts(concept_names)

    for concept, facts in grouped.items():
        display_title = normalize_page_title(concept)
        fact_contents = unique_fact_contents(facts)
        text_to_sources = fact_sources(facts)

        definitions: list[str] = []
        formulas: list[str] = []
        interpretations: list[str] = []
        examples: list[str] = []
        cautions: list[str] = []
        key_points: list[str] = []

        for item in fact_contents:
            if _has_template_markers(item):
                continue
            semantic = classify_semantic_fact(item)
            if semantic == "instruction":
                continue
            elif semantic == "definition":
                definitions.append(item)
            elif semantic == "formula":
                formulas.append(item)
            elif semantic == "interpretation":
                interpretations.append(item)
            elif semantic == "example":
                examples.append(item)
            elif semantic == "caution":
                cautions.append(item)
            else:
                key_points.append(item)

        key_points = [item for item in key_points if not _is_low_signal_key_point(item)]
        interpretations = [item for item in interpretations if not _is_low_signal_key_point(item)]

        selected_definition = select_fallback_definition(
            concept=concept, definitions=definitions,
            key_points=key_points, fact_contents=fact_contents,
        )
        definition = selected_definition if selected_definition is not None else "No definition available."

        definition_norm = _normalize_text_for_compare(definition)
        key_points = [item for item in key_points if _normalize_text_for_compare(item) != definition_norm]
        interpretations = [item for item in interpretations if _normalize_text_for_compare(item) != definition_norm]

        formulas = trim_section(formulas, 3)
        interpretations = trim_section(interpretations, 4)
        examples = trim_section(examples, 2)
        cautions = trim_section(cautions, 3)
        key_points = trim_section(key_points, 4)

        if (
            definition == "No definition available."
            and not formulas and not interpretations
            and not examples and not cautions and not key_points
            and not include_empty_pages
        ):
            continue

        intro = build_enhanced_intro(display_title, definition, interpretations, key_points)

        hide_sources = all_sources_are_uuids(text_to_sources)

        if hide_sources:
            definition_rendered = [definition]
            formula_rendered = formulas
            interpretation_rendered = interpretations
            example_rendered = examples
            caution_rendered = cautions
            key_point_rendered = key_points
            combined_notes: list[str] = []
        else:
            citation_index = 1
            citation_map: dict[tuple[str, ...], int] = {}

            definition_rendered, definition_notes, citation_index = citation_suffixes(
                [definition], text_to_sources, citation_map, citation_index)
            formula_rendered, formula_notes, citation_index = citation_suffixes(
                formulas, text_to_sources, citation_map, citation_index)
            interpretation_rendered, interpretation_notes, citation_index = citation_suffixes(
                interpretations, text_to_sources, citation_map, citation_index)
            example_rendered, example_notes, citation_index = citation_suffixes(
                examples, text_to_sources, citation_map, citation_index)
            caution_rendered, caution_notes, citation_index = citation_suffixes(
                cautions, text_to_sources, citation_map, citation_index)
            key_point_rendered, key_point_notes, citation_index = citation_suffixes(
                key_points, text_to_sources, citation_map, citation_index)

            combined_notes = (
                definition_notes + formula_notes + interpretation_notes
                + example_notes + caution_notes + key_point_notes
            )

        lines: list[str] = [
            f"# {display_title}", "", intro, "", "---", "",
            "## Definition",
            definition_rendered[0] if definition_rendered else definition,
        ]

        if formula_rendered:
            lines.extend(["", "---", "", "## Formula", ""])
            for item in formula_rendered:
                lines.extend(["```", item, "```", ""])

        if interpretation_rendered:
            lines.extend(["---", "", "## Interpretation", ""])
            for item in interpretation_rendered:
                lines.append(f"- {item}")
            lines.append("")

        if example_rendered:
            lines.extend(["---", "", "## Example", ""])
            for item in example_rendered:
                lines.append(f"- {item}")
            lines.append("")

        if caution_rendered:
            lines.extend(["---", "", "## Cautions", ""])
            for item in caution_rendered:
                lines.append(f"- {item}")
            lines.append("")

        if key_point_rendered:
            lines.extend(["---", "", "## Key Points", ""])
            for item in key_point_rendered:
                lines.append(f"- {item}")
            lines.append("")

        lines.extend(["---", "", "## Related Concepts"])
        related = related_map.get(concept, [])
        if related:
            for item in related:
                lines.append(f"- [[{normalize_page_title(item)}]]")
        else:
            lines.append("- None")

        if not hide_sources:
            lines.extend(["", "---", "", "## Sources"])
            if combined_notes:
                lines.extend(combined_notes)
            else:
                lines.append("- None")

        pages[display_title] = "\n".join(lines)

    return pages


# ===================================================================
# Wiki renderer (templates, wikilinks, clean sources)
# ===================================================================

def generate_pages_wiki(
    grouped: dict[str, list[Fact]], include_empty_pages: bool = False
) -> dict[str, str]:
    """
    Wiki-style page renderer with:
    - Concept-type templates (ratio, method, system, general)
    - Wikilinks in body text
    - Chunk-co-occurrence related concepts
    - Intro paragraph instead of duplicate lead/definition
    - All non-duplicate facts promoted to content
    - Clean source attribution (no chunk UUIDs)
    """
    pages: dict[str, str] = {}
    concept_names = list(grouped.keys())
    _init_acronyms(concept_names)

    all_display_titles = {normalize_page_title(c) for c in concept_names}

    # Build alias map: acronym → display title of the concept that contains it.
    # Allows inject_wikilinks() to link e.g. "FIFO" → [[First In First Out (FIFO)]].
    _raw_acronyms = build_acronym_map(concept_names)
    alias_map: dict[str, str] = {}
    for _c in concept_names:
        _d = normalize_page_title(_c)
        for _lower, _upper in _raw_acronyms.items():
            if _upper in _d:
                alias_map.setdefault(_lower, _d)

    concept_chunks: dict[str, set[str]] = {}
    for concept, facts in grouped.items():
        concept_chunks[concept] = {
            f.source_chunk_id for f in facts if f.source_chunk_id
        }

    for concept, facts in grouped.items():
        display_title = normalize_page_title(concept)
        fact_contents = unique_fact_contents(facts)
        concept_type = classify_concept_type(concept, fact_contents)

        # --- Select definition ---
        definitions: list[str] = []
        key_points_raw: list[str] = []

        for item in fact_contents:
            if _has_template_markers(item):
                continue
            label = classify_fact(item)
            if label == "definition":
                definitions.append(item)
            elif label == "key_point":
                key_points_raw.append(item)

        selected_definition = select_fallback_definition(
            concept=concept, definitions=definitions,
            key_points=key_points_raw, fact_contents=fact_contents,
        )
        definition = selected_definition if selected_definition else "No definition available."

        # --- Gather ALL usable content ---
        all_content = promote_all_facts_to_content(fact_contents, definition)

        formulas: list[str] = []
        interpretations: list[str] = []
        cautions: list[str] = []
        details: list[str] = []

        for item in all_content:
            sem = classify_semantic_fact(item)
            if sem == "formula":
                formulas.append(item)
            elif sem == "interpretation":
                interpretations.append(item)
            elif sem == "caution":
                cautions.append(item)
            else:
                details.append(item)

        formulas = trim_section(formulas, 3)
        interpretations = trim_section(interpretations, 4)
        cautions = trim_section(cautions, 3)
        details = trim_section(details, 6)

        has_content = (
            definition != "No definition available."
            or formulas or interpretations or cautions or details
        )
        if not has_content and not include_empty_pages:
            continue

        # --- Build intro paragraph ---
        intro = emphasize_concept_once(definition, display_title)
        if interpretations:
            extra = interpretations[0].strip()
            extra = extra[0].upper() + extra[1:] if extra else extra
            intro = f"{intro} {extra}"

        # Strip "**Title**: Title ..." redundancy when the definition already
        # re-introduces the concept (e.g. normalized title has "(DSI)" but
        # the definition says "DSI is ..." without parentheses).
        bolded_prefix = f"**{display_title}**: "
        if intro.startswith(bolded_prefix):
            remainder = intro[len(bolded_prefix):]
            _stopwords = {"a", "an", "the", "of", "in", "on", "by", "to", "for", "and", "or"}
            title_words = display_title.lower().split()
            title_first = next((w for w in title_words if w not in _stopwords), "")
            remainder_words = re.findall(r"[a-z0-9]+", remainder.lower())
            if title_first and remainder_words and remainder_words[0] == title_first:
                intro = remainder

        intro = inject_wikilinks(intro, all_display_titles, display_title, alias_map=alias_map)

        # --- Assemble page by concept type (no footnote suffixes) ---
        lines: list[str] = [f"# {display_title}", "", intro, "", "---"]

        if concept_type == "ratio":
            lines.extend(["", "## Definition", ""])
            lines.append(definition)
            if formulas:
                lines.extend(["", "---", "", "## Formula", ""])
                for item in formulas:
                    lines.append(f"```\n{item}\n```")
                    lines.append("")
            if interpretations:
                lines.extend(["", "---", "", "## What It Tells You", ""])
                for item in interpretations:
                    lines.append(f"- {inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)}")
                lines.append("")
            if details:
                lines.extend(["", "---", "", "## Key Points", ""])
                for item in details:
                    lines.append(f"- {inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)}")
                lines.append("")

        elif concept_type in ("method", "system"):
            lines.extend(["", "## Definition", ""])
            lines.append(definition)
            if details:
                lines.extend(["", "---", "", "## How It Works", ""])
                for item in details:
                    lines.append(f"- {inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)}")
                lines.append("")
            if formulas:
                lines.extend(["", "---", "", "## Formula", ""])
                for item in formulas:
                    lines.append(f"```\n{item}\n```")
                    lines.append("")
            if interpretations:
                lines.extend(["", "---", "", "## Key Points", ""])
                for item in interpretations:
                    lines.append(f"- {inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)}")
                lines.append("")

        else:
            lines.extend(["", "## Definition", ""])
            lines.append(definition)
            combined_detail = details + interpretations
            if combined_detail:
                lines.extend(["", "---", "", "## Key Points", ""])
                for item in combined_detail:
                    lines.append(f"- {inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)}")
                lines.append("")

        if cautions:
            lines.extend(["", "---", "", "## Cautions", ""])
            for item in cautions:
                lines.append(f"- {inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)}")
            lines.append("")

        # Related Concepts (chunk co-occurrence with IDF penalty)
        lines.extend(["", "---", "", "## Related Concepts"])
        related = build_related_concepts_by_chunks(
            concept, concept_chunks, concept_names, grouped=grouped
        )
        if related:
            for item in related:
                lines.append(f"- [[{normalize_page_title(item)}]]")
        else:
            lines.append("- None")

        # See Also — explicitly contrasting concepts (antonyms / siblings)
        see_also = sorted(
            normalize_page_title(other)
            for other in concept_names
            if other != concept
            and (has_antonym_conflict(concept, other) or is_sibling(concept, other))
        )
        if see_also:
            lines.extend(["", "---", "", "## See Also"])
            for item in see_also:
                lines.append(f"- [[{item}]]")

        # Source attribution — show filenames when available, hide UUIDs
        unique_sources = sorted({
            fact.source_chunk_id.split("::")[0]
            for fact in facts if fact.source_chunk_id
        })
        has_real_sources = any(not looks_like_uuid(s) for s in unique_sources)
        if has_real_sources:
            lines.extend(["", "---", "", "## Sources"])
            for src in unique_sources:
                if not looks_like_uuid(src):
                    lines.append(f"- {src}")

        pages[display_title] = "\n".join(lines)

    return pages


# ===================================================================
# Output formatters
# ===================================================================

def render_pages_document(pages: dict[str, str]) -> str:
    """Render all pages into one clearly delineated text document."""
    if not pages:
        return "# Concept Pages\n\nNo pages generated.\n"

    sections: list[str] = ["# Concept Pages", ""]
    for i, (concept, page_text) in enumerate(pages.items(), start=1):
        sections.append(f"--- PAGE {i}: {concept} ---")
        sections.append(page_text)
        sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def render_pages_preview(pages: dict[str, str], max_pages: int = 2) -> str:
    """Render only the first N pages for console preview."""
    if max_pages < 1:
        return ""

    selected = list(pages.items())[:max_pages]
    if not selected:
        return ""

    sections: list[str] = []
    for i, (concept, page_text) in enumerate(selected, start=1):
        sections.append(f"--- PREVIEW PAGE {i}: {concept} ---")
        sections.append(page_text)
        sections.append("")

    return "\n".join(sections).rstrip()