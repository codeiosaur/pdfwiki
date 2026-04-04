"""
Page renderers.

Three rendering modes:
  generate_pages          — standard (Lead/Definition/Key Points with footnotes)
  generate_pages_enhanced — semantic sections with footnotes
  generate_pages_wiki     — wiki-style with templates, wikilinks, clean sources
"""

import re
from concurrent.futures import ThreadPoolExecutor

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
from generate.page_layout import build_wiki_page_lines, is_question_prompt

import generate.titles as _titles_module


def _init_acronyms(concept_names: list[str]) -> None:
    """Populate the global acronym map from concept names."""
    _titles_module.ACRONYM_CANONICAL = build_acronym_map(concept_names)


def _build_link_maps(concept_names: list[str]) -> tuple[set[str], dict[str, str]]:
    """Build display-title and acronym alias maps for wikilink injection."""
    all_display_titles = {normalize_page_title(c) for c in concept_names}
    raw_acronyms = build_acronym_map(concept_names)
    alias_map: dict[str, str] = {}

    for concept in concept_names:
        display_title = normalize_page_title(concept)
        for alias_lower, acronym in raw_acronyms.items():
            if acronym in display_title:
                alias_map.setdefault(alias_lower, display_title)

    return all_display_titles, alias_map


def _build_frontmatter_from_sources(source_names: list[str]) -> list[str]:
    """Build a small YAML frontmatter block from source filenames.

    Expects a list of source name strings (filenames, without chunk suffix).
    Returns a list of lines to prepend to the page content.
    """
    if not source_names:
        return []
    tags = []
    for s in source_names:
        if not s:
            continue
        base = s.split("::")[0]
        # strip common extensions
        if "." in base:
            base = base.rsplit(".", 1)[0]
        tags.append(base)
    tags = sorted(set(tags))
    lines: list[str] = ["---", "tags:"]
    for t in tags:
        lines.append(f"  - {t}")
    # derive a simple folder hint from the first tag (useful in Obsidian)
    if tags:
        folder = tags[0].split("-")[0]
        if folder:
            lines.append(f"folder: {folder}")
    lines.append("---")
    return lines


# ===================================================================
# Standard renderer
# ===================================================================

def generate_pages(
    grouped: dict[str, list[Fact]],
    include_empty_pages: bool = False,
    workers: int = 1,
) -> dict[str, str]:
    """
    Generate concept pages from grouped facts using deterministic fact types.

    Each page contains:
    - Definition (best definition fact)
    - Key Points (only key_point facts)
    - Excludes examples and instructions
    """
    concept_names = sorted(grouped.keys())
    _init_acronyms(concept_names)
    all_display_titles, alias_map = _build_link_maps(concept_names)
    related_map = build_related_concepts(concept_names, max_related=4)

    def _render_one(concept: str) -> tuple[str, str] | None:
        facts = grouped[concept]
        display_title = normalize_page_title(concept)
        fact_contents = [item for item in unique_fact_contents(facts) if not is_question_prompt(item)]
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
        lead = inject_wikilinks(lead, all_display_titles, display_title, alias_map=alias_map)

        hide_sources = all_sources_are_uuids(text_to_sources)

        if hide_sources:
            definition_rendered = [
                inject_wikilinks(definition, all_display_titles, display_title, alias_map=alias_map)
            ]
            key_point_rendered = [
                inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)
                for item in key_points
            ]
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
            definition_rendered = [
                inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)
                for item in definition_rendered
            ]
            key_point_rendered = [
                inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)
                for item in key_point_rendered
            ]

        if definition == "No definition available." and not key_points and not include_empty_pages:
            return None

        unique_sources = sorted({
            src.split("::")[0]
            for srcs in text_to_sources.values()
            for src in srcs
            if src
        })
        real_sources = [s for s in unique_sources if not looks_like_uuid(s)]
        fm_lines = _build_frontmatter_from_sources(real_sources) if real_sources else []

        lines: list[str] = fm_lines + [
            f"# {display_title}", "",
            "## Lead", lead, "",
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

        return display_title, "\n".join(lines)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            raw: list = list(executor.map(_render_one, concept_names))
    else:
        raw = [_render_one(c) for c in concept_names]
    pages: dict[str, str] = {}
    for result in raw:
        if result is not None:
            pages[result[0]] = result[1]
    return pages


# ===================================================================
# Enhanced renderer (semantic sections with footnotes)
# ===================================================================

def generate_pages_enhanced(
    grouped: dict[str, list[Fact]],
    include_empty_pages: bool = False,
    workers: int = 1,
) -> dict[str, str]:
    """
    Generate richer wiki-style pages with semantic sections.

    This renderer keeps deterministic logic and source citations while structuring
    content into narrative sections similar to curated study notes.
    """
    concept_names = sorted(grouped.keys())
    _init_acronyms(concept_names)
    all_display_titles, alias_map = _build_link_maps(concept_names)
    related_map = build_related_concepts(concept_names, max_related=4)

    def _render_one(concept: str) -> tuple[str, str] | None:
        facts = grouped[concept]
        display_title = normalize_page_title(concept)
        fact_contents = [item for item in unique_fact_contents(facts) if not is_question_prompt(item)]
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
            return None

        intro = build_enhanced_intro(display_title, definition, interpretations, key_points)
        intro = inject_wikilinks(intro, all_display_titles, display_title, alias_map=alias_map)

        hide_sources = all_sources_are_uuids(text_to_sources)

        if hide_sources:
            definition_rendered = [
                inject_wikilinks(definition, all_display_titles, display_title, alias_map=alias_map)
            ]
            formula_rendered = [
                inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)
                for item in formulas
            ]
            interpretation_rendered = [
                inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)
                for item in interpretations
            ]
            example_rendered = [
                inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)
                for item in examples
            ]
            caution_rendered = [
                inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)
                for item in cautions
            ]
            key_point_rendered = [
                inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)
                for item in key_points
            ]
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

            definition_rendered = [
                inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)
                for item in definition_rendered
            ]
            formula_rendered = [
                inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)
                for item in formula_rendered
            ]
            interpretation_rendered = [
                inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)
                for item in interpretation_rendered
            ]
            example_rendered = [
                inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)
                for item in example_rendered
            ]
            caution_rendered = [
                inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)
                for item in caution_rendered
            ]
            key_point_rendered = [
                inject_wikilinks(item, all_display_titles, display_title, alias_map=alias_map)
                for item in key_point_rendered
            ]

        # Build frontmatter from any real (non-UUID) source filenames we have.
        unique_sources = sorted({
            src.split("::")[0]
            for srcs in text_to_sources.values()
            for src in srcs
            if src
        })
        real_sources = [s for s in unique_sources if not looks_like_uuid(s)]
        fm_lines = _build_frontmatter_from_sources(real_sources) if real_sources else []

        lines: list[str] = fm_lines + [
            f"# {display_title}", "", intro, "", "---", "",
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

        return display_title, "\n".join(lines)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            raw: list = list(executor.map(_render_one, concept_names))
    else:
        raw = [_render_one(c) for c in concept_names]
    pages: dict[str, str] = {}
    for result in raw:
        if result is not None:
            pages[result[0]] = result[1]
    return pages


# ===================================================================
# Wiki renderer (templates, wikilinks, clean sources)
# ===================================================================

def generate_pages_wiki(
    grouped: dict[str, list[Fact]],
    include_empty_pages: bool = False,
    workers: int = 1,
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
    concept_names = sorted(grouped.keys())
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

    def _render_one(concept: str) -> tuple[str, str] | None:
        facts = grouped[concept]
        display_title = normalize_page_title(concept)
        fact_contents = [item for item in unique_fact_contents(facts) if not is_question_prompt(item)]
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
        all_content = promote_all_facts_to_content(fact_contents, definition, include_examples=True)

        formulas: list[str] = []
        interpretations: list[str] = []
        examples: list[str] = []
        cautions: list[str] = []
        details: list[str] = []

        for item in all_content:
            sem = classify_semantic_fact(item)
            if sem == "formula":
                formulas.append(item)
            elif sem == "interpretation":
                interpretations.append(item)
            elif sem == "example":
                examples.append(item)
            elif sem == "caution":
                cautions.append(item)
            else:
                details.append(item)

        formulas = trim_section(formulas, 3)
        interpretations = trim_section(interpretations, 4)
        examples = trim_section(examples, 3)
        cautions = trim_section(cautions, 3)
        details = trim_section(details, 6)

        has_content = (
            definition != "No definition available."
            or formulas or interpretations or cautions or details
        )
        if not has_content and not include_empty_pages:
            return None

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

        # Related Concepts (chunk co-occurrence with IDF penalty)
        related = build_related_concepts_by_chunks(
            concept, concept_chunks, concept_names, max_related=5, grouped=grouped
        )
        related = [normalize_page_title(item) for item in related]

        # See Also — explicitly contrasting concepts (antonyms / siblings)
        see_also = sorted(
            normalize_page_title(other)
            for other in concept_names
            if other != concept
            and (has_antonym_conflict(concept, other) or is_sibling(concept, other))
        )
        if see_also:
            see_also = [normalize_page_title(item) for item in see_also]

        # Source attribution — show filenames when available, hide UUIDs
        unique_sources = sorted({
            fact.source_chunk_id.split("::")[0]
            for fact in facts if fact.source_chunk_id
        })
        has_real_sources = any(not looks_like_uuid(s) for s in unique_sources)
        if has_real_sources:
            sources = [src for src in unique_sources if not looks_like_uuid(src)]
        else:
            sources = []

        lines = build_wiki_page_lines(
            display_title=display_title,
            intro=intro,
            definition=definition,
            concept_type=concept_type,
            formulas=formulas,
            interpretations=interpretations,
            examples=examples,
            cautions=cautions,
            details=details,
            related=related,
            see_also=see_also,
            sources=sources,
            include_examples=True,
        )

        # Prepend YAML frontmatter when we have human-friendly source names.
        fm_lines = _build_frontmatter_from_sources(sources) if sources else []
        return display_title, "\n".join(fm_lines + lines)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            raw: list = list(executor.map(_render_one, concept_names))
    else:
        raw = [_render_one(c) for c in concept_names]
    pages: dict[str, str] = {}
    for result in raw:
        if result is not None:
            pages[result[0]] = result[1]
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