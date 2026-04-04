"""
Pass 3: LLM-based synthesis renderer.

After concept grouping and transformation, this optional pass asks the LLM
to write a rich, prose-style wiki page for each concept using its grouped
facts as the sole source of truth.  Falls back to the wiki-renderer output
for any concept where synthesis fails.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, List, Optional

from extract.fact_extractor import Fact
from generate.related import build_related_concepts_by_chunks
from generate.titles import normalize_page_title
from generate.util import unique_fact_contents
from generate.wiki_helpers import inject_wikilinks

if TYPE_CHECKING:
    from backend.base import LLMBackend


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_synthesis_prompt(
    display_title: str,
    fact_contents: List[str],
    related_titles: List[str],
) -> str:
    numbered = "\n".join(f"  {i + 1}. {f}" for i, f in enumerate(fact_contents))
    link_hints = (
        ", ".join(f"[[{t}]]" for t in related_titles)
        if related_titles
        else "none"
    )
    return f"""You are writing an Obsidian wiki study-note page for the concept "{display_title}".

Source facts — use ONLY these; do not invent additional information:
{numbered}

Other concepts you may link with [[wikilinks]] (use the exact titles shown): {link_hints}

Write the page body in Markdown:
- Start with # {display_title} as the page title.
- Follow with a short intro paragraph (1–3 sentences) defining the concept. Bold the concept name on first use (**{display_title}**).
- Use ## section headers that fit the content. Choose from: Formula, How It Works, Interpretation, Comparison, Worked Example, Key Takeaways, Cautions. Only include sections that have content from the source facts.
- If a formula is present, render it in a fenced code block (```).
- If multiple variants or methods exist, create a Markdown comparison table.
- Under Worked Example, include any concrete numbers from the source facts.
- Link related concepts using [[wikilinks]] on first mention in the body text.
- Do NOT output a ## Related Concepts, ## See Also, ## Sources section, or YAML frontmatter — those are appended automatically.
- Do NOT invent facts, examples, or numbers not in the source list above.

Output ONLY the Markdown page body, starting with # {display_title}."""


# ---------------------------------------------------------------------------
# Page section helpers
# ---------------------------------------------------------------------------

def _extract_frontmatter(wiki_page: str) -> str:
    """Return the YAML frontmatter block (including delimiters) or empty string."""
    if not wiki_page.startswith("---"):
        return ""
    end = wiki_page.find("\n---", 3)
    if end == -1:
        return ""
    return wiki_page[: end + 4]


def _extract_tail_sections(wiki_page: str) -> str:
    """Return everything from ## Related Concepts onward in the wiki page."""
    for marker in ["\n## Related Concepts", "\n\n---\n\n## Related Concepts"]:
        idx = wiki_page.find(marker)
        if idx != -1:
            return wiki_page[idx:].strip()
    return ""


def _strip_auto_sections(body: str) -> str:
    """Remove any Related Concepts / See Also / Sources section the LLM added."""
    for header in ["## Related Concepts", "## See Also", "## Sources"]:
        idx = body.find(f"\n{header}")
        if idx != -1:
            body = body[:idx]
    return body.rstrip()


def _ensure_title_heading(body: str, display_title: str) -> str:
    """Prepend # {display_title} if the LLM omitted the heading."""
    stripped = body.lstrip()
    if not stripped.startswith("#"):
        return f"# {display_title}\n\n{body}"
    return body


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------

def synthesize_pages(
    grouped: dict[str, list[Fact]],
    backend: "LLMBackend",
    wiki_pages: dict[str, str],
    workers: int = 1,
) -> dict[str, str]:
    """
    Generate a wiki page per concept via LLM synthesis.

    Each concept's facts are passed to the LLM with a prompt that requests
    a structured Markdown page.  The frontmatter and Related Concepts section
    are taken from the existing ``wiki_pages`` output so that wikilinks and
    source attribution remain consistent.

    Falls back to the corresponding wiki page on failure (empty response,
    exception, or all-models-failed).

    Args:
        grouped:    Final concept groups (concept → [Fact]).
        backend:    LLM backend used for generation.
        wiki_pages: Pre-rendered wiki pages keyed by display title (fallback).
        workers:    Parallel synthesis workers (default 1 = sequential).

    Returns:
        dict mapping display title → synthesized Markdown page.
    """
    concept_names = sorted(grouped.keys())
    all_display_titles = {normalize_page_title(c) for c in concept_names}

    concept_chunks: dict[str, set[str]] = {
        c: {f.source_chunk_id for f in facts if f.source_chunk_id}
        for c, facts in grouped.items()
    }

    def _synthesize_one(concept: str) -> tuple[str, str]:
        display_title = normalize_page_title(concept)
        fallback = wiki_pages.get(display_title, "")

        facts = grouped[concept]
        fact_contents = list(unique_fact_contents(facts))
        if not fact_contents:
            return display_title, fallback

        related = build_related_concepts_by_chunks(
            concept, concept_chunks, concept_names, max_related=5, grouped=grouped
        )
        related_titles = [normalize_page_title(r) for r in related]

        prompt = _build_synthesis_prompt(display_title, fact_contents, related_titles)

        try:
            raw = backend.generate(prompt)
        except Exception as exc:
            logging.warning("Synthesis failed for '%s': %s", concept, exc)
            return display_title, fallback

        if not raw or not raw.strip():
            logging.warning("Synthesis returned empty response for '%s'", concept)
            return display_title, fallback

        body = raw.strip()
        body = _strip_auto_sections(body)
        body = _ensure_title_heading(body, display_title)
        body = inject_wikilinks(body, all_display_titles, display_title)

        frontmatter = _extract_frontmatter(fallback)
        tail = _extract_tail_sections(fallback)

        parts = [p for p in [frontmatter, body, tail] if p]
        page = "\n\n".join(parts)

        print(f"  Pass 3 [synthesis]: wrote '{display_title}'")
        return display_title, page

    results: dict[str, str] = {}

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_concept = {
                executor.submit(_synthesize_one, c): c for c in concept_names
            }
            for future in as_completed(future_to_concept):
                concept = future_to_concept[future]
                try:
                    title, page = future.result()
                    if page:
                        results[title] = page
                except Exception as exc:
                    logging.warning("Synthesis error for '%s': %s", concept, exc)
                    display_title = normalize_page_title(concept)
                    if wiki_pages.get(display_title):
                        results[display_title] = wiki_pages[display_title]
    else:
        for concept in concept_names:
            title, page = _synthesize_one(concept)
            if page:
                results[title] = page

    return results
