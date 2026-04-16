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

from backend.pool import BackendPool


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

# Maximum facts sent per synthesis request.  Concepts with more facts than this
# have their list truncated.  Models plateau in quality around 20-25 facts while
# additional facts increase input token cost without benefit.
_MAX_SYNTHESIS_FACTS = 25

# Static instructions sent as the system message on every synthesis call.
# Keeping this constant lets providers cache it after the first request.
_SYNTHESIS_SYSTEM_PROMPT = """You are writing Obsidian wiki study-note pages for an academic subject.

Write the page body in Markdown:
- Start with # {title} as the page title (the exact title will be given in the user message).
- Follow with a short intro paragraph (1–3 sentences) defining the concept. Bold the concept name on first use (**Concept Name**).
- Use ## section headers that fit the content. Choose from: Formula, How It Works, Interpretation, Comparison, Worked Example, Key Takeaways, Cautions. Only include sections that have content from the source facts.
- If a formula is present, render it in a fenced code block.
- If multiple variants or methods exist, create a Markdown comparison table.
- Under Worked Example, include any concrete numbers from the source facts.
- Link related concepts using [[wikilinks]] on first mention in the body text.
- Do NOT output a ## Related Concepts, ## See Also, ## Sources section, or YAML frontmatter — those are appended automatically.
- Do NOT invent facts, examples, or numbers not provided.
- Do NOT reference source facts by number (e.g. "(fact 3)") — write prose that integrates the information naturally.

Output ONLY the Markdown page body, starting with # followed by the concept title."""


def _build_synthesis_prompt(
    display_title: str,
    fact_contents: List[str],
    related_titles: List[str],
) -> str:
    """Build the user-turn prompt (variable per concept). System prompt is separate."""
    numbered = "\n".join(f"  {i + 1}. {f}" for i, f in enumerate(fact_contents))
    link_hints = (
        ", ".join(f"[[{t}]]" for t in related_titles)
        if related_titles
        else "none"
    )
    return f"""Write a wiki page for the concept "{display_title}".

Source facts — use ONLY these; do not invent additional information:
{numbered}

Other concepts you may link with [[wikilinks]] (use the exact titles shown): {link_hints}

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


def _add_backend_attribution(frontmatter: str, backend_label: str, model: str) -> str:
    """
    Add backend and model attribution to the YAML frontmatter.

    If frontmatter is empty, create a minimal one.  Otherwise, insert the
    attribution before the closing '---'.
    """
    if not frontmatter:
        # Create minimal frontmatter if none exists
        return f"---\ngenerated_by_backend: {backend_label}\ngenerated_by_model: {model}\n---"

    # Insert before closing '---'
    closing_idx = frontmatter.rfind("\n---")
    if closing_idx == -1:
        return frontmatter

    # Insert new fields before the closing delimiter
    insert_point = closing_idx
    new_lines = f"generated_by_backend: {backend_label}\ngenerated_by_model: {model}\n"
    return frontmatter[:insert_point] + "\n" + new_lines + frontmatter[insert_point:]


def _extract_tail_sections(wiki_page: str) -> str:
    """Return everything from ## Related Concepts onward in the wiki page."""
    for marker in ["\n## Related Concepts", "\n\n---\n\n## Related Concepts"]:
        idx = wiki_page.find(marker)
        if idx != -1:
            return wiki_page[idx:].strip()
    return ""


def _strip_thinking_tags(body: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. Qwen3)."""
    return re.sub(r"<think>.*?</think>", "", body, flags=re.DOTALL).strip()


def _strip_fact_citations(body: str) -> str:
    """Remove (fact N) / (facts N, M) citations the LLM may leave from the numbered prompt."""
    return re.sub(r"\s*\(facts?\s[\d,\s]+\)", "", body)


def _strip_fact_labels(body: str) -> str:
    """Remove 'Fact <word>.:' labels emitted by enrichment that survive unsynthesized pages."""
    return re.sub(r"\bFact\s+\w+\.:\s*", "", body)


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


def _mark_fallback(page: str, reason: str) -> str:
    """Stamp a fallback page's frontmatter with `generated_by_backend: fallback-<reason>`.

    Pass 3 attribution is normally added in the success path.  When synthesis
    fails and we return the pre-rendered wiki page, this helper records why so
    downstream QA tools can bucket fallbacks separately rather than lumping
    them into an "unknown" backend.
    """
    if not page:
        return page
    fm = _extract_frontmatter(page)
    if fm and "generated_by_backend:" in fm:
        return page
    body = page[len(fm):] if fm else page
    new_fm = _add_backend_attribution(fm, f"fallback-{reason}", "none")
    separator = "" if body.startswith("\n") else "\n"
    return new_fm + separator + body


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------

def synthesize_pages(
    grouped: dict[str, list[Fact]],
    backend: "LLMBackend",
    wiki_pages: dict[str, str],
    workers: int = 1,
    streaming: bool = False,
) -> dict[str, str] | list[tuple[str, str]]:
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
        streaming:  If True, return a list of (title, page) tuples as they complete.
                    If False (default), return a dict after all synthesis completes.

    Returns:
        If streaming=False: dict mapping display title → synthesized Markdown page.
        If streaming=True: list of (title, page) tuples (maintains order from work-stealing).
    """
    concept_names = sorted(grouped.keys())
    all_display_titles = {normalize_page_title(c) for c in concept_names}

    concept_chunks: dict[str, set[str]] = {
        c: {f.source_chunk_id for f in facts if f.source_chunk_id}
        for c, facts in grouped.items()
    }

    def _synthesize_one(concept: str, synth_backend: "LLMBackend") -> tuple[str, str]:
        """Synthesize a single concept using the given backend."""
        import time as _time
        _t0 = _time.perf_counter()
        display_title = normalize_page_title(concept)
        fallback = wiki_pages.get(display_title, "")

        facts = grouped[concept]
        fact_contents = list(unique_fact_contents(facts))[:_MAX_SYNTHESIS_FACTS]
        if not fact_contents:
            return display_title, _mark_fallback(fallback, "no-facts")

        related = build_related_concepts_by_chunks(
            concept, concept_chunks, concept_names, max_related=5, grouped=grouped
        )
        related_titles = [normalize_page_title(r) for r in related]

        prompt = _build_synthesis_prompt(display_title, fact_contents, related_titles)

        try:
            raw = synth_backend.generate(prompt, context=display_title,
                                         system_prompt=_SYNTHESIS_SYSTEM_PROMPT)
        except Exception as exc:
            logging.warning("Synthesis failed for '%s': %s", concept, exc)
            return display_title, _mark_fallback(fallback, "exception")

        if not raw or not raw.strip():
            logging.warning("Synthesis returned empty response for '%s'", concept)
            return display_title, _mark_fallback(fallback, "empty-response")

        body = raw.strip()
        body = _strip_thinking_tags(body)
        body = _strip_fact_citations(body)
        body = _strip_fact_labels(body)
        if not body:
            logging.warning("Synthesis response for '%s' was all thinking tokens, no content", concept)
            return display_title, _mark_fallback(fallback, "thinking-only")
        body = _strip_auto_sections(body)
        body = _ensure_title_heading(body, display_title)
        body = inject_wikilinks(body, all_display_titles, display_title)

        frontmatter = _extract_frontmatter(fallback)
        tail = _extract_tail_sections(fallback)

        # Add backend attribution to frontmatter
        used_label = synth_backend.last_used_label() if hasattr(synth_backend, "last_used_label") else synth_backend.label
        model = synth_backend.model if hasattr(synth_backend, "model") else "unknown"
        frontmatter = _add_backend_attribution(frontmatter, used_label, model)

        parts = [p for p in [frontmatter, body, tail] if p]
        page = "\n\n".join(parts)

        elapsed = _time.perf_counter() - _t0
        print(f"  Pass 3 [synthesis]: wrote '{display_title}' [{used_label}] ({elapsed:.0f}s)")
        return display_title, page

    results_list: list[tuple[str, str]] = []
    results: dict[str, str] = {}

    # Use dispatch() if backend is a BackendPool, otherwise use ThreadPoolExecutor
    if isinstance(backend, BackendPool):
        def _process_batch(concepts: List[str], synth_backend: "LLMBackend", batch_size: int) -> List[tuple[str, str]]:
            """Process a batch of concepts using the given backend."""
            batch_results = []
            for concept in concepts:
                try:
                    title, page = _synthesize_one(concept, synth_backend)
                    if page:
                        batch_results.append((title, page))
                except Exception as exc:
                    logging.warning("Synthesis error for concept: %s", exc)
                    display_title = normalize_page_title(concept)
                    if wiki_pages.get(display_title):
                        batch_results.append((
                            display_title,
                            _mark_fallback(wiki_pages[display_title], "dispatch-error"),
                        ))
            return batch_results

        # Use work-stealing dispatch: each concept is an item, workers pull dynamically
        batch_results = backend.dispatch(
            concept_names,
            _process_batch,
            default_batch_size=1  # One concept per batch unit
        )
        # Flatten results: dispatch returns list of (title, page) tuples
        for title, page in batch_results:
            results_list.append((title, page))
            results[title] = page
    elif workers > 1:
        # Fallback for non-pool backends: use ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_concept = {
                executor.submit(_synthesize_one, c, backend): c for c in concept_names
            }
            for future in as_completed(future_to_concept):
                concept = future_to_concept[future]
                try:
                    title, page = future.result()
                    if page:
                        results_list.append((title, page))
                        results[title] = page
                except Exception as exc:
                    logging.warning("Synthesis error for '%s': %s", concept, exc)
                    display_title = normalize_page_title(concept)
                    if wiki_pages.get(display_title):
                        marked = _mark_fallback(wiki_pages[display_title], "executor-error")
                        results_list.append((display_title, marked))
                        results[display_title] = marked
    else:
        # Sequential synthesis
        for concept in concept_names:
            title, page = _synthesize_one(concept, backend)
            if page:
                results_list.append((title, page))
                results[title] = page

    return results_list if streaming else results
