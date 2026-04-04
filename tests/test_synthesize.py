"""Tests for generate.synthesize (Pass 3)."""

import pytest
from extract.fact_extractor import Fact
from generate.synthesize import (
    _build_synthesis_prompt,
    _extract_frontmatter,
    _extract_tail_sections,
    _strip_auto_sections,
    _ensure_title_heading,
    synthesize_pages,
)


def _fact(content: str, concept: str = "Inventory Turnover", src: str = "doc.pdf::chunk-1") -> Fact:
    return Fact(id="f1", concept=concept, content=content, source_chunk_id=src)


WIKI_PAGE = """\
---
tags:
  - doc
folder: doc
---
# Inventory Turnover

**Inventory Turnover** measures how quickly inventory sells.

---

## Formula

```
Inventory Turnover = COGS ÷ Average Inventory
```

---

## Related Concepts
- [[Cost of Goods Sold]]
- [[Days Sales in Inventory]]

---

## Sources
- doc.pdf"""


class TestBuildSynthesisPrompt:
    def test_includes_all_facts(self):
        facts = ["Fact one.", "Fact two.", "Fact three."]
        prompt = _build_synthesis_prompt("Inventory Turnover", facts, [])
        assert "Fact one." in prompt
        assert "Fact two." in prompt
        assert "Fact three." in prompt

    def test_includes_concept_title(self):
        prompt = _build_synthesis_prompt("Inventory Turnover", ["A fact."], [])
        assert "Inventory Turnover" in prompt

    def test_includes_related_titles(self):
        prompt = _build_synthesis_prompt("Inventory Turnover", ["A fact."], ["Cost of Goods Sold", "FIFO"])
        assert "[[Cost of Goods Sold]]" in prompt
        assert "[[FIFO]]" in prompt

    def test_no_related_says_none(self):
        prompt = _build_synthesis_prompt("Concept", ["A fact."], [])
        assert "none" in prompt.lower()

    def test_instructs_no_related_concepts_section(self):
        prompt = _build_synthesis_prompt("Concept", ["A fact."], [])
        assert "Related Concepts" in prompt
        assert "appended automatically" in prompt


class TestSectionHelpers:
    def test_extract_frontmatter_returns_yaml_block(self):
        fm = _extract_frontmatter(WIKI_PAGE)
        assert fm.startswith("---")
        assert "tags:" in fm
        assert fm.endswith("---")

    def test_extract_frontmatter_empty_when_absent(self):
        assert _extract_frontmatter("# No frontmatter here\n\nBody.") == ""

    def test_extract_tail_returns_related_onwards(self):
        tail = _extract_tail_sections(WIKI_PAGE)
        assert "## Related Concepts" in tail
        assert "[[Cost of Goods Sold]]" in tail
        assert "## Sources" in tail

    def test_extract_tail_empty_when_absent(self):
        assert _extract_tail_sections("# Title\n\nJust a body.") == ""

    def test_strip_auto_sections_removes_related_concepts(self):
        body = "# Title\n\nIntro.\n\n## Formula\n\n```\nX = Y\n```\n\n## Related Concepts\n- [[A]]\n## Sources\n- doc.pdf"
        stripped = _strip_auto_sections(body)
        assert "## Related Concepts" not in stripped
        assert "## Sources" not in stripped
        assert "## Formula" in stripped

    def test_strip_auto_sections_no_op_when_absent(self):
        body = "# Title\n\nIntro.\n\n## Formula\n\n```\nX = Y\n```"
        assert _strip_auto_sections(body) == body

    def test_ensure_title_heading_prepends_when_missing(self):
        body = "Intro paragraph without a heading."
        result = _ensure_title_heading(body, "My Concept")
        assert result.startswith("# My Concept")

    def test_ensure_title_heading_no_op_when_present(self):
        body = "# My Concept\n\nIntro."
        assert _ensure_title_heading(body, "My Concept") == body


class TestSynthesizePages:
    def _grouped(self) -> dict:
        return {
            "Inventory Turnover": [
                _fact("Inventory turnover measures how quickly inventory is sold."),
                _fact("Inventory Turnover = COGS ÷ Average Inventory"),
            ]
        }

    def _wiki_pages(self) -> dict:
        return {"Inventory Turnover": WIKI_PAGE}

    def test_uses_llm_output_on_success(self, mock_backend):
        mock_backend.responses = [
            "# Inventory Turnover\n\nSynthesized intro.\n\n## Formula\n\n```\nX = Y\n```"
        ]
        pages = synthesize_pages(self._grouped(), mock_backend, self._wiki_pages())
        assert "Inventory Turnover" in pages
        assert "Synthesized intro." in pages["Inventory Turnover"]

    def test_appends_tail_from_wiki_page(self, mock_backend):
        mock_backend.responses = ["# Inventory Turnover\n\nSynthesized intro."]
        pages = synthesize_pages(self._grouped(), mock_backend, self._wiki_pages())
        assert "## Related Concepts" in pages["Inventory Turnover"]
        assert "[[Cost of Goods Sold]]" in pages["Inventory Turnover"]

    def test_prepends_frontmatter_from_wiki_page(self, mock_backend):
        mock_backend.responses = ["# Inventory Turnover\n\nSynthesized intro."]
        pages = synthesize_pages(self._grouped(), mock_backend, self._wiki_pages())
        assert pages["Inventory Turnover"].startswith("---")
        assert "tags:" in pages["Inventory Turnover"]

    def test_falls_back_to_wiki_page_on_empty_response(self, mock_backend):
        mock_backend.responses = [""]
        pages = synthesize_pages(self._grouped(), mock_backend, self._wiki_pages())
        assert pages["Inventory Turnover"] == WIKI_PAGE

    def test_falls_back_to_wiki_page_on_backend_failure(self, mock_backend):
        mock_backend.should_fail = True
        pages = synthesize_pages(self._grouped(), mock_backend, self._wiki_pages())
        assert pages["Inventory Turnover"] == WIKI_PAGE

    def test_strips_llm_added_related_concepts(self, mock_backend):
        mock_backend.responses = [
            "# Inventory Turnover\n\nIntro.\n\n## Related Concepts\n- [[Something]]\n## Sources\n- doc.pdf"
        ]
        pages = synthesize_pages(self._grouped(), mock_backend, self._wiki_pages())
        # LLM-added Related Concepts section should be stripped; wiki tail replaces it
        page = pages["Inventory Turnover"]
        # Only one ## Related Concepts section should appear
        assert page.count("## Related Concepts") == 1
        assert "[[Cost of Goods Sold]]" in page  # from wiki tail

    def test_parallel_matches_sequential(self, mock_backend):
        grouped = {
            "Concept A": [_fact("Fact about A.", concept="Concept A")],
            "Concept B": [_fact("Fact about B.", concept="Concept B")],
        }
        wiki = {
            "Concept A": "# Concept A\n\n## Related Concepts\n- None\n## Sources\n- doc.pdf",
            "Concept B": "# Concept B\n\n## Related Concepts\n- None\n## Sources\n- doc.pdf",
        }
        mock_backend.responses = [
            "# Concept A\n\nSynthesized A.",
            "# Concept B\n\nSynthesized B.",
        ]
        seq = synthesize_pages(grouped, mock_backend, wiki, workers=1)

        mock_backend.responses = [
            "# Concept A\n\nSynthesized A.",
            "# Concept B\n\nSynthesized B.",
        ]
        par = synthesize_pages(grouped, mock_backend, wiki, workers=2)

        assert set(seq.keys()) == set(par.keys())
