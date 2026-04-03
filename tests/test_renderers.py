"""Regression tests for generate.renderers page output."""

import re

import pytest

from extract.fact_extractor import Fact
from generate.renderers import generate_pages, generate_pages_enhanced, generate_pages_wiki
from generate.util import build_enhanced_intro


class TestGeneratePagesWikiDoubleTitle:
    """
    Regression: when normalize_page_title() produces a display title that
    differs from the verbatim concept in the definition (e.g. adds parentheses
    for an acronym), emphasize_concept_once() falls back to prepending
    "**Title**: definition text...".  If the definition already opens with the
    same concept words this creates a redundant "Title: Title is ..." pattern.
    The renderer must strip it.
    """

    def _pages_for(self, concept: str, definition: str) -> dict[str, str]:
        facts = [Fact(id="1", concept=concept, content=definition, source_chunk_id="chunk-1")]
        return generate_pages_wiki({concept: facts})

    def test_no_double_title_dsi_concept(self):
        """
        "Days Sales In Inventory DSI" normalizes to "Days Sales in Inventory (DSI)".
        The definition uses "DSI" (no parens), so emphasize_concept_once cannot
        match the full normalized title and prepends it.  After the fix the intro
        must NOT contain the pattern **Title**: Title.
        """
        concept = "Days Sales In Inventory DSI"
        definition = (
            "Days Sales In Inventory DSI is a measure of how many days it takes "
            "a company to sell its average inventory balance."
        )
        pages = self._pages_for(concept, definition)
        assert pages, "Expected at least one page to be generated"
        page_text = next(iter(pages.values()))

        # The bolded-prefix-colon pattern must not appear
        assert "**Days Sales in Inventory (DSI)**: Days" not in page_text

    def test_no_double_title_when_title_matches_definition_start(self):
        """
        When the definition starts with the concept name verbatim, no prefix is
        prepended by emphasize_concept_once, so no stripping is needed and the
        output is clean bold-in-place.
        """
        concept = "Gross Profit"
        definition = "Gross Profit is the revenue remaining after deducting the cost of goods sold."
        pages = self._pages_for(concept, definition)
        page_text = next(iter(pages.values()))

        # Should have bold in-place, NOT a colon-separated prefix
        assert "**Gross Profit**: Gross Profit" not in page_text
        assert "**Gross Profit**" in page_text

    def test_unrelated_definition_keeps_prefix(self):
        """
        If the definition does NOT start with the concept words, the prepended
        bold prefix should be retained (it anchors the intro to the concept).
        """
        concept = "Gross Profit"
        # Definition mentions completely different content — no concept tokens at start
        definition = "Revenue minus the direct cost of producing goods."
        pages = self._pages_for(concept, definition)
        page_text = next(iter(pages.values()))

        # Prefix must be kept because the remainder doesn't start with "gross"
        assert "**Gross Profit**:" in page_text


class TestStandardAndEnhancedWikilinks:
    def test_standard_renderer_links_acronyms_in_lead(self):
        grouped = {
            "First In First Out": [
                Fact(
                    id="1",
                    concept="First In First Out",
                    content="FIFO is a cost flow assumption that affects Cost of Goods Sold.",
                    source_chunk_id="chunk-1",
                ),
                Fact(
                    id="2",
                    concept="First In First Out",
                    content="FIFO assigns the oldest costs first.",
                    source_chunk_id="chunk-1",
                ),
            ],
            "Cost of Goods Sold": [
                Fact(
                    id="3",
                    concept="Cost of Goods Sold",
                    content="Cost of Goods Sold is the expense recognized when inventory is sold.",
                    source_chunk_id="chunk-1",
                ),
            ],
        }

        pages = generate_pages(grouped)
        page_text = pages["First in First Out"]

        assert "[[Cost of Goods Sold]]" in page_text

    def test_enhanced_renderer_links_related_terms_in_sections(self):
        grouped = {
            "Gross Profit": [
                Fact(
                    id="1",
                    concept="Gross Profit",
                    content="Gross Profit is revenue minus the cost of goods sold.",
                    source_chunk_id="chunk-1",
                ),
                Fact(
                    id="2",
                    concept="Gross Profit",
                    content="Gross Profit indicates how much remains to cover operating expenses.",
                    source_chunk_id="chunk-1",
                ),
            ],
            "Cost of Goods Sold": [
                Fact(
                    id="3",
                    concept="Cost of Goods Sold",
                    content="Cost of Goods Sold is the expense recognized when inventory is sold.",
                    source_chunk_id="chunk-1",
                ),
            ],
        }

        pages = generate_pages_enhanced(grouped)
        page_text = pages["Gross Profit"]

        assert "[[Cost of Goods Sold]]" in page_text


class TestWikiArticleStructure:
    def test_wiki_renderer_filters_questions_and_keeps_article_sections(self):
        grouped = {
            "Inventory Valuation": [
                Fact(
                    id="1",
                    concept="Inventory Valuation",
                    content="Inventory valuation is the method used to assign a value to ending inventory.",
                    source_chunk_id="chunk-1",
                ),
                Fact(
                    id="2",
                    concept="Inventory Valuation",
                    content="The lower-of-cost-or-market rule can reduce reported inventory value when market value falls.",
                    source_chunk_id="chunk-1",
                ),
                Fact(
                    id="3",
                    concept="Inventory Valuation",
                    content="When should a company use specific identification for inventory valuation?",
                    source_chunk_id="chunk-1",
                ),
                Fact(
                    id="4",
                    concept="Inventory Valuation",
                    content="For example, a retailer may use FIFO when prices are rising.",
                    source_chunk_id="chunk-1",
                ),
                Fact(
                    id="5",
                    concept="Inventory Valuation",
                    content="However, the method can be misleading when values fluctuate sharply.",
                    source_chunk_id="chunk-1",
                ),
            ],
            "Cost of Goods Sold": [
                Fact(
                    id="6",
                    concept="Cost of Goods Sold",
                    content="Cost of Goods Sold is the expense recognized when inventory is sold.",
                    source_chunk_id="chunk-1",
                ),
                Fact(
                    id="7",
                    concept="Cost of Goods Sold",
                    content="Cost of Goods Sold rises when the cost flow assumption changes reported costs.",
                    source_chunk_id="chunk-1",
                ),
            ],
        }

        pages = generate_pages_wiki(grouped)
        page_text = pages["Inventory Valuation"]

        assert "?" not in page_text
        # Definition header is intentionally removed; definition appears in the
        # intro/lead paragraph instead.
        assert "## Definition" not in page_text
        assert "## Key Takeaways" in page_text or "## How It Works" in page_text
        assert "## Example" not in page_text
        assert "## Cautions" in page_text
        assert "## Related Concepts" in page_text
        assert "[[Cost of Goods Sold]]" in page_text


class TestParallelRendererEquivalence:
    """Parallel rendering must produce identical output to sequential rendering."""

    @staticmethod
    def _make_grouped() -> dict:
        concepts = ["Gross Profit", "Cost of Goods Sold", "Inventory Valuation", "FIFO", "LIFO"]
        grouped = {}
        for i, concept in enumerate(concepts):
            grouped[concept] = [
                Fact(id=f"{i}-1", concept=concept,
                     content=f"{concept} is a key accounting concept used in financial reporting.",
                     source_chunk_id="chunk-1"),
                Fact(id=f"{i}-2", concept=concept,
                     content=f"{concept} affects the balance sheet and income statement.",
                     source_chunk_id="chunk-1"),
            ]
        return grouped

    def test_wiki_parallel_matches_sequential(self):
        grouped = self._make_grouped()
        sequential = generate_pages_wiki(grouped, workers=1)
        parallel = generate_pages_wiki(grouped, workers=4)
        assert set(sequential.keys()) == set(parallel.keys())
        for key in sequential:
            assert sequential[key] == parallel[key], f"Mismatch for concept: {key}"

    def test_wiki_parallel_output_order_is_deterministic(self):
        grouped = self._make_grouped()
        run1 = list(generate_pages_wiki(grouped, workers=4).keys())
        run2 = list(generate_pages_wiki(grouped, workers=4).keys())
        assert run1 == run2


class TestBuildEnhancedIntroSentenceCap:
    """The combined primary + secondary text must not exceed 2 sentences."""

    def test_two_long_sentences_capped(self):
        definition = (
            "A concept is a mental representation that captures the essential "
            "properties of a category. It forms the basis for classification and "
            "reasoning. It is also used in philosophy and cognitive science."
        )
        interpretations = [
            "This enables efficient categorization of new observations. "
            "Without it, each object would be assessed from scratch."
        ]
        result = build_enhanced_intro("Concept", definition, interpretations, [])
        sentences = re.split(r"(?<=[.!?])\s+", result.strip())
        assert len(sentences) <= 2

    def test_single_sentence_primary_no_secondary_unchanged(self):
        definition = "A ratio measures the relationship between two quantities."
        result = build_enhanced_intro("Ratio", definition, [], [])
        assert result.count(".") >= 1
        sentences = re.split(r"(?<=[.!?])\s+", result.strip())
        assert len(sentences) == 1

    def test_primary_plus_secondary_equals_two_sentences(self):
        definition = "A ratio is a comparison of two values."
        interpretations = ["A higher ratio indicates stronger performance."]
        result = build_enhanced_intro("Ratio", definition, interpretations, [])
        sentences = re.split(r"(?<=[.!?])\s+", result.strip())
        assert len(sentences) == 2
