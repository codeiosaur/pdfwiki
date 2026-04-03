"""Tests for ingest.pdf_loader._chunk_text — chunker edge cases and robustness."""

import pytest

from ingest.pdf_loader import Chunk, _chunk_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _words(n: int, word: str = "word") -> str:
    """Return a space-separated string of *n* copies of *word*."""
    return " ".join([word] * n)


def _sentence(n_words: int) -> str:
    """Return a sentence of *n_words* words ending with a period."""
    return _words(n_words) + "."


# ---------------------------------------------------------------------------
# Very short text (< min_chunk_words)
# ---------------------------------------------------------------------------

class TestShortText:
    def test_single_short_sentence_produces_one_chunk(self):
        text = "This is a very short sentence."
        chunks, fallback_slices = _chunk_text(text, "test.pdf", min_chunk_words=800, max_chunk_words=1200)
        assert len(chunks) == 1
        assert fallback_slices == 0

    def test_short_text_content_preserved(self):
        text = "Inventory valuation is important."
        chunks, _ = _chunk_text(text, "test.pdf", min_chunk_words=800, max_chunk_words=1200)
        assert chunks[0].text == text.strip()

    def test_short_text_has_correct_source(self):
        chunks, _ = _chunk_text("Short text.", "my_doc.pdf", min_chunk_words=800, max_chunk_words=1200)
        assert chunks[0].source == "my_doc.pdf"

    def test_empty_text_returns_no_chunks(self):
        chunks, fallback_slices = _chunk_text("", "test.pdf")
        assert chunks == []
        assert fallback_slices == 0

    def test_whitespace_only_returns_no_chunks(self):
        chunks, _ = _chunk_text("   \n\n   ", "test.pdf")
        assert chunks == []


# ---------------------------------------------------------------------------
# No sentence boundaries (triggers word-slice fallback)
# ---------------------------------------------------------------------------

class TestNoSentenceBoundaries:
    def test_text_with_no_periods_triggers_word_slice(self):
        # A single very long "sentence" with no punctuation — forces word-level fallback
        text = _words(2500)
        chunks, fallback_slices = _chunk_text(text, "ocr.pdf", min_chunk_words=800, max_chunk_words=1200)
        assert len(chunks) >= 2
        assert fallback_slices >= 2

    def test_word_slice_chunks_respect_max_chunk_words(self):
        text = _words(2500)
        chunks, _ = _chunk_text(text, "ocr.pdf", min_chunk_words=800, max_chunk_words=1200)
        for chunk in chunks:
            assert len(chunk.text.split()) <= 1200

    def test_overall_fallback_triggers_when_sentence_split_produces_nothing(self):
        # A single word repeated — no sentence-ending punctuation at all
        text = _words(3000, word="word")
        chunks, fallback_slices = _chunk_text(text, "test.pdf", min_chunk_words=800, max_chunk_words=1200)
        assert len(chunks) > 0
        assert fallback_slices > 0

    def test_oversized_single_sentence_is_word_sliced(self):
        # One giant sentence longer than max_chunk_words
        text = _sentence(1500)
        chunks, fallback_slices = _chunk_text(text, "test.pdf", min_chunk_words=800, max_chunk_words=1200)
        assert len(chunks) >= 2
        assert fallback_slices >= 2
        for chunk in chunks:
            assert len(chunk.text.split()) <= 1200

    def test_fallback_slices_zero_for_normal_text(self):
        # Normal multi-sentence text within word limits → no fallback
        sentences = " ".join(_sentence(50) for _ in range(20))
        _, fallback_slices = _chunk_text(sentences, "test.pdf", min_chunk_words=800, max_chunk_words=1200)
        assert fallback_slices == 0


# ---------------------------------------------------------------------------
# Heading-rich text (many short sentences)
# ---------------------------------------------------------------------------

class TestHeadingRichText:
    def _heading_rich(self, n_headings: int = 30, body_words: int = 30) -> str:
        """Alternate short heading sentences with short body sentences."""
        lines = []
        for i in range(n_headings):
            lines.append(f"Chapter {i} Heading.")
            lines.append(_sentence(body_words))
        return " ".join(lines)

    def test_heading_rich_text_produces_at_least_one_chunk(self):
        text = self._heading_rich()
        chunks, _ = _chunk_text(text, "textbook.pdf", min_chunk_words=800, max_chunk_words=1200)
        assert len(chunks) >= 1

    def test_heading_rich_chunks_under_max_size(self):
        text = self._heading_rich(n_headings=50, body_words=40)
        chunks, _ = _chunk_text(text, "textbook.pdf", min_chunk_words=800, max_chunk_words=1200)
        for chunk in chunks:
            assert len(chunk.text.split()) <= 1200

    def test_all_words_accounted_for(self):
        text = self._heading_rich(n_headings=20, body_words=20)
        chunks, _ = _chunk_text(text, "textbook.pdf", min_chunk_words=800, max_chunk_words=1200)
        total_chunk_words = sum(len(c.text.split()) for c in chunks)
        original_words = len(text.split())
        # Allow ±5% for sentence-boundary re-joining artefacts
        assert abs(total_chunk_words - original_words) / original_words < 0.05

    def test_each_chunk_has_unique_id(self):
        text = self._heading_rich()
        chunks, _ = _chunk_text(text, "textbook.pdf")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Custom word limits
# ---------------------------------------------------------------------------

class TestCustomWordLimits:
    def test_small_max_produces_more_chunks(self):
        text = " ".join(_sentence(30) for _ in range(20))
        chunks_small, _ = _chunk_text(text, "test.pdf", min_chunk_words=10, max_chunk_words=50)
        chunks_large, _ = _chunk_text(text, "test.pdf", min_chunk_words=100, max_chunk_words=300)
        assert len(chunks_small) >= len(chunks_large)

    def test_min_equals_max_still_chunks(self):
        text = " ".join(_sentence(30) for _ in range(5))
        chunks, _ = _chunk_text(text, "test.pdf", min_chunk_words=50, max_chunk_words=50)
        assert len(chunks) >= 1
