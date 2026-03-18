from pdfwiki.extractor import (
    chunk_by_page,
    has_headings,
    has_page_markers,
    has_paragraphs,
    resolve_use_markitdown,
    smart_chunk,
)


def test_chunk_by_page_uses_page_markers_when_present():
    text = "\n\n".join([
        "--- Page 1 ---\nalpha",
        "--- Page 2 ---\nbeta",
        "--- Page 3 ---\ngamma",
        "--- Page 4 ---\ndelta",
    ])

    chunks = chunk_by_page(text, pages_per_chunk=2)

    assert len(chunks) == 2
    assert "alpha" in chunks[0]
    assert "beta" in chunks[0]
    assert "gamma" in chunks[1]
    assert "delta" in chunks[1]


def test_chunk_by_page_falls_back_to_semantic_for_markerless_text():
    # Markerless, slide-like text: short lines separated by blank lines.
    slides = []
    for i in range(1, 121):
        slides.append(
            f"Slide {i}\n"
            "- key idea one with enough detail to avoid tiny chunks\n"
            "- key idea two with enough detail to avoid tiny chunks\n"
            "- key idea three with enough detail to avoid tiny chunks"
        )
    text = "\n\n".join(slides)

    chunks = smart_chunk(text, pages_per_chunk=2)

    # Old behavior often collapsed this into single-digit chunks.
    assert len(chunks) >= 15
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_detection_helpers_cover_markers_headings_and_paragraphs():
    text = "# Title\n\nparagraph one\n\nparagraph two\n\n--- Page 1 ---\ntext"
    assert has_page_markers(text)
    assert has_headings(text)
    assert has_paragraphs(text)


def test_smart_chunk_priority_prefers_page_markers_over_headings():
    text = "\n\n".join([
        "--- Page 1 ---\n# Slide One\na",
        "--- Page 2 ---\n# Slide Two\nb",
        "--- Page 3 ---\n# Slide Three\nc",
    ])
    chunks = smart_chunk(text, pages_per_chunk=1)
    assert len(chunks) == 3


def test_smart_chunk_enforces_max_chars_on_all_strategies():
    text = "# Heading\n\n" + ("A" * 8000) + "\n\n" + ("B" * 8000)
    chunks = smart_chunk(text, max_chars=1200, overlap=100)
    assert chunks
    assert all(len(c) <= 1200 for c in chunks)


def test_resolve_use_markitdown_prefers_explicit_argument(monkeypatch):
    monkeypatch.setenv("USE_MARKITDOWN", "false")
    assert resolve_use_markitdown(True) is True
    assert resolve_use_markitdown(False) is False


def test_resolve_use_markitdown_reads_primary_env(monkeypatch):
    monkeypatch.setenv("USE_MARKITDOWN", "true")
    assert resolve_use_markitdown(None) is True
    monkeypatch.setenv("USE_MARKITDOWN", "0")
    assert resolve_use_markitdown(None) is False


def test_resolve_use_markitdown_uses_compat_alias_when_primary_missing(monkeypatch):
    monkeypatch.delenv("USE_MARKITDOWN", raising=False)
    monkeypatch.setenv("PDF_TO_NOTES_USE_MARKITDOWN", "yes")
    assert resolve_use_markitdown(None) is True
