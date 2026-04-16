"""Tests for postprocess.enrich_thin_concepts parallelism and ordering."""

from unittest.mock import MagicMock

from extract.fact_extractor import Fact
from postprocess import (
    enrich_thin_concepts,
    evaluate_concepts,
    generate_redirect_pages,
)


def _make_grouped(thin_count: int = 3, fat_count: int = 2) -> dict[str, list[Fact]]:
    """Build a grouped dict with some thin (< 4 facts) and fat (>= 4 facts) concepts."""
    grouped: dict[str, list[Fact]] = {}
    for i in range(thin_count):
        concept = f"Thin Concept {i}"
        grouped[concept] = [
            Fact(id=f"t{i}-{j}", concept=concept,
                 content=f"Fact {j} about {concept}.", source_chunk_id="chunk-1")
            for j in range(2)
        ]
    for i in range(fat_count):
        concept = f"Fat Concept {i}"
        grouped[concept] = [
            Fact(id=f"f{i}-{j}", concept=concept,
                 content=f"Fact {j} about {concept}.", source_chunk_id="chunk-1")
            for j in range(5)
        ]
    return grouped


def _make_backend(response: str = '["Extra fact A.", "Extra fact B."]') -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = response
    return backend


def test_enrich_preserves_all_concepts() -> None:
    """All concepts are present in the output regardless of whether they were enriched."""
    grouped = _make_grouped(thin_count=3, fat_count=2)
    backend = _make_backend()
    result = enrich_thin_concepts(grouped, backend, min_facts=4, workers=1)
    assert set(result.keys()) == set(grouped.keys())


def test_enrich_fat_concepts_unchanged() -> None:
    """Concepts with >= min_facts facts are not touched."""
    grouped = _make_grouped(thin_count=0, fat_count=3)
    backend = _make_backend()
    result = enrich_thin_concepts(grouped, backend, min_facts=4, workers=1)
    backend.generate.assert_not_called()
    for concept, facts in grouped.items():
        assert len(result[concept]) == len(facts)


def test_enrich_thin_concepts_receive_new_facts() -> None:
    """Thin concepts get the facts returned by the backend appended."""
    grouped = _make_grouped(thin_count=2, fat_count=0)
    backend = _make_backend('["New fact one.", "New fact two."]')
    result = enrich_thin_concepts(grouped, backend, min_facts=4, workers=1)
    for concept in grouped:
        assert len(result[concept]) > len(grouped[concept])


def test_enrich_parallel_matches_sequential_keys() -> None:
    """Parallel enrichment produces the same concept keys as sequential."""
    grouped = _make_grouped(thin_count=4, fat_count=2)
    backend_seq = _make_backend()
    backend_par = _make_backend()
    seq = enrich_thin_concepts(grouped, backend_seq, min_facts=4, workers=1)
    par = enrich_thin_concepts(grouped, backend_par, min_facts=4, workers=4)
    assert set(seq.keys()) == set(par.keys())


def test_enrich_parallel_key_order_is_deterministic() -> None:
    """Two parallel runs on the same input must produce keys in the same order."""
    grouped = _make_grouped(thin_count=4, fat_count=2)
    run1 = list(enrich_thin_concepts(grouped, _make_backend(), min_facts=4, workers=4).keys())
    run2 = list(enrich_thin_concepts(grouped, _make_backend(), min_facts=4, workers=4).keys())
    assert run1 == run2


def test_enrich_backend_failure_skips_concept() -> None:
    """If the backend raises, that concept keeps its original facts and others proceed."""
    grouped = _make_grouped(thin_count=3, fat_count=0)
    backend = MagicMock()
    backend.generate.side_effect = RuntimeError("API down")
    result = enrich_thin_concepts(grouped, backend, min_facts=4, workers=1)
    assert set(result.keys()) == set(grouped.keys())
    for concept in grouped:
        assert len(result[concept]) == len(grouped[concept])


def _fact(concept: str, n: int = 1) -> list[Fact]:
    return [
        Fact(id=f"{concept}-{i}", concept=concept,
             content=f"Fact {i}.", source_chunk_id="chunk-1")
        for i in range(n)
    ]


class TestNearDuplicatesAndRedirects:
    """Regression tests for the accounting-vault merge bugs.

    These specific pairs were incorrectly redirected in the April 2026 run:
    Accounts Payable → Accounts Receivable, Fixed Cost → Fixed Asset,
    Direct Material → Direct Labor, Activity Base → Activity Rate,
    Indirect Materials Expense → Direct Material.
    """

    def test_cousin_pairs_not_flagged_as_near_duplicates(self) -> None:
        grouped = {
            "Accounts Payable": _fact("Accounts Payable", 2),
            "Accounts Receivable": _fact("Accounts Receivable", 5),
            "Fixed Cost": _fact("Fixed Cost", 2),
            "Fixed Asset": _fact("Fixed Asset", 5),
            "Direct Material": _fact("Direct Material", 2),
            "Direct Labor": _fact("Direct Labor", 5),
            "Activity Base": _fact("Activity Base", 2),
            "Activity Rate": _fact("Activity Rate", 5),
        }
        result = evaluate_concepts(grouped)
        pairs = {frozenset(p) for p in result["near_duplicates"]}
        assert frozenset({"Accounts Payable", "Accounts Receivable"}) not in pairs
        assert frozenset({"Fixed Cost", "Fixed Asset"}) not in pairs
        assert frozenset({"Direct Material", "Direct Labor"}) not in pairs
        assert frozenset({"Activity Base", "Activity Rate"}) not in pairs

    def test_substring_match_uses_token_boundaries(self) -> None:
        # "direct material" is a string-substring of "indirect materials
        # expense", but NOT a token-level subsequence — the prefix of
        # "indirect" bleeds into "direct".  Must not be flagged.
        grouped = {
            "Direct Material": _fact("Direct Material", 5),
            "Indirect Materials Expense": _fact("Indirect Materials Expense", 2),
        }
        result = evaluate_concepts(grouped)
        pairs = {frozenset(p) for p in result["near_duplicates"]}
        assert frozenset({"Direct Material", "Indirect Materials Expense"}) not in pairs

    def test_generate_redirect_pages_rejects_cousins(self) -> None:
        # Even if a caller somehow passes a cousin pair through as a
        # near-duplicate, the redirect generator must refuse to emit a stub.
        grouped = {
            "Fixed Cost": _fact("Fixed Cost", 2),
            "Fixed Asset": _fact("Fixed Asset", 5),
        }
        redirects = generate_redirect_pages([("Fixed Cost", "Fixed Asset")], grouped)
        assert redirects == {}

    def test_generate_redirect_pages_rejects_antonyms(self) -> None:
        grouped = {
            "Accounts Payable": _fact("Accounts Payable", 2),
            "Accounts Receivable": _fact("Accounts Receivable", 5),
        }
        redirects = generate_redirect_pages(
            [("Accounts Payable", "Accounts Receivable")], grouped
        )
        assert redirects == {}

    def test_generate_redirect_stub_has_backend_attribution(self) -> None:
        # Substring-style near-duplicates still produce redirects — those
        # redirects must carry the `generated_by_backend: redirect` marker so
        # the QA tool doesn't bucket them as "unknown".
        grouped = {
            "Inventory Turnover": _fact("Inventory Turnover", 5),
            "Inventory Turnover Concept": _fact("Inventory Turnover Concept", 2),
        }
        redirects = generate_redirect_pages(
            [("Inventory Turnover", "Inventory Turnover Concept")], grouped
        )
        assert redirects, "expected a redirect stub for legit near-duplicate"
        page = next(iter(redirects.values()))
        assert "generated_by_backend: redirect" in page
        assert "generated_by_model: none" in page
