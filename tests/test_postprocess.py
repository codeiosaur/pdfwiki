"""Tests for postprocess.enrich_thin_concepts parallelism and ordering."""

from unittest.mock import MagicMock

from extract.fact_extractor import Fact
from postprocess import enrich_thin_concepts


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
