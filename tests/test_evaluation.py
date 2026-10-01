import pytest

from extract.fact_extractor import Fact
from evaluation.fixtures import load_fixture_corpus
from evaluation.scoring import score_case, score_suite


def _case(case_id: str):
    return next(case for case in load_fixture_corpus() if case.id == case_id)


def test_fixture_corpus_covers_expected_source_shapes() -> None:
    cases = load_fixture_corpus()

    assert [case.id for case in cases] == [
        "clean-accounting",
        "slide-bullets",
        "ocr-notes",
        "diagram-labels",
        "noise-only",
    ]
    assert all(chunk.location_label for case in cases for chunk in case.chunks)
    assert _case("noise-only").noise_only


def test_score_case_accepts_aliases_and_composed_source_ids() -> None:
    case = _case("clean-accounting")
    facts = [
        Fact("f1", "FIFO", "FIFO uses the oldest inventory costs.", "clean-accounting.pdf::clean-01"),
        Fact("f2", "Inventory Turnover Ratio", "COGS is divided by average inventory.", "clean-accounting.pdf::clean-02"),
    ]
    pages = {
        "FIFO": "# FIFO\n\nThe oldest inventory is sold first.",
        "Inventory Turnover Ratio": "# Inventory Turnover\n\nCost of goods sold / average inventory.",
    }

    result = score_case(case, facts, pages)

    assert result["metrics"]["source_id_preservation_rate"] == 1.0
    assert result["metrics"]["gold_concept_recall"] == 1.0
    assert result["metrics"]["page_recall"] == 1.0
    assert result["metrics"]["missing_required_term_count"] == 0
    assert result["metrics"]["broken_wikilink_count"] == 0


def test_score_case_reports_contract_failures_deterministically() -> None:
    case = _case("slide-bullets")
    facts = [
        Fact("f1", "Operations Management", "Footer text", "unknown-id"),
        Fact("f2", "Week 3", "Footer text", ""),
    ]
    pages = {"Operations Management": "not a page with [[a broken link"}

    result = score_case(case, facts, pages)

    assert result["metrics"]["source_id_preservation_rate"] == 0.0
    assert result["metrics"]["missing_source_id_count"] == 1
    assert result["metrics"]["gold_concept_recall"] == 0.0
    assert result["metrics"]["forbidden_concept_count"] == 2
    assert result["metrics"]["page_heading_rate"] == 0.0
    assert result["metrics"]["broken_wikilink_count"] == 1


def test_score_suite_returns_macro_averages() -> None:
    cases = (_case("clean-accounting"), _case("noise-only"))
    results = {
        "clean-accounting": (
            [Fact("f1", "FIFO", "Oldest inventory is sold first.", "clean-01")],
            {"FIFO": "# FIFO\n\nOldest inventory."},
        ),
        "noise-only": ([], {}),
    }

    report = score_suite(cases, results)

    assert report["summary"]["case_count"] == 2
    assert report["summary"]["source_id_preservation_rate"] == 1.0
    assert report["summary"]["gold_concept_recall"] == 0.5
    assert report["summary"]["scored_case_counts"]["gold_concept_recall"] == 1
    assert report["summary"]["noise_rejection_pass"] == 1.0


def test_empty_output_is_not_a_perfect_source_or_page_score() -> None:
    result = score_case(_case("clean-accounting"), [], {})
    assert result["metrics"]["source_id_preservation_rate"] is None
    assert result["metrics"]["page_heading_rate"] is None
    assert result["metrics"]["gold_concept_recall"] == 0.0


def test_unlisted_concepts_are_unreviewed_not_assumed_false() -> None:
    case = _case("clean-accounting")
    result = score_case(case, [Fact("f1", "Supply Chain", "A supported fact.", "wrong.pdf::clean-01")])
    assert result["metrics"]["source_id_preservation_rate"] == 0.0
    assert result["metrics"]["confirmed_false_positive_count"] == 0
    assert result["details"]["unreviewed_concepts"] == ["Supply Chain"]


def test_noise_case_rejects_any_output() -> None:
    case = _case("noise-only")
    result = score_case(case, [Fact("f1", "Extra", "A fact.", "noise-01")], {})
    assert result["metrics"]["noise_rejection_pass"] is False
    assert result["metrics"]["confirmed_false_positive_count"] == 1
    assert result["metrics"]["gold_concept_recall"] is None


def test_score_suite_requires_every_case_result() -> None:
    with pytest.raises(ValueError, match="Missing evaluation results"):
        score_suite((_case("clean-accounting"),), {})
