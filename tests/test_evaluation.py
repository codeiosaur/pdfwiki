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
    assert result["metrics"]["valid_markdown_page_rate"] == 0.0
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
    assert report["summary"]["gold_concept_recall"] == 0.75
