"""Deterministic evaluation fixtures and scoring for PDF-to-Wiki."""

from evaluation.fixtures import EvaluationCase, GoldConcept, load_fixture_corpus
from evaluation.scoring import score_case, score_suite

__all__ = [
    "EvaluationCase",
    "GoldConcept",
    "load_fixture_corpus",
    "score_case",
    "score_suite",
]
