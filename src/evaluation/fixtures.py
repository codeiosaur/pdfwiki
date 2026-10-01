"""Load the versioned, model-independent evaluation fixture corpus."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class EvaluationChunk:
    """A fixed text input representing one source unit for a benchmark case."""

    id: str
    text: str
    source: str
    location_label: str


@dataclass(frozen=True)
class GoldConcept:
    """A concept the benchmark expects a successful pipeline to identify."""

    name: str
    aliases: tuple[str, ...] = ()
    required_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationCase:
    """A single fixed input plus its human-authored evaluation rubric."""

    id: str
    description: str
    chunks: tuple[EvaluationChunk, ...]
    expected_concepts: tuple[GoldConcept, ...]
    forbidden_concepts: tuple[str, ...] = ()
    noise_only: bool = False


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _require_string(value: object, field: str, case_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Fixture case '{case_id}' requires a non-empty {field}")
    return value.strip()


def load_fixture_corpus(version: str = "v1") -> tuple[EvaluationCase, ...]:
    """Load a versioned corpus without involving a PDF parser or LLM backend."""
    corpus_path = _REPOSITORY_ROOT / "evaluations" / "fixtures" / version / "corpus.json"
    with corpus_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    cases_raw = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError(f"Fixture corpus '{version}' must contain a non-empty cases list")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for raw_case in cases_raw:
        if not isinstance(raw_case, dict):
            raise ValueError("Each fixture case must be an object")
        case_id = _require_string(raw_case.get("id"), "id", "<unknown>")
        if case_id in seen_ids:
            raise ValueError(f"Fixture corpus '{version}' repeats case id '{case_id}'")
        seen_ids.add(case_id)

        chunks_raw = raw_case.get("chunks")
        if not isinstance(chunks_raw, list) or not chunks_raw:
            raise ValueError(f"Fixture case '{case_id}' must contain chunks")
        chunks = tuple(
            EvaluationChunk(
                id=_require_string(chunk.get("id"), "chunk id", case_id),
                text=_require_string(chunk.get("text"), "chunk text", case_id),
                source=_require_string(chunk.get("source"), "chunk source", case_id),
                location_label=_require_string(chunk.get("location_label"), "chunk location_label", case_id),
            )
            for chunk in chunks_raw
            if isinstance(chunk, dict)
        )
        if len(chunks) != len(chunks_raw):
            raise ValueError(f"Fixture case '{case_id}' contains a non-object chunk")

        concepts_raw = raw_case.get("expected_concepts", [])
        if not isinstance(concepts_raw, list):
            raise ValueError(f"Fixture case '{case_id}' has invalid expected_concepts")
        concepts = tuple(
            GoldConcept(
                name=_require_string(concept.get("name"), "concept name", case_id),
                aliases=tuple(str(alias).strip() for alias in concept.get("aliases", []) if str(alias).strip()),
                required_terms=tuple(
                    str(term).strip().lower()
                    for term in concept.get("required_terms", [])
                    if str(term).strip()
                ),
            )
            for concept in concepts_raw
            if isinstance(concept, dict)
        )
        if len(concepts) != len(concepts_raw):
            raise ValueError(f"Fixture case '{case_id}' contains a non-object expected concept")

        forbidden_raw = raw_case.get("forbidden_concepts", [])
        if not isinstance(forbidden_raw, list) or not all(isinstance(item, str) for item in forbidden_raw):
            raise ValueError(f"Fixture case '{case_id}' has invalid forbidden_concepts")
        noise_only = raw_case.get("noise_only", False)
        if not isinstance(noise_only, bool):
            raise ValueError(f"Fixture case '{case_id}' has invalid noise_only")
        if noise_only and concepts:
            raise ValueError(f"Fixture case '{case_id}' cannot have gold concepts when noise_only")

        cases.append(
            EvaluationCase(
                id=case_id,
                description=_require_string(raw_case.get("description"), "description", case_id),
                chunks=chunks,
                expected_concepts=concepts,
                forbidden_concepts=tuple(item.strip() for item in forbidden_raw if item.strip()),
                noise_only=noise_only,
            )
        )

    return tuple(cases)
