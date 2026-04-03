from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional
import logging
import os
import random
import sys
import time

from backend import LLMBackend
from extract.fact_extractor import (
    Fact,
    _parse_json_object,
    assign_concepts_to_statements,
    derive_seed_concepts,
    extract_facts_batched,
    extract_raw_statements_batched,
    load_builtin_seeds,
    load_seeds_from_file,
)
from generate.titles import concept_tokens
from ingest.pdf_loader import Chunk, load_pdf_chunks
from transform.matching import is_sibling, has_antonym_conflict

_INVALID_FILENAME_CHARS = str.maketrans({c: "" for c in r'\/:*?"<>|'})


@dataclass
class PipelineMetrics:
    """Per-run observability summary collected at the end of run_pipeline_two_pass."""
    total_chunks: int = 0
    total_statements: int = 0
    total_facts: int = 0
    pass1_time_s: float = field(default=0.0)
    pass2_time_s: float = field(default=0.0)
    pass1_retries: int = 0
    pass1_fallback_hops: int = 0
    pass2_retries: int = 0
    pass2_fallback_hops: int = 0

    def print_summary(self) -> None:
        print("\n=== PIPELINE METRICS ===")
        rows = [
            ("chunks", self.total_chunks),
            ("statements", self.total_statements),
            ("facts", self.total_facts),
            ("pass1_time_s", f"{self.pass1_time_s:.0f}"),
            ("pass2_time_s", f"{self.pass2_time_s:.0f}"),
            ("pass1_retries", self.pass1_retries),
            ("pass1_fallback_hops", self.pass1_fallback_hops),
            ("pass2_retries", self.pass2_retries),
            ("pass2_fallback_hops", self.pass2_fallback_hops),
        ]
        for label, value in rows:
            print(f"  {label:<22}: {value}")


def validate_pipeline_inputs(pdf_path: str, output_dir: Path, seeds_file: Optional[str] = None) -> None:
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"Error: PDF file not found: {pdf_file}")
        sys.exit(1)
    if not pdf_file.is_file():
        print(f"Error: PDF path is not a file: {pdf_file}")
        sys.exit(1)
    if not os.access(pdf_file, os.R_OK):
        print(f"Error: PDF file is not readable: {pdf_file}")
        sys.exit(1)

    if seeds_file:
        seeds_path = Path(seeds_file)
        if not seeds_path.exists():
            print(f"Error: Seeds file not found: {seeds_path}")
            sys.exit(1)
        if not seeds_path.is_file():
            print(f"Error: Seeds path is not a file: {seeds_path}")
            sys.exit(1)
        if not os.access(seeds_path, os.R_OK):
            print(f"Error: Seeds file is not readable: {seeds_path}")
            sys.exit(1)

    parent_dir = output_dir if output_dir.exists() else output_dir.parent
    if parent_dir and not os.access(parent_dir, os.W_OK):
        print(f"Error: Cannot write to output directory: {output_dir}")
        print(f"Check write access to: {parent_dir}")
        sys.exit(1)


def write_vault(pages: dict[str, str], output_dir: Path) -> None:
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            print(f"Error: Cannot create output directory '{output_dir}': {exc}")
            print("Check that you have write permission to the parent directory.")
            sys.exit(1)
        except OSError as exc:
            print(f"Error: Failed to create output directory '{output_dir}': {exc}")
            sys.exit(1)
    else:
        existing = list(output_dir.iterdir())
        if existing:
            print(
                f"Warning: Output directory '{output_dir}' already exists and "
                f"contains {len(existing)} file(s). Adding pages to existing directory."
            )

    written = 0
    for title, content in pages.items():
        safe_name = title.translate(_INVALID_FILENAME_CHARS).strip() or "Unnamed Concept"
        file_path = output_dir / f"{safe_name}.md"
        try:
            file_path.write_text(content, encoding="utf-8")
            written += 1
        except PermissionError:
            print(f"Error: Cannot write '{file_path}': permission denied.")
            print("Check that you have write access to the output directory.")
            sys.exit(1)
        except OSError as exc:
            print(f"Error: Failed to write '{file_path}': {exc}")
            sys.exit(1)

    print(f"Vault written: {written} pages -> {output_dir}/")


def generate_chunk_batches(chunks: List[Chunk], batch_size: int = 2) -> Iterator[List[Chunk]]:
    if batch_size < 1:
        batch_size = 1
    for i in range(0, len(chunks), batch_size):
        yield chunks[i : i + batch_size]


def _should_pace_batches(backend: LLMBackend) -> bool:
    if backend.provider == "anthropic":
        return True
    if backend.provider != "openai_compat":
        return False
    base_url = backend.base_url.lower()
    local_hosts = ("localhost", "127.0.0.1", "::1")
    return not any(host in base_url for host in local_hosts)


def resolve_seed_concepts(
    statements: List[dict],
    pass2_backend: LLMBackend,
    seeds_file: Optional[str] = None,
) -> List[str]:
    if seeds_file:
        try:
            seeds = load_seeds_from_file(seeds_file)
            print(f"Pass 1.5: Loaded {len(seeds)} seed concepts from {seeds_file}")
            return seeds
        except Exception as exc:
            print(f"Pass 1.5: Failed to load seeds file ({exc}), falling back to auto-generation")

    print(
        f"Pass 1.5: Deriving seed concepts from statements "
        f"[{pass2_backend.label}:{pass2_backend.model}]..."
    )
    t0 = time.time()
    seeds = derive_seed_concepts(statements, pass2_backend)
    if seeds:
        print(f"Pass 1.5 complete: {len(seeds)} seed concepts derived ({time.time() - t0:.0f}s)")
        return seeds

    builtin = load_builtin_seeds()
    print(
        f"Pass 1.5: Auto-generation failed, using built-in seed list "
        f"({len(builtin)} concepts)"
    )
    return builtin


def _anchor_facts_to_seeds(facts: List[Fact], seeds: List[str]) -> List[Fact]:
    seed_set = set(seeds)
    seed_token_map = {s: concept_tokens(s) for s in seeds}

    remapped = 0
    result: List[Fact] = []
    for fact in facts:
        if fact.concept in seed_set:
            result.append(fact)
            continue

        fact_toks = concept_tokens(fact.concept)
        if not fact_toks:
            result.append(fact)
            continue

        best_seed: Optional[str] = None
        best_score = 0.0
        for seed, seed_toks in seed_token_map.items():
            if not seed_toks:
                continue
            shared = len(fact_toks & seed_toks)
            score = shared / len(fact_toks)
            if score > best_score:
                best_score = score
                best_seed = seed

        if best_seed is not None and best_score >= 0.5:
            result.append(Fact(
                id=fact.id,
                concept=best_seed,
                content=fact.content,
                source_chunk_id=fact.source_chunk_id,
            ))
            remapped += 1
        else:
            result.append(fact)

    if remapped:
        print(f"  [seed-anchor] Remapped {remapped} facts to nearest seed concept")
    return result


def run_pipeline_two_pass(
    pdf_path: str,
    pass1_backend: LLMBackend,
    pass2_backend: LLMBackend,
    batch_size: int = 2,
    max_workers: int = 5,
    max_chunks: Optional[int] = None,
    seeds_file: Optional[str] = None,
    pass1_batch_size: Optional[int] = None,
    pass2_batch_size: Optional[int] = None,
) -> List[Fact]:
    _p1_batch = pass1_batch_size if pass1_batch_size is not None else batch_size
    _p2_batch = pass2_batch_size if pass2_batch_size is not None else batch_size

    chunks = load_pdf_chunks(pdf_path=pdf_path)
    if isinstance(max_chunks, int) and max_chunks > 0:
        chunks = chunks[:max_chunks]

    chunk_batches = list(generate_chunk_batches(chunks, batch_size=_p1_batch))
    all_statements: List[dict] = []

    print(
        f"Pass 1: Extracting raw statements from {len(chunks)} chunks "
        f"[{pass1_backend.label}:{pass1_backend.model}]..."
    )
    t0 = time.perf_counter()
    total_batches = len(chunk_batches)
    completed_batches = 0
    pace_batches = _should_pace_batches(pass1_backend)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for index, batch in enumerate(chunk_batches):
            if pace_batches and index > 0:
                time.sleep(random.uniform(1.0, 3.0))
            futures.append(executor.submit(extract_raw_statements_batched, batch, pass1_backend, _p1_batch))
        for future in as_completed(futures):
            try:
                batch_statements = future.result()
                all_statements.extend(batch_statements)
                completed_batches += 1
                print(
                    f"  Pass 1 progress: {completed_batches}/{total_batches} batches complete "
                    f"({len(all_statements)} raw statements)"
                )
            except Exception as exc:
                logging.warning("Pass 1 batch failed: %s", exc)
                continue

    p1_elapsed = time.perf_counter() - t0
    print(f"Pass 1 complete: {len(all_statements)} raw statements extracted ({p1_elapsed:.0f}s)")

    seeds = resolve_seed_concepts(all_statements, pass2_backend, seeds_file=seeds_file)

    use_strict = getattr(pass2_backend, "is_openrouter", False) and bool(seeds)
    if use_strict:
        print("Pass 2: strict seed enforcement enabled (OpenRouter enum schema)")

    print(
        f"Pass 2: Assigning concept names to {len(all_statements)} statements "
        f"[{pass2_backend.label}:{pass2_backend.model}]..."
    )
    t0 = time.perf_counter()
    chunk_size = max(_p2_batch * 4, 32)
    statement_chunks = [
        all_statements[i:i + chunk_size]
        for i in range(0, len(all_statements), chunk_size)
    ]
    all_facts: List[Fact] = []
    total_statement_batches = len(statement_chunks)
    completed_statement_batches = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                assign_concepts_to_statements, chunk, pass2_backend, seeds,
                _p2_batch, use_strict,
            )
            for chunk in statement_chunks
        ]
        for future in as_completed(futures):
            try:
                all_facts.extend(future.result())
                completed_statement_batches += 1
                print(
                    f"  Pass 2 progress: {completed_statement_batches}/{total_statement_batches} batches complete "
                    f"({len(all_facts)} facts)"
                )
            except Exception as exc:
                logging.warning("Pass 2 batch failed: %s", exc)
                continue

    p2_elapsed = time.perf_counter() - t0
    print(f"Pass 2 complete: {len(all_facts)} facts with concept names ({p2_elapsed:.0f}s)")

    if not use_strict:
        all_facts = _anchor_facts_to_seeds(all_facts, seeds)

    p1_m = pass1_backend.metrics()
    p2_m = pass2_backend.metrics()
    PipelineMetrics(
        total_chunks=len(chunks),
        total_statements=len(all_statements),
        total_facts=len(all_facts),
        pass1_time_s=p1_elapsed,
        pass2_time_s=p2_elapsed,
        pass1_retries=p1_m.get("retry_count", 0),
        pass1_fallback_hops=p1_m.get("fallback_hops", 0),
        pass2_retries=p2_m.get("retry_count", 0),
        pass2_fallback_hops=p2_m.get("fallback_hops", 0),
    ).print_summary()

    return all_facts


def run_pipeline_parallel(
    pdf_path: str,
    backend: LLMBackend,
    batch_size: int = 2,
    max_workers: int = 5,
    max_chunks: Optional[int] = None,
) -> List[Fact]:
    chunks = load_pdf_chunks(pdf_path=pdf_path)
    if isinstance(max_chunks, int) and max_chunks > 0:
        chunks = chunks[:max_chunks]
    chunk_batches = list(generate_chunk_batches(chunks, batch_size=batch_size))

    all_facts: List[Fact] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(extract_facts_batched, batch, backend, batch_size)
            for batch in chunk_batches
        ]

        for future in as_completed(futures):
            try:
                batch_facts = future.result()
                all_facts.extend(batch_facts)
            except Exception:
                continue

    return all_facts
