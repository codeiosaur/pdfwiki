from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional
import logging
import os
import random
import sys
import threading
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
    pass2_start_offset_s: Optional[float] = None   # seconds into P1 when P2 first dispatched

    def print_summary(self) -> None:
        print("\n=== PIPELINE METRICS ===")
        rows: list = [
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
        if self.pass2_start_offset_s is not None:
            rows.append(("p2_start_offset_s", f"{self.pass2_start_offset_s:.0f}"))
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
    pass1_max_workers: Optional[int] = None,
    pass2_max_workers: Optional[int] = None,
) -> List[Fact]:
    _p1_batch = pass1_batch_size if pass1_batch_size is not None else batch_size
    _p2_batch = pass2_batch_size if pass2_batch_size is not None else batch_size
    _p1_workers = pass1_max_workers if pass1_max_workers is not None else max_workers
    _p2_workers = pass2_max_workers if pass2_max_workers is not None else max_workers

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

    def _p1_label() -> str:
        return pass1_backend.last_used_label() if hasattr(pass1_backend, "last_used_label") else pass1_backend.label

    def _p2_label() -> str:
        return pass2_backend.last_used_label() if hasattr(pass2_backend, "last_used_label") else pass2_backend.label

    from backend.pool import BackendPool as _BackendPool
    if isinstance(pass1_backend, _BackendPool):
        # Pool dispatch: each backend gets a proportional slice of chunks at its
        # own preferred batch size, all running concurrently.
        all_statements = pass1_backend.dispatch(
            chunks,
            lambda subset, backend, bs: extract_raw_statements_batched(subset, backend, bs),
            _p1_batch,
        )
        used_labels = ", ".join(pass1_backend.member_labels())
        print(
            f"  Pass 1 dispatch: {len(chunks)} chunks → "
            f"{len(pass1_backend._backends)} backends [{used_labels}]"
        )
    else:
        with ThreadPoolExecutor(max_workers=_p1_workers) as executor:
            futures = []
            for index, batch in enumerate(chunk_batches):
                if pace_batches and index > 0:
                    time.sleep(random.uniform(1.0, 3.0))
                def _p1_task(b=batch):
                    stmts = extract_raw_statements_batched(b, pass1_backend, _p1_batch)
                    return stmts, _p1_label()
                f = executor.submit(_p1_task)
                futures.append(f)
            last_completion = t0
            for future in as_completed(futures):
                try:
                    batch_statements, backend_label = future.result()
                    all_statements.extend(batch_statements)
                    completed_batches += 1
                    now = time.perf_counter()
                    batch_elapsed = now - last_completion
                    pass_elapsed = now - t0
                    last_completion = now
                    print(
                        f"  Pass 1 progress: {completed_batches}/{total_batches} batches complete "
                        f"({len(all_statements)} raw statements, {batch_elapsed:.0f}s/batch, "
                        f"{pass_elapsed:.0f}s total) [{backend_label}]"
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
    all_facts: List[Fact] = []

    from backend.pool import BackendPool as _BackendPool
    if isinstance(pass2_backend, _BackendPool):
        # Pool dispatch: each backend gets a proportional slice of all statements
        # and processes it at its own preferred batch size concurrently.
        def _p2_process(subset, backend, batch_size):
            return assign_concepts_to_statements(subset, backend, seeds, batch_size, use_strict)

        all_facts = pass2_backend.dispatch(all_statements, _p2_process, _p2_batch)
        used_labels = ", ".join(pass2_backend.member_labels())
        print(
            f"  Pass 2 dispatch: {len(all_statements)} statements → "
            f"{len(pass2_backend._backends)} backends [{used_labels}]"
        )
    else:
        chunk_size = max(_p2_batch * 4, 32)
        statement_chunks = [
            all_statements[i:i + chunk_size]
            for i in range(0, len(all_statements), chunk_size)
        ]
        total_statement_batches = len(statement_chunks)
        completed_statement_batches = 0
        with ThreadPoolExecutor(max_workers=_p2_workers) as executor:
            futures = []
            for chunk in statement_chunks:
                def _p2_task(c=chunk):
                    facts = assign_concepts_to_statements(c, pass2_backend, seeds, _p2_batch, use_strict)
                    return facts, _p2_label()
                f = executor.submit(_p2_task)
                futures.append(f)
            last_completion = t0
            for future in as_completed(futures):
                try:
                    batch_facts, backend_label = future.result()
                    all_facts.extend(batch_facts)
                    completed_statement_batches += 1
                    now = time.perf_counter()
                    batch_elapsed = now - last_completion
                    pass_elapsed = now - t0
                    last_completion = now
                    print(
                        f"  Pass 2 progress: {completed_statement_batches}/{total_statement_batches} batches complete "
                        f"({len(all_facts)} facts, {batch_elapsed:.0f}s/batch, "
                        f"{pass_elapsed:.0f}s total) [{backend_label}]"
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


def run_pipeline_streaming(
    pdf_path: str,
    pass1_backend: LLMBackend,
    pass2_backend: LLMBackend,
    batch_size: int = 2,
    max_workers: int = 5,
    max_chunks: Optional[int] = None,
    seeds_file: Optional[str] = None,
    pass1_batch_size: Optional[int] = None,
    pass2_batch_size: Optional[int] = None,
    pass1_max_workers: Optional[int] = None,
    pass2_max_workers: Optional[int] = None,
) -> List[Fact]:
    """
    Streaming variant of run_pipeline_two_pass.

    Pass 1 batches are enqueued as they complete and consumed by Pass 2 in
    real time, with no hard barrier between the two passes:

    - seeds_file provided: seeds are loaded before Pass 1 starts; Pass 2
      consumer is live from the very first batch.
    - seeds derived from statements: Pass 2 is unblocked once ~40% of Pass 1
      batches have completed, then immediately processes the backlog and all
      subsequent batches as they arrive.
    """
    _p1_batch = pass1_batch_size if pass1_batch_size is not None else batch_size
    _p2_batch = pass2_batch_size if pass2_batch_size is not None else batch_size
    _p1_workers = pass1_max_workers if pass1_max_workers is not None else max_workers
    _p2_workers = pass2_max_workers if pass2_max_workers is not None else max_workers

    chunks = load_pdf_chunks(pdf_path=pdf_path)
    if isinstance(max_chunks, int) and max_chunks > 0:
        chunks = chunks[:max_chunks]
    chunk_batches = list(generate_chunk_batches(chunks, batch_size=_p1_batch))
    total_batches = len(chunk_batches)

    seeds_ready = threading.Event()
    seeds_holder: List = [None]   # written before seeds_ready is set; read after
    all_statements: List[dict] = []
    all_facts: List[Fact] = []

    # ── Seed resolution ───────────────────────────────────────────────────
    # When a seeds file is provided, load it now so Pass 2 can start with
    # batch 1.  Otherwise, derive seeds at ~40% of Pass 1 batches.
    if seeds_file:
        seeds = resolve_seed_concepts([], pass2_backend, seeds_file=seeds_file)
        seeds_holder[0] = seeds
        seeds_ready.set()
        seed_threshold: Optional[int] = None
    else:
        seed_threshold = max(1, int(total_batches * 0.4))

    # ── Shared statement buffer ───────────────────────────────────────────
    # Pass 1 appends to this deque; Pass 2 workers pull from it at their own
    # preferred batch size.  p1_done signals that no more statements will arrive.
    _stmt_buf: deque = deque()
    _stmt_lock = threading.Lock()
    _p1_done = threading.Event()
    p2_first_submit_time: List[Optional[float]] = [None]

    def _s_p2_label() -> str:
        return pass2_backend.last_used_label() if hasattr(pass2_backend, "last_used_label") else pass2_backend.label

    def _pull(n: int) -> Optional[List[dict]]:
        """Pull up to n statements from the shared buffer.
        Returns None when the buffer is empty AND Pass 1 is finished."""
        while True:
            with _stmt_lock:
                if _stmt_buf:
                    batch = [_stmt_buf.popleft() for _ in range(min(n, len(_stmt_buf)))]
                    return batch
            if _p1_done.is_set():
                return None   # truly empty and no more coming
            time.sleep(0.05)  # brief spin-wait for next P1 result

    def _pass2_consumer() -> None:
        seeds_ready.wait()
        seeds = seeds_holder[0]
        use_strict = getattr(pass2_backend, "is_openrouter", False) and bool(seeds)
        if use_strict:
            print("Pass 2 [streaming]: strict seed enforcement enabled")

        from backend.pool import BackendPool as _BackendPool
        if isinstance(pass2_backend, _BackendPool):
            # Each backend worker loops independently, pulling its preferred
            # batch size from the shared buffer until the buffer is exhausted.
            p2_facts_lock = threading.Lock()
            p2_batch_num_lock = threading.Lock()
            p2_batch_num = [0]

            _PERMANENT_QUOTA_MARKERS = (
                "daily free allocation", "neurons", "daily", "upgrade",
            )
            _MAX_CONSECUTIVE_FAILURES = 3

            def _worker(backend, bs):
                from backend.base import LLMBackendError as _LLMBackendError
                consecutive_failures = 0
                while True:
                    stmts = _pull(bs)
                    if stmts is None:
                        return
                    with p2_batch_num_lock:
                        p2_batch_num[0] += 1
                        bnum = p2_batch_num[0]
                    if p2_first_submit_time[0] is None:
                        p2_first_submit_time[0] = time.perf_counter()
                    try:
                        facts = assign_concepts_to_statements(
                            stmts, backend, seeds, bs, use_strict,
                            raise_on_json_failure=True,
                        )
                    except _LLMBackendError as exc:
                        exc_str = str(exc).lower()
                        is_permanent = any(m in exc_str for m in _PERMANENT_QUOTA_MARKERS)
                        consecutive_failures += 1
                        too_many = consecutive_failures >= _MAX_CONSECUTIVE_FAILURES
                        if is_permanent or too_many:
                            reason = "daily quota exhausted" if is_permanent else f"{consecutive_failures} consecutive failures"
                            print(
                                f"  Pass 2 [streaming]: [{backend.label}] retiring worker ({reason}), "
                                f"re-queuing {len(stmts)} statements for other backends"
                            )
                            with _stmt_lock:
                                _stmt_buf.extend(stmts)
                            return
                        print(
                            f"  Pass 2 [streaming]: [{backend.label}] batch {bnum} "
                            f"failed, re-queuing {len(stmts)} statements"
                        )
                        with _stmt_lock:
                            _stmt_buf.extend(stmts)
                        continue
                    consecutive_failures = 0
                    with p2_facts_lock:
                        all_facts.extend(facts)
                    print(
                        f"  Pass 2 [streaming]: [{backend.label}] batch {bnum} done "
                        f"({len(facts)} facts, {len(all_facts)} total)"
                    )

            worker_threads = []
            for backend, w in zip(pass2_backend._backends, pass2_backend._weights):
                bs = backend.preferred_batch_size or _p2_batch
                for _ in range(max(1, w)):
                    t = threading.Thread(target=_worker, args=(backend, bs), daemon=True)
                    worker_threads.append(t)
                    t.start()
            for t in worker_threads:
                t.join()
        else:
            # Single backend: pull at _p2_batch size, submit to executor.
            p2_futures = []
            p2_batch_num = 0

            def _dispatch(statements: List[dict], executor: ThreadPoolExecutor) -> None:
                nonlocal p2_batch_num
                if not statements:
                    return
                p2_batch_num += 1
                if p2_first_submit_time[0] is None:
                    p2_first_submit_time[0] = time.perf_counter()
                def _p2_task(s=statements):
                    facts = assign_concepts_to_statements(s, pass2_backend, seeds, _p2_batch, use_strict)
                    return facts, _s_p2_label()
                p2_futures.append(executor.submit(_p2_task))

            with ThreadPoolExecutor(max_workers=_p2_workers) as executor:
                while not _p1_done.is_set() or _stmt_buf:
                    batch = _pull(_p2_batch)
                    if batch:
                        _dispatch(batch, executor)
                    else:
                        break

                completed_p2 = 0
                p2_t0 = p2_first_submit_time[0] or time.perf_counter()
                last_completion = p2_t0
                for future in as_completed(p2_futures):
                    try:
                        results, backend_label = future.result()
                        all_facts.extend(results)
                        completed_p2 += 1
                        now = time.perf_counter()
                        batch_elapsed = now - last_completion
                        pass_elapsed = now - p2_t0
                        last_completion = now
                        print(
                            f"  Pass 2 [streaming]: batch {completed_p2}/{p2_batch_num} complete "
                            f"({len(results)} facts, {len(all_facts)} total, "
                            f"{batch_elapsed:.0f}s/batch, {pass_elapsed:.0f}s total) [{backend_label}]"
                        )
                    except Exception as exc:
                        logging.warning("Streaming Pass 2 batch failed: %s", exc)

    consumer_thread = threading.Thread(target=_pass2_consumer, daemon=True)
    consumer_thread.start()

    # ── Pass 1 producer ───────────────────────────────────────────────────
    print(
        f"Pass 1 [streaming]: Extracting raw statements from {len(chunks)} chunks "
        f"[{pass1_backend.label}:{pass1_backend.model}]..."
    )
    t0 = time.perf_counter()
    completed_batches = 0
    seeds_derived = seeds_ready.is_set()   # True already when seeds_file was used
    pace_batches = _should_pace_batches(pass1_backend)
    def _s_p1_label() -> str:
        return pass1_backend.last_used_label() if hasattr(pass1_backend, "last_used_label") else pass1_backend.label

    with ThreadPoolExecutor(max_workers=_p1_workers) as executor:
        futures = []
        for index, batch in enumerate(chunk_batches):
            if pace_batches and index > 0:
                time.sleep(random.uniform(1.0, 3.0))
            def _s_p1_task(b=batch):
                stmts = extract_raw_statements_batched(b, pass1_backend, _p1_batch)
                return stmts, _s_p1_label()
            f = executor.submit(_s_p1_task)
            futures.append(f)
        last_completion = t0
        for future in as_completed(futures):
            try:
                batch_statements, backend_label = future.result()
                all_statements.extend(batch_statements)
                with _stmt_lock:
                    _stmt_buf.extend(batch_statements)
                completed_batches += 1
                now = time.perf_counter()
                batch_elapsed = now - last_completion
                pass_elapsed = now - t0
                last_completion = now
                print(
                    f"  Pass 1 progress: {completed_batches}/{total_batches} batches complete "
                    f"({len(all_statements)} raw statements, {batch_elapsed:.0f}s/batch, "
                    f"{pass_elapsed:.0f}s total) [{backend_label}]"
                )
            except Exception as exc:
                logging.warning("Streaming Pass 1 batch failed: %s", exc)
                continue

            # Mid-stream seed derivation: unblock consumer once threshold reached
            if not seeds_derived and seed_threshold and completed_batches >= seed_threshold:
                print(
                    f"  Pass 1.5 [streaming]: deriving seeds at {completed_batches}/{total_batches} batches..."
                )
                seeds = resolve_seed_concepts(all_statements, pass2_backend, seeds_file=None)
                seeds_holder[0] = seeds
                seeds_ready.set()
                seeds_derived = True

    p1_elapsed = time.perf_counter() - t0
    print(f"Pass 1 [streaming] complete: {len(all_statements)} raw statements ({p1_elapsed:.0f}s)")

    # Safety: if threshold was never reached (e.g. all batches failed), unblock now
    if not seeds_ready.is_set():
        seeds = resolve_seed_concepts(all_statements, pass2_backend, seeds_file=None)
        seeds_holder[0] = seeds
        seeds_ready.set()

    _p1_done.set()  # signal: no more statements will arrive

    # ── Wait for Pass 2 to drain ──────────────────────────────────────────
    consumer_thread.join()
    p2_end = time.perf_counter()
    p2_elapsed = (p2_end - p2_first_submit_time[0]) if p2_first_submit_time[0] else 0.0
    # How many seconds into Pass 1 did Pass 2 first dispatch?
    p2_start_offset_s = (p2_first_submit_time[0] - t0) if p2_first_submit_time[0] else p1_elapsed
    print(f"Pass 2 [streaming] complete: {len(all_facts)} facts ({p2_elapsed:.0f}s)")

    use_strict = getattr(pass2_backend, "is_openrouter", False) and bool(seeds)
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
        pass2_start_offset_s=p2_start_offset_s,
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
