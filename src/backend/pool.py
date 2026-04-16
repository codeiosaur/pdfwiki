"""
BackendPool — distributes LLM calls across multiple backends for a single pass.

For generate() calls: batches are dispatched using weighted rotation. Each
backend acquires a semaphore slot before processing, enforcing its per-backend
concurrency limit. If the chosen backend fails, fallback happens transparently.

For dispatch() calls (batch processing): all items go into a shared queue.
Each backend contributes N workers (from its ``workers`` config). Workers
pull batches from the queue in a work-stealing pattern until exhausted,
ensuring load balancing: faster backends process more items.

Usage:
    pool = BackendPool([local_backend, remote_backend], label="pass2-assign",
                       workers=[4, 2])
    result = pool.generate(prompt, json_schema=schema)  # weighted rotation
    results = pool.dispatch(items, process_fn, batch_size)  # work-stealing
"""

import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Tuple, TypeVar

from backend.base import LLMBackend, BackendConfig, RetryableError

_T = TypeVar("_T")


class BackendPool(LLMBackend):
    """
    Wraps multiple LLMBackend instances and exposes both single-request and
    batch-processing interfaces.

    For generate() calls: weighted rotation dispatch with per-backend semaphore
    slots. Each backend appears in the dispatch order once per unit of its
    ``workers`` weight. Fallback happens transparently if the chosen backend fails.

    For dispatch() calls (batch processing): work-stealing queue. All items go
    into a shared queue. Each backend contributes N workers (from its ``workers``
    config). Workers pull batches from the queue until exhausted, ensuring
    dynamic load balancing based on actual processing speed, not pre-allocation.

    Thread-local tracking: last_used_label() returns the label of the backend
    that handled the most recent generate() or dispatch() call on the calling
    thread, allowing callers to log which backend produced a given result.
    """

    def __init__(
        self,
        backends: list[LLMBackend],
        label: str = "pool",
        weights: Optional[list[int]] = None,
    ) -> None:
        if not backends:
            raise ValueError("BackendPool requires at least one backend")
        # BackendPool does not use a BackendConfig itself — delegate to members.
        # We satisfy the parent __init__ by passing the first member's config.
        super().__init__(backends[0]._config)
        self._backends = backends
        self._label = label

        # Build weighted dispatch order: each backend appears once per weight unit.
        if weights is None:
            weights = [1] * len(backends)
        if len(weights) != len(backends):
            raise ValueError("weights length must match backends length")
        self._weights = weights
        self._dispatch_order: list[LLMBackend] = [
            b for b, w in zip(backends, weights) for _ in range(max(1, w))
        ]
        self._counter = itertools.count()

        # Per-backend semaphores: each backend processes at most workers concurrent requests.
        self._semaphores = [threading.Semaphore(max(1, w)) for w in weights]
        self._total_workers = sum(max(1, w) for w in weights)

        # Per-thread tracking of which backend was last used.
        self._thread_local = threading.local()

    @property
    def label(self) -> str:
        return self._label

    @property
    def total_workers(self) -> int:
        """Total concurrent-request capacity across all member backends."""
        return self._total_workers

    @property
    def model(self) -> str:
        # Return a summary of all models for logging purposes.
        models = [b.model for b in self._backends]
        return ", ".join(models)

    def last_used_label(self) -> str:
        """
        Return the label of the backend that handled the most recent generate()
        call on the calling thread.  Returns the pool label if no call has been
        made yet on this thread.
        """
        return getattr(self._thread_local, "last_backend_label", self._label)

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        json_schema: Optional[dict] = None,
        context: str = "",
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Route the call to the next backend in weighted-rotation order.

        If the chosen backend fails (after its own internal retries), the call
        falls back through the remaining unique backends in pool order before
        raising.  This means a rate-limited or unavailable backend is bypassed
        transparently rather than failing the whole request.
        """
        idx = next(self._counter) % len(self._dispatch_order)
        primary = self._dispatch_order[idx]
        primary_idx = self._backends.index(primary)

        # Try primary, then fall back through remaining unique backends.
        # Each attempt acquires the target backend's semaphore (blocking) to
        # enforce its per-backend slot limit, and releases it when done.
        others = [(i, b) for i, b in enumerate(self._backends) if b is not primary]
        backends_to_try = [(primary_idx, primary)] + others

        last_exc: Optional[Exception] = None
        for i, (b_idx, backend) in enumerate(backends_to_try):
            self._semaphores[b_idx].acquire()
            try:
                self._thread_local.last_backend_label = backend.label
                return backend.generate(
                    prompt,
                    max_tokens=max_tokens,
                    json_schema=json_schema,
                    context=context,
                    system_prompt=system_prompt,
                )
            except Exception as exc:
                last_exc = exc
                if i < len(backends_to_try) - 1:
                    print(f"  [{self._label}] {backend.label} failed ({exc}), "
                          f"trying {backends_to_try[i + 1][1].label}...")
            finally:
                self._semaphores[b_idx].release()

        raise last_exc  # type: ignore[misc]

    def metrics(self) -> dict:
        """Aggregate metrics across all member backends."""
        combined: dict = {}
        for b in self._backends:
            for k, v in b.metrics().items():
                combined[k] = combined.get(k, 0) + v
        return combined

    def member_labels(self) -> list[str]:
        """Return labels of all member backends (for logging)."""
        return [b.label for b in self._backends]

    def dispatch(
        self,
        items: List[_T],
        process_fn: Callable[[List[_T], LLMBackend, int], List],
        default_batch_size: int,
    ) -> List:
        """
        Distribute items across backend workers using work-stealing.

        All items go into a shared queue. Each backend contributes N workers
        (from its ``workers`` config). Workers pull batches from the queue until
        exhausted, with no per-backend concurrency limits—allowing faster
        backends to naturally process more items.

        This ensures true load balancing: faster backends process more items
        due to pulling more frequently, while ThreadPoolExecutor enforces
        overall concurrency (max_workers = sum of all backend workers).

        Results are collected and returned in any order (insertion order is lost).

        Args:
            items:             Full list of items to process (e.g. statements).
            process_fn:        Callable(items_slice, backend, batch_size) -> list.
                               Called by each worker thread with its batch.
            default_batch_size: Fallback batch size for backends without
                               ``preferred_batch_size`` set.

        Returns:
            Flat list of all results from all workers, in any order.
        """
        if not items:
            return []

        from collections import deque

        # Shared state: queue of items, retry queue, results list, locks
        item_queue: deque = deque(items)
        retry_queue: List[Tuple[List, float]] = []  # (batch, retry_until_time)
        queue_lock = threading.Lock()
        results: List = []
        results_lock = threading.Lock()

        def _worker(backend: LLMBackend, backend_idx: int) -> None:
            """
            Worker loop: pull batch from queue or retry_queue, process, repeat.
            If RetryableError is raised, re-queue with a delay instead of blocking.
            No per-backend semaphore; ThreadPoolExecutor enforces total concurrency.
            """
            bs = backend.preferred_batch_size or default_batch_size
            while True:
                batch = None
                # Try to find a batch: either from retry_queue (if ready) or item_queue
                with queue_lock:
                    now = time.time()
                    # Check if any retryable items are ready
                    ready_retries = [i for i, (_, retry_time) in enumerate(retry_queue) if retry_time <= now]
                    if ready_retries:
                        # Take first ready retry
                        batch, _ = retry_queue.pop(ready_retries[0])
                    elif item_queue:
                        # Pull up to batch_size items from the main queue
                        batch = [item_queue.popleft() for _ in range(min(bs, len(item_queue)))]
                        if not batch:
                            return

                    # If no batch ready and nothing in queues, exit
                    if batch is None and not item_queue and not retry_queue:
                        return

                # If no batch found but there are retries pending, sleep briefly and retry
                if batch is None:
                    time.sleep(0.1)
                    continue

                # Process the batch without holding locks (true work-stealing)
                try:
                    self._thread_local.last_backend_label = backend.label
                    result = process_fn(batch, backend, bs)

                    # Append results (thread-safe)
                    with results_lock:
                        results.extend(result)
                except RetryableError as e:
                    # Re-queue batch with retry delay instead of blocking
                    retry_time = time.time() + e.delay_seconds
                    with queue_lock:
                        retry_queue.append((batch, retry_time))

        # Spawn workers: each backend gets N worker threads (no per-backend concurrency limit).
        # ThreadPoolExecutor enforces max_workers = sum of all backend workers.
        with ThreadPoolExecutor(max_workers=self._total_workers) as executor:
            futures = []
            for backend_idx, (backend, num_workers) in enumerate(
                zip(self._backends, self._weights)
            ):
                # Each backend contributes num_workers worker threads
                for _ in range(max(1, num_workers)):
                    f = executor.submit(_worker, backend, backend_idx)
                    futures.append(f)

            # Wait for all workers to complete; any exception is re-raised
            for future in as_completed(futures):
                future.result()

        return results
