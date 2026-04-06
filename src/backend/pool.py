"""
BackendPool — distributes LLM calls across multiple backends for a single pass.

When multiple backends are assigned to a pass, batches are dispatched using a
weighted rotation across all of them.  Each backend appears in the dispatch
order proportional to its ``workers`` weight, so a backend with weight 6
handles 60% of requests when paired with a weight-4 backend.

Usage:
    pool = BackendPool([local_backend, remote_backend], label="pass2-assign",
                       weights=[4, 2])
    result = pool.generate(prompt, json_schema=schema)
"""

import itertools
import threading
from typing import Optional

from backend.base import LLMBackend, BackendConfig


class BackendPool(LLMBackend):
    """
    Wraps multiple LLMBackend instances and exposes the same generate() interface.

    Requests are dispatched using a weighted rotation: each backend appears in
    the dispatch order once per unit of its weight.  Unweighted pools (all
    weights=1) behave identically to the previous round-robin.  The pool is a
    drop-in replacement for any single LLMBackend.

    Thread-local tracking: last_used_label() returns the label of the backend
    that handled the most recent generate() call on the calling thread, allowing
    callers to log which backend produced a given result.
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
                    print(f"  [{self._label}] {backend.label} failed, "
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
