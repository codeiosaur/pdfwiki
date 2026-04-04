"""
BackendPool — distributes LLM calls across multiple backends for a single pass.

When multiple backends are assigned to a pass, batches are dispatched round-robin
across all of them.  Combined with ThreadPoolExecutor, faster backends naturally
absorb more work since they free their slots sooner.

Usage:
    pool = BackendPool([local_backend, remote_backend], label="pass2-assign")
    result = pool.generate(prompt, json_schema=schema)
"""

import itertools
from typing import Optional

from backend.base import LLMBackend, BackendConfig


class BackendPool(LLMBackend):
    """
    Wraps multiple LLMBackend instances and exposes the same generate() interface.

    Batches are dispatched round-robin across member backends.  The pool is a
    drop-in replacement for any single LLMBackend — callers (pipeline, fact
    extractor, synthesizer) require no changes.
    """

    def __init__(self, backends: list[LLMBackend], label: str = "pool") -> None:
        if not backends:
            raise ValueError("BackendPool requires at least one backend")
        # BackendPool does not use a BackendConfig itself — delegate to members.
        # We satisfy the parent __init__ by passing the first member's config.
        super().__init__(backends[0]._config)
        self._backends = backends
        self._label = label
        self._counter = itertools.count()

    @property
    def label(self) -> str:
        return self._label

    @property
    def model(self) -> str:
        # Return a summary of all models for logging purposes.
        models = [b.model for b in self._backends]
        return ", ".join(models)

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        json_schema: Optional[dict] = None,
        context: str = "",
    ) -> str:
        """Route the call to the next backend in round-robin order."""
        idx = next(self._counter) % len(self._backends)
        return self._backends[idx].generate(
            prompt,
            max_tokens=max_tokens,
            json_schema=json_schema,
            context=context,
        )

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
