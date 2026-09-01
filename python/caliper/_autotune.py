"""A JSON-file cache of per-autotune-config timings.

A config's timing is reusable only on the same environment (SKU, driver, CUDA,
compiler, frameworks) and the same kernel source. The key is computed by the
Rust core (:func:`caliper._core.autotune_key`), so adding a config to a kernel
invalidates only that config -- a re-sweep re-times just the new one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from caliper import _core

__all__ = ["AutotuneCache"]


class AutotuneCache:
    """A ``{key: entry}`` store backed by one JSON file.

    ``entry`` is caller-defined (typically ``{"p50_us": float, "measured_at":
    str, "config": {...}}``). :meth:`hits` / :meth:`misses` count lookups since
    construction.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._store: dict[str, Any] = {}
        self._hits = 0
        self._misses = 0
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                loaded = None  # externally truncated / unreadable -- start empty
            if isinstance(loaded, dict):
                self._store = loaded

    # --- keying ---------------------------------------------------------

    @staticmethod
    def key(machine: dict[str, Any], kernel_source_hash: str, config: dict[str, Any]) -> str:
        """The cache key for a ``(machine, kernel source hash, config)`` triple."""
        return _core.autotune_key(json.dumps(machine), kernel_source_hash, json.dumps(config))

    # --- lookup / store ----------------------------------------------------

    def get(
        self, machine: dict[str, Any], kernel_source_hash: str, config: dict[str, Any]
    ) -> Any | None:
        """The cached entry for a triple, or ``None``. Counts a hit or a miss."""
        entry = self._store.get(self.key(machine, kernel_source_hash, config))
        if entry is None:
            self._misses += 1
        else:
            self._hits += 1
        return entry

    def put(
        self,
        machine: dict[str, Any],
        kernel_source_hash: str,
        config: dict[str, Any],
        entry: Any,
    ) -> None:
        """Store ``entry`` for a triple and flush the file atomically."""
        self._store[self.key(machine, kernel_source_hash, config)] = entry
        self._flush()

    def __len__(self) -> int:
        return len(self._store)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def _flush(self) -> None:
        """Write the store to a sibling temp file, then rename it into place."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self._store, indent=1, sort_keys=True))
        tmp.replace(self.path)
