"""The :class:`Result` record.

This is a thin, dict-backed wrapper around the schema defined in the Rust core
(:mod:`caliper._core`). All parsing, normalisation, and validation happen in
Rust; this class exists so Python callers have a typed handle and don't pass raw
JSON around. The mutable builder used by ``bench()`` will extend it once the
measurement pipeline lands.
"""

from __future__ import annotations

import json
from typing import Any

from caliper import _core


class Result:
    """One measurement record.

    Construct an empty record with :meth:`default`, or load one with
    :meth:`from_dict` / :meth:`from_json`. Both loaders round-trip through the
    Rust schema, so unknown keys are dropped and missing sections are filled
    with defaults.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        raw = _core.default_record_json() if data is None else json.dumps(data)
        self._data: dict[str, Any] = json.loads(_core.normalize_record_json(raw))

    @classmethod
    def default(cls) -> Result:
        """An empty record carrying the current schema and core versions."""
        return cls(None)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Result:
        """Load a record from a plain dict."""
        return cls(data)

    @classmethod
    def from_json(cls, text: str) -> Result:
        """Load a record from a JSON string."""
        return cls(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        """A deep copy of the record as a plain, JSON-compatible dict."""
        copy: dict[str, Any] = json.loads(json.dumps(self._data))
        return copy

    def to_json(self) -> str:
        """The record as canonical JSON (stable key order)."""
        return _core.normalize_record_json(json.dumps(self._data))

    def validate(self) -> list[str]:
        """Human-readable schema problems; empty if the record is well-formed."""
        return list(_core.validate_record_json(json.dumps(self._data)))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Result) and other._data == self._data

    def __hash__(self) -> int:
        return hash(self.to_json())

    def __repr__(self) -> str:
        return (
            f"Result(schema_version={self._data['schema_version']!r}, "
            f"caliper_version={self._data['caliper_version']!r})"
        )
