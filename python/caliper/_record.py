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


class _AttrView(dict[str, Any]):
    """A dict whose keys are also reachable as attributes (``section.p50_us``)."""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


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

    # -- read accessors (the frozen public surface) -----------------------

    def _section(self, name: str) -> _AttrView:
        return _AttrView(json.loads(json.dumps(self._data[name])))

    @property
    def schema_version(self) -> str:
        return str(self._data["schema_version"])

    @property
    def caliper_version(self) -> str:
        return str(self._data["caliper_version"])

    @property
    def measured_at(self) -> str | None:
        value = self._data["measured_at"]
        return None if value is None else str(value)

    @property
    def flags(self) -> list[str]:
        return list(self._data["flags"])

    @property
    def throttle_reasons(self) -> list[str]:
        return list(self._data["throttle_reasons"])

    @property
    def timing(self) -> _AttrView:
        return self._section("timing")

    @property
    def roofline(self) -> _AttrView:
        return self._section("roofline")

    @property
    def ptxas(self) -> _AttrView:
        return self._section("ptxas")

    @property
    def occupancy(self) -> _AttrView:
        return self._section("occupancy")

    @property
    def clocks(self) -> _AttrView:
        return self._section("clocks")

    @property
    def machine(self) -> _AttrView:
        return self._section("machine")

    @property
    def kernel(self) -> _AttrView:
        return self._section("kernel")

    @property
    def p10_us(self) -> float | None:
        value = self._data["timing"]["p10_us"]
        return None if value is None else float(value)

    @property
    def p50_us(self) -> float | None:
        value = self._data["timing"]["p50_us"]
        return None if value is None else float(value)

    @property
    def p90_us(self) -> float | None:
        value = self._data["timing"]["p90_us"]
        return None if value is None else float(value)

    @property
    def mean_us(self) -> float | None:
        value = self._data["timing"]["mean_us"]
        return None if value is None else float(value)

    @property
    def min_us(self) -> float | None:
        value = self._data["timing"]["min_us"]
        return None if value is None else float(value)

    @property
    def max_us(self) -> float | None:
        value = self._data["timing"]["max_us"]
        return None if value is None else float(value)

    @property
    def mad_us(self) -> float | None:
        value = self._data["timing"]["mad_us"]
        return None if value is None else float(value)

    @property
    def wall_p50_us(self) -> float | None:
        value = self._data["timing"]["wall_p50_us"]
        return None if value is None else float(value)

    @property
    def launch_overhead_us(self) -> float | None:
        value = self._data["timing"]["launch_overhead_us"]
        return None if value is None else float(value)

    @property
    def achieved_tflops(self) -> float | None:
        value = self._data["roofline"]["achieved_tflops"]
        return None if value is None else float(value)

    @property
    def roofline_pct(self) -> float | None:
        value = self._data["roofline"]["roofline_pct"]
        return None if value is None else float(value)

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
