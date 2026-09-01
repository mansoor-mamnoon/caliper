"""``Grid``: a table of :class:`~caliper.Result` rows.

A sweep produces a ``Grid``. It serialises to JSON (the nested record shape) or
to Parquet (one row per measurement, ``Result`` flattened with dotted names plus
a ``toolchain_hash`` column -- Appendix C), and filters into a smaller ``Grid``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from caliper import _core
from caliper._record import Result

if TYPE_CHECKING:
    import pyarrow

__all__ = ["Grid"]


# Free-form maps in the record: kept as one JSON-string column, not exploded
# into per-key columns (which would give rows different schemas).
_LEAF_MAPS = frozenset({"autotune_config", "shape"})


def _flatten(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested record to dotted names. Free-form maps and lists become
    a single JSON-string value so every row has the same columns."""
    out: dict[str, Any] = {}
    for key, value in obj.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and key not in _LEAF_MAPS:
            out.update(_flatten(value, f"{dotted}."))
        elif isinstance(value, (dict, list)):
            out[dotted] = json.dumps(value, separators=(",", ":"))
        else:
            out[dotted] = value
    return out


def _unflatten(row: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`_flatten`. JSON-string leaves are decoded back."""
    out: dict[str, Any] = {}
    for dotted, value in row.items():
        parts = dotted.split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        if isinstance(value, str) and value[:1] in "[{":
            with contextlib.suppress(json.JSONDecodeError):
                value = json.loads(value)
        node[parts[-1]] = value
    return out


# The canonical column order: every dotted leaf of an empty record, then the
# derived partition column.
_CANONICAL_COLUMNS: list[str] = [
    *_flatten(json.loads(_core.default_record_json())),
    "toolchain_hash",
]


def _toolchain_hash(machine: dict[str, Any]) -> str:
    toolkit = machine.get("toolkit") or {}
    material = json.dumps(sorted(toolkit.items()), separators=(",", ":"))
    material += machine.get("driver") or ""
    return "sha256:" + hashlib.sha256(material.encode()).hexdigest()


class Grid:
    """An ordered collection of measurement records."""

    __slots__ = ("_rows",)

    def __init__(self, rows: Sequence[Result | dict[str, Any]] = ()) -> None:
        self._rows: list[Result] = [r if isinstance(r, Result) else Result(dict(r)) for r in rows]

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[Result]:
        return iter(self._rows)

    def __getitem__(self, i: int) -> Result:
        return self._rows[i]

    def rows(self) -> list[Result]:
        """The records as a list of :class:`Result`."""
        return list(self._rows)

    def filter(self, predicate: Callable[[Result], bool]) -> Grid:
        """A new ``Grid`` of the rows for which ``predicate`` is true."""
        return Grid([r for r in self._rows if predicate(r)])

    # --- serialisation ----------------------------------------------------

    def to_json(self, path: str | Path | None = None, *, indent: int | None = None) -> str:
        """Serialise to a JSON array of nested records. Writes to ``path`` if
        given; always returns the text."""
        text = json.dumps([r.to_dict() for r in self._rows], indent=indent)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_json(cls, source: str | Path) -> Grid:
        """Load a ``Grid`` from a JSON array (a path or the text itself)."""
        text = Path(source).read_text() if isinstance(source, Path) else source
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("a Grid JSON document must be an array of records")
        return cls(data)

    def to_table(self) -> pyarrow.Table:
        """The flattened Parquet-shaped table (needs ``caliper[parquet]``)."""
        pa = _require_pyarrow()
        columns: dict[str, list[Any]] = {col: [] for col in _CANONICAL_COLUMNS}
        for record in self._rows:
            nested = record.to_dict()
            flat = _flatten(nested)
            flat["toolchain_hash"] = _toolchain_hash(nested.get("machine") or {})
            for col in _CANONICAL_COLUMNS:
                columns[col].append(flat.get(col))
        table: pyarrow.Table = pa.table(columns)
        return table

    def to_parquet(self, path: str | Path) -> None:
        """Write the flattened table to a Parquet file (needs
        ``caliper[parquet]``)."""
        pq = _require_pyarrow_parquet()
        pq.write_table(self.to_table(), str(path))

    @classmethod
    def from_parquet(cls, path: str | Path) -> Grid:
        """Load a ``Grid`` from a Parquet file written by :meth:`to_parquet`."""
        pq = _require_pyarrow_parquet()
        table = pq.read_table(str(path))
        rows: list[dict[str, Any]] = []
        for flat in table.to_pylist():
            flat.pop("toolchain_hash", None)
            rows.append(_unflatten({k: v for k, v in flat.items() if v is not None}))
        return cls(rows)


def _require_pyarrow() -> Any:
    try:
        import pyarrow
    except ImportError as exc:  # pragma: no cover - exercised via the extra
        raise ImportError(
            "Grid Parquet support needs pyarrow: pip install 'caliper-gpu[parquet]'"
        ) from exc
    return pyarrow


def _require_pyarrow_parquet() -> Any:
    _require_pyarrow()
    import pyarrow.parquet as pq

    return pq
