"""Read a ``sweep`` spec (Appendix D YAML) and expand it to a cell list.

YAML parsing is a Python concern; validation and expansion are the Rust core's
(``caliper._core.expand_spec``), so the two agree on what a spec means. PyYAML
is an optional dependency: ``pip install 'caliper-gpu[sweep]'``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from caliper import _core

__all__ = ["cell_keys", "load_cells", "parse_spec", "pending_cells"]


def _yaml_load(text: str) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - yaml is in the dev env
        raise ImportError(
            "reading a sweep spec needs PyYAML: pip install 'caliper-gpu[sweep]'"
        ) from exc
    return yaml.safe_load(text)


def _spec_to_json(spec: str | Path | dict[str, Any]) -> str:
    """Normalise a spec (a dict, a ``Path``, a filename string, or YAML text) to
    a JSON string. A one-line ``str`` that names an existing file is read as a
    path; anything else is parsed as YAML text."""
    if isinstance(spec, dict):
        return json.dumps(spec)
    if isinstance(spec, Path):
        text = spec.read_text()
    elif "\n" not in spec and Path(spec).is_file():
        text = Path(spec).read_text()
    else:
        text = spec
    parsed = _yaml_load(text)
    if not isinstance(parsed, dict):
        raise ValueError("a sweep spec must be a YAML mapping")
    return json.dumps(parsed)


def parse_spec(spec: str | Path | dict[str, Any]) -> dict[str, Any]:
    """The spec as a plain dict (``Path`` / YAML text / dict all accepted).
    Does *not* validate -- use :func:`load_cells` for that."""
    parsed: dict[str, Any] = json.loads(_spec_to_json(spec))
    return parsed


def load_cells(spec: str | Path | dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and expand a spec to its deduplicated cell list.

    ``spec`` is a dict, a ``Path`` / filename string pointing at a YAML file, or
    the YAML text itself. Raises ``ValueError`` (with a typed message) on any
    malformed field.
    """
    cells: list[dict[str, Any]] = json.loads(_core.expand_spec(_spec_to_json(spec)))
    return cells


def cell_keys(cells: list[dict[str, Any]]) -> list[str]:
    """The stable ``--resume`` key for each cell."""
    keys: list[str] = _core.spec_cell_keys(json.dumps(cells))
    return keys


def pending_cells(cells: list[dict[str, Any]], done_keys: list[str]) -> list[dict[str, Any]]:
    """The cells in ``cells`` whose key is not in ``done_keys`` (``--resume``)."""
    left: list[dict[str, Any]] = json.loads(
        _core.spec_pending(json.dumps(cells), json.dumps(done_keys))
    )
    return left
