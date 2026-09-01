"""Read a ``sweep`` spec (Appendix D YAML) and expand it to a cell list.

YAML parsing is a Python concern; validation and expansion are the Rust core's
(``caliper._core.expand_spec``), so the two agree on what a spec means.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from caliper import _core

__all__ = ["load_cells", "pending_cells"]


def _spec_to_json(spec: str | Path | dict[str, Any]) -> str:
    """Normalise a spec (a ``Path``, YAML text, or a dict) to a JSON string."""
    if isinstance(spec, dict):
        return json.dumps(spec)
    text = spec.read_text() if isinstance(spec, Path) else spec
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise ValueError("a sweep spec must be a YAML mapping")
    return json.dumps(parsed)


def load_cells(spec: str | Path | dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and expand a spec to its deduplicated cell list.

    ``spec`` is a ``Path`` to the YAML file, the YAML text itself, or an
    already-parsed dict. Raises ``ValueError`` (with a typed message) on any
    malformed field.
    """
    cells: list[dict[str, Any]] = json.loads(_core.expand_spec(_spec_to_json(spec)))
    return cells


def pending_cells(cells: list[dict[str, Any]], done_keys: list[str]) -> list[dict[str, Any]]:
    """The cells in ``cells`` whose key is not in ``done_keys`` (``--resume``)."""
    left: list[dict[str, Any]] = json.loads(
        _core.spec_pending(json.dumps(cells), json.dumps(done_keys))
    )
    return left
