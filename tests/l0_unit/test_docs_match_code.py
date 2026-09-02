"""L0: the reference docs stay in sync with the code (W4D3-T2 verify).

`docs/api.md` carries a `###` heading with the real signature for every public
callable and a schema table with every record field; `docs/cli.md` documents
every CLI subcommand and the exit codes. This catches a signature / command /
schema change that forgets the docs.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import caliper
from caliper import _core, api

pytestmark = pytest.mark.l0

REPO = Path(__file__).resolve().parents[2]
API_MD = (REPO / "docs" / "api.md").read_text()
CLI_MD = (REPO / "docs" / "cli.md").read_text()
CLI_PY = (REPO / "python" / "caliper" / "cli.py").read_text()


def _public_functions() -> dict[str, Callable[..., Any]]:
    """`caliper.__all__` entries that are plain functions (not classes / str)."""
    out: dict[str, Callable[..., Any]] = {}
    for name in caliper.__all__:
        obj = getattr(api, name, getattr(caliper, name, None))
        if inspect.isfunction(obj):
            out[name] = obj
    return out


_CALLABLES = _public_functions()


def _heading(name: str) -> str:
    """The `### name(...)` heading line for `name` in api.md, or ''."""
    m = re.search(rf"^### `?{re.escape(name)}\(.*$", API_MD, re.MULTILINE)
    return m.group(0) if m else ""


@pytest.mark.parametrize("name", sorted(_CALLABLES))
def test_api_md_heading_matches_the_real_signature(name: str) -> None:
    heading = _heading(name)
    assert heading, f"docs/api.md has no `### {name}(...)` heading"
    sig = inspect.signature(_CALLABLES[name])
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        assert param.name in heading, (
            f"docs/api.md's `{name}` heading is missing parameter '{param.name}'\n  {heading}"
        )
    if any(p.kind is inspect.Parameter.KEYWORD_ONLY for p in sig.parameters.values()):
        assert " *, " in heading or "(*, " in heading, (
            f"docs/api.md's `{name}` heading drops the keyword-only `*` marker"
        )


def test_api_md_mentions_every_public_symbol() -> None:
    # the rigorous per-signature check is above; this only catches a symbol
    # that vanished from the page entirely.
    missing = [n for n in caliper.__all__ if n not in API_MD]
    assert not missing, f"docs/api.md never mentions: {missing}"


def test_api_md_schema_table_lists_every_record_field() -> None:
    start = API_MD.index("## The record / row schema")
    end = API_MD.index("\n## ", start)
    table = API_MD[start:end]

    record = json.loads(_core.default_record_json())
    for key, value in record.items():
        fields = list(value) if isinstance(value, dict) else [key]
        if key == "machine":
            fields += list(record["machine"]["toolkit"])
        for field in fields:
            assert field in table, f"docs/api.md schema table is missing `{key}.{field}`"


def _cli_subcommands() -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(CLI_PY)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_parser"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
    return names


def test_cli_md_documents_every_subcommand_with_a_heading() -> None:
    subcommands = _cli_subcommands()
    assert subcommands, "no subcommands found in cli.py"
    for name in subcommands:
        assert re.search(rf"^## `caliper {re.escape(name)}[ `]", CLI_MD, re.MULTILINE), (
            f"docs/cli.md has no `## caliper {name}` section"
        )


def test_cli_md_spells_out_the_exit_codes() -> None:
    for code in ("`0`", "`1`", "`2`"):
        assert code in CLI_MD


def test_the_docs_are_linked_from_the_readme() -> None:
    readme = (REPO / "README.md").read_text()
    for target in (
        "docs/api.md",
        "docs/cli.md",
        "docs/shapes.md",
        "docs/acceptance/manual-playbook.md",
    ):
        assert target in readme, f"README does not link {target}"
