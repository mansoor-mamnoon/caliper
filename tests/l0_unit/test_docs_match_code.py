"""L0: the reference docs stay in sync with the code (W4D3-T2 verify).

`docs/api.md` must mention every public `caliper` symbol and every record-schema
section field; `docs/cli.md` must document every CLI subcommand and the exit
codes. This catches a signature/command change that forgets the docs.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import caliper
from caliper import _core

pytestmark = pytest.mark.l0

DOCS = Path(__file__).resolve().parents[2] / "docs"
API_MD = (DOCS / "api.md").read_text()
CLI_MD = (DOCS / "cli.md").read_text()
CLI_PY = (Path(__file__).resolve().parents[2] / "python" / "caliper" / "cli.py").read_text()


def test_api_md_documents_every_public_symbol() -> None:
    missing = [name for name in caliper.__all__ if name not in API_MD]
    assert not missing, f"docs/api.md is missing: {missing}"


def test_api_md_lists_every_record_schema_field() -> None:
    record = json.loads(_core.default_record_json())
    fields: list[str] = []
    for key, value in record.items():
        if isinstance(value, dict):
            fields.extend(value.keys())
        else:
            fields.append(key)
    missing = sorted({f for f in fields if f not in API_MD})
    assert not missing, f"docs/api.md's schema table is missing: {missing}"


def _cli_subcommands() -> set[str]:
    tree = ast.parse(CLI_PY)
    names: set[str] = set()
    for node in ast.walk(tree):
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


def test_cli_md_documents_every_subcommand() -> None:
    subcommands = _cli_subcommands()
    assert subcommands, "no subcommands found in cli.py"
    missing = [
        c for c in subcommands if f"caliper {c} " not in CLI_MD and f"`caliper {c}`" not in CLI_MD
    ]
    assert not missing, f"docs/cli.md is missing: {missing}"


def test_cli_md_spells_out_the_exit_codes() -> None:
    for code in ("`0`", "`1`", "`2`"):
        assert code in CLI_MD


def test_the_docs_index_is_linked_from_the_readme() -> None:
    readme = (DOCS.parent / "README.md").read_text()
    for target in ("docs/api.md", "docs/cli.md", "docs/acceptance/manual-playbook.md"):
        assert target in readme, f"README does not link {target}"
