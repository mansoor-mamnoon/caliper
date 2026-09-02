"""L0: the acceptance traceability map covers every requirement.

`docs/acceptance/traceability.md` must have a row for every `FR-N` / `NFR-N` in
the plan, and the acceptance notebook must be a valid GPU notebook. This is the
no-GPU guard for the acceptance work -- the on-device runs themselves happen on
Colab / a rented instance and land under `docs/acceptance/reports/`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.l0

REPO = Path(__file__).resolve().parents[2]
PLAN = (REPO / "docs" / "plan.md").read_text()
TRACE = (REPO / "docs" / "acceptance" / "traceability.md").read_text()


def _requirements() -> set[str]:
    """Every `FR-N` / `NFR-N` the plan defines a row for."""
    return set(re.findall(r"\|\s*\*\*((?:N?FR)-\d+)\*\*\s*\|", PLAN))


def test_traceability_has_a_table_row_per_requirement() -> None:
    reqs = _requirements()
    assert len(reqs) >= 25, f"only found {len(reqs)} requirements in the plan"
    # each requirement must start its own table row: `| FR-1 ` (not just appear
    # as a substring of FR-19 etc.)
    missing = sorted(r for r in reqs if not re.search(rf"^\| {re.escape(r)}[ `]", TRACE, re.M))
    assert not missing, f"traceability.md has no `| {{req}} ...` row for: {missing}"


def test_every_traceability_row_has_a_status() -> None:
    rows = [ln for ln in TRACE.splitlines() if re.match(r"\| (?:N?FR)-\d+ ", ln)]
    assert len(rows) >= 25, f"only parsed {len(rows)} requirement rows"
    for row in rows:
        status = row.rstrip("| ").rsplit("|", 1)[-1].strip()
        assert status in ("pending", "**CI**", "**CI** (logic)"), f"odd status: {status!r}"


def test_traceability_is_linked_to_the_harness() -> None:
    assert "notebooks/acceptance.ipynb" in TRACE
    assert (REPO / "examples" / "acceptance-sweep.yaml").exists()


def test_the_acceptance_notebook_is_a_valid_gpu_notebook() -> None:
    nb = json.loads((REPO / "notebooks" / "acceptance.ipynb").read_text())
    assert nb["nbformat"] == 4
    assert nb["metadata"].get("accelerator") == "GPU"
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    for token in (
        "caliper doctor",
        "caliper selftest",
        "sweep(",
        "validate_records",
        "caliper compare",
    ):
        assert token in src, f"acceptance.ipynb never calls {token}"


def test_the_acceptance_sweep_spec_is_twenty_runnable_cells() -> None:
    pytest.importorskip("yaml")
    from caliper._spec import load_cells

    cells = load_cells(str(REPO / "examples" / "acceptance-sweep.yaml"))
    assert len(cells) == 20  # square-pow2 (5) x {bf16, fp16} x {row, col}
    # the corpus gemm kernel only has bf16 / fp16 / fp32 paths today
    assert {c["dtype"] for c in cells} <= {"bf16", "fp16", "fp32"}
