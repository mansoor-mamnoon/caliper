"""L0: the acceptance traceability map covers every requirement.

`docs/acceptance/traceability.md` must have a row for every `FR-N` / `NFR-N` in
the plan, and the acceptance notebook must be a valid GPU notebook. This is the
no-GPU guard for W4D4 -- the on-device runs themselves happen on Colab / a
rented instance and land under `docs/acceptance/reports/`.
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


def test_traceability_covers_every_requirement() -> None:
    reqs = _requirements()
    assert len(reqs) >= 25, f"only found {len(reqs)} requirements in the plan"
    missing = sorted(r for r in reqs if r not in TRACE)
    assert not missing, f"traceability.md has no row for: {missing}"


def test_traceability_marks_status_and_is_linked() -> None:
    assert "**CI**" in TRACE and "pending" in TRACE
    assert "notebooks/acceptance.ipynb" in TRACE
    for spec in ("examples/acceptance-sweep.yaml",):
        assert (REPO / spec).exists()


def test_the_acceptance_notebook_is_a_valid_gpu_notebook() -> None:
    nb = json.loads((REPO / "notebooks" / "acceptance.ipynb").read_text())
    assert nb["nbformat"] == 4
    assert nb["metadata"].get("accelerator") == "GPU"
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    # it drives the scriptable playbook steps
    for token in (
        "caliper doctor",
        "caliper selftest",
        "sweep(",
        "validate_records",
        "caliper compare",
    ):
        assert token in src, f"acceptance.ipynb never calls {token}"


def test_the_acceptance_sweep_spec_is_thirty_cells() -> None:
    pytest.importorskip("yaml")
    from caliper._spec import load_cells

    cells = load_cells(str(REPO / "examples" / "acceptance-sweep.yaml"))
    assert len(cells) == 30  # square-pow2 (5) x {bf16,fp16,fp8_e4m3} x {row,col}
