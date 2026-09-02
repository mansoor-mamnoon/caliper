"""L0: the ``do_bench`` misleads example scripts and their writeup.

The experiments themselves need a GPU (they run on Colab via ``make
writeup-data``); here we only check that the scripts import cleanly with no
torch, that the CSV template and the ``report()`` output agree on their shape,
and that ``docs/why-do_bench-misleads.md`` carries the cited issue links.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from examples.misleads import _common, cold_warmup, fast_kernel, l2_resident, run_all

pytestmark = pytest.mark.l0

REPO = Path(__file__).resolve().parents[2]
WRITEUP = REPO / "docs" / "why-do_bench-misleads.md"

EXPERIMENTS = {"fast_kernel": fast_kernel, "cold_warmup": cold_warmup, "l2_resident": l2_resident}


def test_every_experiment_module_has_a_kernel_and_a_main() -> None:
    for mod in EXPERIMENTS.values():
        assert callable(mod.kernel)
        assert callable(mod.main)
    assert callable(run_all.main)


def test_the_committed_csv_template_matches_the_writer_schema() -> None:
    with _common.CSV_PATH.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "the CSV template should carry placeholder rows"
    assert list(rows[0].keys()) == _common.FIELDS
    assert {r["experiment"] for r in rows} == set(EXPERIMENTS)


def test_a_measurement_helper_degrades_honestly_without_cuda() -> None:
    # no torch on the dev box -> a clear RuntimeError, not an ImportError leak.
    with pytest.raises(RuntimeError, match="Colab"):
        _common.arch_tag()


def test_nsys_command_points_at_the_right_script() -> None:
    cmd = _common.nsys_command("fast_kernel")
    assert "nsys profile" in cmd
    assert "examples/misleads/fast_kernel.py --nsys" in cmd


def test_the_writeup_exists_and_cites_the_issues() -> None:
    text = WRITEUP.read_text()
    for issue in (
        "triton-lang/triton/issues/2306",
        "triton-lang/triton/issues/1252",
        "triton-lang/triton/issues/404",
        "triton-lang/triton/issues/2832",
        "flashinfer-ai/flashinfer-bench/issues/195",
    ):
        assert issue in text, f"{issue} not cited in the writeup"
    assert "data/misleads.csv" in text
    for experiment in EXPERIMENTS:
        assert experiment in text
