"""L0: the ``do_bench`` misleads example scripts and their writeup.

The experiments themselves need a GPU (they run on Colab via ``make
writeup-data``); here we only check that the scripts import cleanly with no
torch, that the CSV template / the ``METHODS`` map / the writeup tables agree,
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

MODULES = {"fast_kernel": fast_kernel, "cold_warmup": cold_warmup, "l2_resident": l2_resident}


def test_every_experiment_module_has_a_kernel_and_a_main() -> None:
    for mod in MODULES.values():
        assert callable(mod.kernel)
        assert callable(mod.main)
    assert callable(run_all.main)
    assert set(_common.ORDER) == set(MODULES)
    assert _common.ORDER[0] == "cold_warmup"  # measured cold, before the others


def test_the_committed_csv_template_matches_METHODS_and_the_field_order() -> None:
    with _common.CSV_PATH.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "the CSV template should carry placeholder rows"
    assert list(rows[0].keys()) == _common.FIELDS

    by_experiment: dict[str, list[str]] = {}
    for row in rows:
        by_experiment.setdefault(row["experiment"], []).append(row["method"])
    assert by_experiment == {name: list(methods) for name, methods in _common.METHODS.items()}
    # every non-nsys row is a blank slot until `make writeup-data` runs on a GPU
    assert all(row["value_us"] == "" for row in rows)


def test_write_csv_is_atomic_and_leaves_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.csv"
    monkeypatch.setattr(_common, "CSV_PATH", target)
    row = {"experiment": "x", "arch": "sm_80", "method": "m", "value_us": "1.0", "note": "n"}
    _common.write_csv([row])
    assert target.exists()
    assert not target.with_suffix(".csv.tmp").exists()
    assert target.read_text().splitlines()[0] == ",".join(_common.FIELDS)


def test_a_measurement_helper_degrades_honestly_without_cuda() -> None:
    # no torch on the dev box -> a clear RuntimeError, not an ImportError leak.
    with pytest.raises(RuntimeError, match="Colab"):
        _common.arch_tag()
    with pytest.raises(RuntimeError, match="Colab"):
        _common.require_cuda()


@pytest.mark.parametrize("mod", list(MODULES.values()))
def test_a_script_kernel_gives_the_friendly_error_off_gpu(mod: object) -> None:
    with pytest.raises(RuntimeError, match="Colab"):
        mod.kernel()  # type: ignore[attr-defined]


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
    for name in MODULES:
        assert name in text
    # the writeup is honest that the numbers aren't in the repo
    assert "*template*" in text
    assert "blank in the repo" in text
