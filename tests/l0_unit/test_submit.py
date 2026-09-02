"""L0: `caliper submit` bundle assembly and the shared bundle validator.

Exhaustive coverage of the manifest / tolerance math is in ``cargo test``;
these drive `caliper.submit` + `caliper.validate_records` against the committed
``tests/testdata`` bundle fixtures and check the two acceptance playbooks:
#13 (a `submit --dry-run` bundle passes `caliper validate`) and #14 (each of
the four rejection classes fails with a specific message).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from caliper import submit
from caliper.api import validate_records

pytestmark = pytest.mark.l0

DATA = Path(__file__).resolve().parents[1] / "testdata"


# -- playbook #13: submit --dry-run -> validate ---------------------------------


def test_submit_dry_run_builds_a_bundle_that_validates(tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    result = submit(DATA / "base.json", out=out, calibration=(101.0, 100.0))

    assert result["n_rows"] == 2
    assert result["branch"] is None
    for name in ("manifest.json", "rows.parquet", "fingerprint.json"):
        assert (out / name).exists()

    m = result["manifest"]
    assert m["arch"] == "sm_80"
    assert m["kernels"] == ["gemm", "rmsnorm"]
    assert m["tier"] == "locked"
    assert m["calibration"]["within_tolerance"] is True
    assert len(m["toolchain_hash"]) == 64

    report = validate_records(out)
    assert report["ok"], report["problems"]
    assert report["n"] == 2
    assert report["bundle"] == str(out)


def test_submit_needs_rows() -> None:
    with pytest.raises(ValueError, match="no rows"):
        submit([])


def test_submit_a_repo_url_is_not_supported(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a git checkout"):
        submit(DATA / "base.json", repo=tmp_path, dry_run=False)


# -- playbook #14: the four rejection classes ---------------------------------


@pytest.mark.parametrize(
    ("bundle", "needle"),
    [
        ("bundle_missing_field", "missing required field kernel.name"),
        ("bundle_nonreproducing", "determinism repeat CoV"),
        ("bundle_slow_calibration", "calibration GEMM p50 is"),
    ],
)
def test_a_bad_bundle_is_rejected_with_a_specific_message(bundle: str, needle: str) -> None:
    report = validate_records(DATA / bundle)
    assert report["ok"] is False
    flat = " ".join(p for entry in report["problems"] for p in entry["problems"])
    assert needle in flat


def test_an_over_peak_row_is_rejected_by_a_bare_validate() -> None:
    report = validate_records(DATA / "over_peak_row.json")
    assert report["ok"] is False
    assert "outside the plausible range" in report["problems"][0]["problems"][0]


def test_the_clean_bundle_fixture_still_validates() -> None:
    assert validate_records(DATA / "bundle_ok")["ok"] is True


def test_validate_names_what_a_non_bundle_directory_is_missing(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="not a bundle: missing"):
        validate_records(tmp_path)


# -- manifest details --------------------------------------------------------


def test_unlocked_rows_and_a_repeat_become_tier_and_determinism(tmp_path: Path) -> None:
    result = submit(DATA / "bundle_nonreproducing/rows.jsonl", out=tmp_path / "b")
    m = result["manifest"]
    assert m["tier"] == "unlocked"
    assert m["determinism"]["n_repeats"] == 4
    assert m["determinism"]["within_tolerance"] is False
