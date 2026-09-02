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


def test_submit_repo_must_be_a_local_git_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a path to a local"):
        submit(DATA / "base.json", repo="https://github.com/x/caliper-results", dry_run=False)
    with pytest.raises(ValueError, match="not a git checkout"):
        submit(DATA / "base.json", repo=tmp_path, dry_run=False)


def test_submit_to_a_repo_writes_a_branch_and_a_re_run_is_a_clean_error(tmp_path: Path) -> None:
    import subprocess

    repo = tmp_path / "caliper-results"
    repo.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    (repo / "README.md").write_text("seed\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "seed"], check=True)

    result = submit(DATA / "base.json", repo=repo, dry_run=False)
    assert result["branch"].startswith("submit/sm_80-")
    bundle_dir = next((repo / "results").glob("sm_80/*"))
    assert (bundle_dir / "manifest.json").exists()

    # the branch already exists -> a clean ValueError, not a CalledProcessError.
    with pytest.raises(ValueError, match="git checkout failed"):
        submit(DATA / "base.json", repo=repo, dry_run=False)


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


def test_a_bundle_with_a_tampered_manifest_verdict_is_still_rejected(tmp_path: Path) -> None:
    import json
    import shutil

    src = DATA / "bundle_nonreproducing"
    dst = tmp_path / "tampered"
    shutil.copytree(src, dst)
    manifest = json.loads((dst / "manifest.json").read_text())
    manifest.pop("determinism", None)  # the submitter hides the bad repeat
    (dst / "manifest.json").write_text(json.dumps(manifest))

    report = validate_records(dst)
    assert report["ok"] is False  # recomputed from rows.jsonl, not trusted
    flat = " ".join(p for entry in report["problems"] for p in entry["problems"])
    assert "determinism repeat CoV" in flat


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


def test_submit_rejects_rows_from_more_than_one_machine(tmp_path: Path) -> None:
    import json

    rows = json.loads((DATA / "base.json").read_text())
    rows[1]["machine"] = {**rows[1]["machine"], "gpu_name": "a different card"}
    mixed = tmp_path / "mixed.json"
    mixed.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="machine fingerprint differs"):
        submit(mixed)
