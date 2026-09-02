"""L0: the regression threshold model behind ``caliper compare``.

Exhaustive coverage of the band / verdict / delta math is in ``cargo test``;
these drive the whole thing through ``caliper.compare`` against the committed
``tests/testdata`` fixtures -- one facet that regresses, one that does not --
and check the DoD: an injected slowdown fires, an injected spill regression
fires with the delta shown, a within-noise difference stays silent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from caliper import _core, compare

pytestmark = pytest.mark.l0

DATA = Path(__file__).resolve().parents[1] / "testdata"


def _facet(report: dict[str, Any], kernel: str) -> dict[str, Any]:
    return next(f for f in report["facets"] if f["key"]["kernel"] == kernel)


@pytest.mark.parametrize("suffix", [".json", ".parquet"])
def test_an_injected_slowdown_fires_and_within_noise_stays_silent(suffix: str) -> None:
    report = compare(DATA / f"base{suffix}", DATA / f"slow{suffix}")

    gemm = _facet(report, "gemm")
    assert gemm["verdict"] == "regression"
    assert gemm["delta_pct"] == pytest.approx((272.0 - 243.2) / 243.2)
    assert gemm["delta_pct"] > gemm["noise_band_pct"]

    rmsnorm = _facet(report, "rmsnorm")
    assert rmsnorm["verdict"] == "within_noise"  # +0.5%, inside the band

    assert report["any_regression"] is True
    assert report["summary"]["regressions"] == 1


def test_the_slowdown_report_also_carries_the_spill_delta() -> None:
    # acceptance playbook #12: one command shows the slowdown *and* the spill.
    gemm = _facet(compare(DATA / "base.json", DATA / "slow.json"), "gemm")
    assert gemm["spill_regression"] is True
    assert gemm["ptxas_delta"]["spill_stores_bytes"] == 256
    assert gemm["ptxas_delta"]["regs_per_thread"] == 8


def test_an_injected_spill_regression_fires_with_the_timing_held_flat() -> None:
    report = compare(DATA / "base.json", DATA / "spill.json")
    gemm = _facet(report, "gemm")
    assert gemm["verdict"] == "within_noise"  # +0.2%, not a *timing* regression
    assert gemm["spill_regression"] is True
    assert gemm["ptxas_delta"]["spill_stores_bytes"] == 384
    assert gemm["ptxas_delta"]["spill_loads_bytes"] == 128
    assert report["any_regression"] is True  # ... but the run still fails
    assert report["summary"]["spill_regressions"] == 1
    assert report["summary"]["regressions"] == 0


def test_comparing_a_dataset_to_itself_is_silent() -> None:
    report = compare(DATA / "base.json", DATA / "base.json", fail_on_regression=True)
    assert report["any_regression"] is False
    assert report["exit_code"] == 0
    assert {f["verdict"] for f in report["facets"]} == {"within_noise"}


def test_an_explicit_threshold_overrides_the_derived_band() -> None:
    # the gemm facet is ~11.8% slower; a 20% threshold silences the *timing*
    # verdict (the independent spill regression still fails the run).
    report = compare(DATA / "base.json", DATA / "slow.json", threshold=0.20)
    assert _facet(report, "gemm")["verdict"] == "within_noise"
    assert _facet(report, "gemm")["noise_band_pct"] == pytest.approx(0.20)
    assert report["summary"]["regressions"] == 0


def test_the_arch_filter_selects_only_matching_rows() -> None:
    matched = compare(DATA / "base.json", DATA / "slow.json", arch="sm_80")
    assert matched["arch"] == "sm_80"
    assert matched["summary"]["facets"] == 2

    none = compare(DATA / "base.json", DATA / "slow.json", arch="sm_90")
    assert none["summary"]["facets"] == 0
    assert none["any_regression"] is False


def test_a_dropped_autotune_config_is_flagged() -> None:
    base = json.loads((DATA / "base.json").read_text())
    extra = json.loads(json.dumps(base[0]))
    extra["kernel"]["autotune_config"] = {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}
    extra["timing"]["p50_us"] = 999.0  # slower, so the kept-config row is still base[0]
    report = json.loads(_core.compare_datasets(json.dumps([*base, extra]), json.dumps(base), "{}"))
    gemm = _facet(report, "gemm")
    assert gemm["autotune_configs_dropped"] == ['{"BLOCK_K":32,"BLOCK_M":256,"BLOCK_N":128}']
    assert report["summary"]["configs_dropped"] == 1


def test_json_and_parquet_fixtures_compare_identically() -> None:
    from_json = compare(DATA / "base.json", DATA / "slow.json")
    from_parquet = compare(DATA / "base.parquet", DATA / "slow.parquet")
    assert from_json["summary"] == from_parquet["summary"]
    assert from_json["any_regression"] == from_parquet["any_regression"]
