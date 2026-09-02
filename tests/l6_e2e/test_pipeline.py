"""L6 (Colab A100): the dataset pipeline end to end -- ``sweep`` -> ``validate``
-> ``compare`` on a tiny corpus matrix.

Needs a CUDA device with Triton; skipped otherwise. The no-GPU coverage of each
stage lives in ``tests/l0_unit`` / ``tests/l1_contract``; this test is the
proof they compose. (``submit --dry-run`` joins this chain once ``caliper
submit`` lands in W4D1.)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.l6

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():  # pragma: no cover - CI has no GPU
    pytest.skip("needs a CUDA device", allow_module_level=True)


def _sweep_gemm(parquet: Path) -> None:
    from caliper import sweep
    from caliper.corpus.kernels import gemm

    spec = {
        "target": "corpus:gemm",
        "dtypes": ["bf16"],
        "shapes": [{"M": 1024, "N": 1024, "K": 1024}, {"M": 2048, "N": 2048, "K": 2048}],
    }
    sweep(
        spec,
        run_cell=lambda cell, config: gemm.run(cell, config).to_dict(),
        parquet=parquet,
    )


def test_sweep_then_validate_then_compare(tmp_path: Path) -> None:
    from caliper import compare
    from caliper.api import validate_records

    base = tmp_path / "base.parquet"
    _sweep_gemm(base)

    report = validate_records(base)
    assert report["ok"], report["problems"]
    assert report["n"] == 2

    # comparing the run to itself is silent ...
    same = compare(base, base, fail_on_regression=True)
    assert same["any_regression"] is False
    assert same["exit_code"] == 0

    # ... and a hand-perturbed +25% row is caught.
    from caliper import Grid

    rows = [r.to_dict() for r in Grid.from_parquet(base)]
    rows[0]["timing"]["p50_us"] *= 1.25
    slow = tmp_path / "slow.json"
    slow.write_text(json.dumps(rows))

    regressed = compare(base, slow, fail_on_regression=True)
    assert regressed["any_regression"] is True
    assert regressed["exit_code"] == 1
    assert regressed["summary"]["regressions"] == 1


def test_submit_dry_run_joins_the_chain_in_w4() -> None:
    pytest.skip("caliper submit --dry-run lands in W4D1; pipeline extends then")
