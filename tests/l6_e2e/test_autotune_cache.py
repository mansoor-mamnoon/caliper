"""L6 (Colab A100): the autotune-config cache end to end through sweep().

Runs a real corpus GEMM sweep with the kernel's own Triton configs, then a
second sweep with one config added, and asserts only the new config was
re-timed. Needs a CUDA device with Triton; skipped otherwise.

The no-GPU coverage of the same hit/miss logic lives in
``tests/l1_contract/test_sweep.py::test_autotune_configs_are_timed_once_and_the_fastest_kept``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.l6

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():  # pragma: no cover - CI has no GPU
    pytest.skip("needs a CUDA device", allow_module_level=True)


def _configs(kernel: Any) -> list[dict[str, Any]]:
    """The kwargs of a `@triton.autotune` kernel's configs."""
    out: list[dict[str, Any]] = []
    for c in getattr(kernel, "configs", []):
        out.append({str(k): v for k, v in c.kwargs.items()})
    return out


def test_adding_a_config_re_times_only_the_new_one(tmp_path: Path) -> None:
    from caliper import sweep
    from caliper.corpus.kernels import gemm

    spec = {
        "target": "corpus:gemm",
        "dtypes": ["bf16"],
        "shapes": [{"M": 4096, "N": 4096, "K": 4096}],
        "autotune": "from_kernel",
    }
    cache = tmp_path / "autotune.json"
    base_configs = _configs(gemm.kernel)[:3]

    timed: list[dict[str, Any]] = []

    def run_cell(cell: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        timed.append(config)
        record: dict[str, Any] = gemm.run(cell, config).to_dict()
        return record

    sweep(
        spec,
        run_cell=run_cell,
        configs_for=lambda _c: base_configs,
        cache_path=cache,
        parquet=tmp_path / "v1.parquet",
    )
    assert len(timed) == len(base_configs)

    timed.clear()
    sweep(
        spec,
        run_cell=run_cell,
        configs_for=lambda _c: [*base_configs, {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}],
        cache_path=cache,
        parquet=tmp_path / "v2.parquet",
    )
    assert len(timed) == 1  # only the added config
