"""L0: the reference kernel corpus's no-GPU surface.

Every corpus kernel module must import cleanly without Triton or PyTorch
installed (this dev box has neither), so its pure pieces -- ``content_hash``,
``roofline_spec``, the autotune ``CONFIGS`` -- can be checked here. The actual
Triton kernels only run on a CUDA host; that path is exercised by
``tests/l6_e2e/test_autotune_cache.py`` (gemm) and on Colab.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from caliper.corpus import _common
from caliper.corpus.kernels import gemm, rmsnorm, softmax

pytestmark = pytest.mark.l0

KERNELS = [gemm, rmsnorm, softmax]


# -- _common ------------------------------------------------------------------


def test_content_hash_is_deterministic_and_matches_sha256(tmp_path: Path) -> None:
    f = tmp_path / "k.py"
    f.write_text("print('hello')\n")
    want = "sha256:" + hashlib.sha256(f.read_bytes()).hexdigest()
    assert _common.content_hash(f) == want
    assert _common.content_hash(f) == _common.content_hash(f)  # deterministic

    f.write_text("print('changed')\n")
    assert _common.content_hash(f) != want  # content-addressed, not path-addressed


def test_has_module_finds_stdlib_and_rejects_nonsense() -> None:
    assert _common.has_module("json")
    assert not _common.has_module("no_such_module_xyz")


def test_triton_pin_is_a_real_full_length_commit_sha() -> None:
    assert _common.TRITON_PIN.repo == "triton-lang/triton"
    assert len(_common.TRITON_PIN.commit) == 40
    assert all(c in "0123456789abcdef" for c in _common.TRITON_PIN.commit)


def test_require_live_deps_degrades_honestly_without_torch_or_triton() -> None:
    # Neither torch nor triton is installed on this dev box.
    with pytest.raises(NotImplementedError, match=r"corpus\.kernels\.gemm\.run"):
        _common.require_live_deps("gemm")


# -- every kernel module --------------------------------------------------


@pytest.mark.parametrize("mod", KERNELS)
def test_kernel_module_imports_without_triton(mod: Any) -> None:
    # Getting this far means the module-level import already succeeded (this
    # dev box has no triton); the guard flag must agree.
    assert mod.TRITON_AVAILABLE is False
    assert mod.kernel is None


@pytest.mark.parametrize("mod", KERNELS)
def test_kernel_source_hash_is_sha256_of_its_own_file(mod: Any) -> None:
    assert _common.content_hash(mod.__file__) == mod.SOURCE_HASH


@pytest.mark.parametrize("mod", KERNELS)
def test_run_raises_cleanly_without_live_deps(mod: Any) -> None:
    shape = {"M": 16, "N": 16, "K": 16} if mod is gemm else {"ROWS": 16, "COLS": 16}
    with pytest.raises(NotImplementedError, match="run"):
        mod.run({"shape": shape, "dtype": "bf16", "layout": "row"})


# -- roofline_spec: pure math, hand-computable -----------------------------


def test_gemm_roofline_spec_matches_hand_computed_flops_and_bytes() -> None:
    spec = gemm.roofline_spec({"M": 1024, "N": 2048, "K": 512}, "bf16")
    assert spec is not None
    assert spec["flops"] == pytest.approx(2 * 1024 * 2048 * 512)
    assert spec["bytes_hbm"] == pytest.approx((1024 * 512 + 512 * 2048 + 1024 * 2048) * 2)


def test_gemm_roofline_spec_is_none_for_a_missing_dimension() -> None:
    assert gemm.roofline_spec({"M": 1024, "N": 2048}, "bf16") is None


@pytest.mark.parametrize(("mod", "flops_per_elem"), [(rmsnorm, 4.0), (softmax, 5.0)])
def test_elementwise_roofline_spec_matches_hand_computed_flops_and_bytes(
    mod: Any, flops_per_elem: float
) -> None:
    spec = mod.roofline_spec({"ROWS": 256, "COLS": 4096}, "bf16")
    assert spec is not None
    assert spec["flops"] == pytest.approx(flops_per_elem * 256 * 4096)
    assert spec["bytes_hbm"] == pytest.approx(2 * 256 * 4096 * 2)  # read + write, bf16


@pytest.mark.parametrize("mod", [rmsnorm, softmax])
def test_elementwise_roofline_spec_is_none_for_a_missing_dimension(mod: Any) -> None:
    assert mod.roofline_spec({"ROWS": 256}, "bf16") is None


# -- assemble_result: the "valid row" half of the DoD, off-device ----------


def _fake_machine() -> dict[str, Any]:
    return {
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "sm_arch": "sm_80",
        "vram_mib": 40960,
        "sm_count": 108,
        "cuda_runtime": "12.4",
        "toolkit": {"triton": "3.1.0", "torch": "2.5.0", "ptxas": "12.4", "nvcc": None},
    }


def test_assemble_result_builds_a_schema_valid_record() -> None:
    samples_us = [243.1, 243.4, 242.9, 244.0, 243.2]
    spec = gemm.roofline_spec({"M": 4096, "N": 4096, "K": 4096}, "bf16")
    assert spec is not None

    result = _common.assemble_result(
        kernel_name="gemm",
        kernel_impl="triton",
        dtype="bf16",
        layout="row",
        shape={"M": 4096, "N": 4096, "K": 4096},
        source_hash=gemm.SOURCE_HASH,
        autotune_config=gemm.CONFIGS[0],
        samples_us=samples_us,
        machine=_fake_machine(),
        flops=spec["flops"],
        bytes_hbm=spec["bytes_hbm"],
        baseline="cublas",
        baseline_pct=0.94,
    )

    assert result.validate() == []  # the DoD's "valid row"
    assert result.p50_us == pytest.approx(243.2)
    assert set(result.flags) == {"clocks-unlocked", "corpus-live-timing"}
    assert result.kernel["source_hash"] == gemm.SOURCE_HASH
    assert result.roofline["baseline"] == "cublas"
    pct = result.roofline_pct
    assert pct is not None and 0.0 <= pct <= 1.5


def test_assemble_result_omits_roofline_without_flop_counts() -> None:
    result = _common.assemble_result(
        kernel_name="softmax",
        kernel_impl="triton",
        dtype="bf16",
        layout=None,
        shape={"ROWS": 4096, "COLS": 4096},
        source_hash=softmax.SOURCE_HASH,
        autotune_config={"BLOCK_SIZE": 4096},
        samples_us=[10.0, 11.0, 10.5],
        machine=_fake_machine(),
    )
    assert result.validate() == []
    assert result.roofline["achieved_tflops"] is None


# -- gemm's autotune contract (tests/l6_e2e/test_autotune_cache.py needs this) --


def test_gemm_configs_are_a_non_empty_list_of_distinct_block_shapes() -> None:
    assert len(gemm.CONFIGS) >= 3
    seen = {tuple(sorted(c.items())) for c in gemm.CONFIGS}
    assert len(seen) == len(gemm.CONFIGS)  # no duplicate configs
    for cfg in gemm.CONFIGS:
        assert {"BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M"} <= cfg.keys()
