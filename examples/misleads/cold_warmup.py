"""Experiment (b): a fixed warmup budget that misses steady state.

``do_bench`` defaults to a 25 ms warmup. On a cold device the first ~100 ms of
kernels run while the SM clock is still ramping (and, on a shared Colab box,
while it settles after the previous tenant). A kernel measured inside that
window reads faster than it will ever run again. caliper takes a generous
warmup budget and then trims to the steady-state index
(``_core.steady_state_index``), so the reported median is the number you get in
production.

    python examples/misleads/cold_warmup.py
    python examples/misleads/cold_warmup.py --nsys
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.misleads._common import (
    caliper_steady_median_us,
    do_bench_median_us,
    report,
)

M = 4096  # a square-ish GEMM: enough work that the clock matters


def kernel() -> Any:
    import torch

    a = torch.randn(M, M, device="cuda", dtype=torch.float16)
    b = torch.randn(M, M, device="cuda", dtype=torch.float16)
    out = torch.empty(M, M, device="cuda", dtype=torch.float16)

    def fn() -> None:
        torch.matmul(a, b, out=out)

    return fn


def main(*, write: bool) -> None:
    fn = kernel()
    # do_bench first, from cold, so its short warmup lands in the ramp window.
    quoted = do_bench_median_us(fn)
    steady = caliper_steady_median_us(fn, warmup_ms=1000.0, rep_ms=800.0)
    report(
        "cold_warmup",
        {"do_bench_warmup_25ms": quoted, "caliper_steady_state": steady},
        note=f"fp16 matmul {M}x{M}; run from a cold device",
        write=write,
    )


if __name__ == "__main__":
    if "--nsys" in sys.argv:
        import torch

        fn = kernel()
        for _ in range(2000):
            fn()
        torch.cuda.synchronize()
    else:
        main(write="--write" in sys.argv)
