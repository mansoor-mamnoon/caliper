"""Experiment (a): a sub-20 us kernel.

``do_bench`` records a CUDA-event pair around every single launch. For a kernel
that runs in ~10 us, the ~1-3 us each ``cudaEventRecord`` costs is a real
fraction of the measurement -- and a hand-rolled per-iteration ``synchronize()``
loop is far worse. Timing a *batch* of launches between one event pair
(``caliper.bench(batch=…)``) divides that overhead away.

    python examples/misleads/fast_kernel.py            # measure, print a table
    python examples/misleads/fast_kernel.py --nsys     # just spin, for nsys
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.misleads._common import (
    batched_launch_us,
    do_bench_median_us,
    naive_per_iter_sync_us,
    report,
)

N = 1 << 16  # 64Ki elements: a handful of us of copy on any recent GPU


def kernel() -> Any:
    import torch

    x = torch.randn(N, device="cuda")
    out = torch.empty_like(x)

    def fn() -> None:
        torch.add(x, 1.0, out=out)

    return fn


def main(*, write: bool) -> None:
    fn = kernel()
    report(
        "fast_kernel",
        {
            "naive_per_iter_sync": naive_per_iter_sync_us(fn),
            "do_bench_default": do_bench_median_us(fn),
            "caliper_batched": batched_launch_us(fn),
        },
        note=f"elementwise add, {N} elements",
        write=write,
    )


if __name__ == "__main__":
    if "--nsys" in sys.argv:
        import torch

        fn = kernel()
        for _ in range(5000):
            fn()
        torch.cuda.synchronize()
    else:
        main(write="--write" in sys.argv)
