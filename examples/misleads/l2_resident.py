"""Experiment (c): an L2-resident working set, timed with and without a flush.

A kernel whose inputs fit in L2 (tens of MB on recent GPUs) will, in a tight
loop with no cache management, read warm cache on every iteration after the
first. The number looks great and is unreachable in a real pipeline where the
data was last touched by some other kernel. ``do_bench`` clears an L2-sized
buffer between reps; a hand-rolled loop usually does not.

    python examples/misleads/l2_resident.py
    python examples/misleads/l2_resident.py --nsys
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.misleads._common import (
    do_bench_median_us,
    no_flush_median_us,
    report,
)

MB = 1 << 20
BYTES = 8 * MB  # 8 MiB: comfortably inside the L2 of an A100 / L4 / H100


def kernel() -> Any:
    import torch

    x = torch.randn(BYTES // 4, device="cuda")  # fp32
    out = torch.empty_like(x)

    def fn() -> None:
        torch.mul(x, 2.0, out=out)

    return fn


def main(*, write: bool) -> None:
    fn = kernel()
    warm = no_flush_median_us(fn)
    flushed = do_bench_median_us(fn)
    report(
        "l2_resident",
        {"no_flush": warm, "do_bench_flushed": flushed},
        note=f"{BYTES // MB} MiB working set (fits in L2)",
        write=write,
    )
    if warm:
        print(f"  flush penalty:  {flushed / warm:5.2f}x")


if __name__ == "__main__":
    if "--nsys" in sys.argv:
        import torch

        fn = kernel()
        for _ in range(4000):
            fn()
        torch.cuda.synchronize()
    else:
        main(write="--write" in sys.argv)
