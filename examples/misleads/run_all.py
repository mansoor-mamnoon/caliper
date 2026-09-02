"""Run every ``do_bench`` misleads experiment and rewrite
``docs/data/misleads.csv``.

    make writeup-data
    #  == python examples/misleads/run_all.py

Needs PyTorch + a CUDA device -- run it on Colab, in a **fresh** runtime
(``cold_warmup`` runs first and needs a genuinely cold device). Each
experiment also prints the ``nsys`` command for the ground-truth column, which
you fill in by hand.

The write is atomic: a failed run leaves the committed CSV untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.misleads import cold_warmup, fast_kernel, l2_resident
from examples.misleads._common import CSV_PATH, ORDER, write_csv

_BY_NAME = {"cold_warmup": cold_warmup, "fast_kernel": fast_kernel, "l2_resident": l2_resident}


def main() -> None:
    rows: list[dict[str, str]] = []
    for name in ORDER:
        rows.extend(_BY_NAME[name].main())
    write_csv(rows)
    print(f"\nwrote {CSV_PATH.relative_to(Path(__file__).resolve().parents[2])}")


if __name__ == "__main__":
    main()
