"""Run every ``do_bench`` misleads experiment and (re)write
``docs/data/misleads.csv``.

    make writeup-data
    #  == python examples/misleads/run_all.py

Needs PyTorch + a CUDA device -- run it on Colab. Each experiment also prints
the ``nsys`` command to fill in the ground-truth column.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.misleads import cold_warmup, fast_kernel, l2_resident
from examples.misleads._common import CSV_PATH


def main() -> None:
    if CSV_PATH.exists():
        CSV_PATH.unlink()
    for experiment in (fast_kernel, cold_warmup, l2_resident):
        experiment.main(write=True)
    print(f"\nwrote {CSV_PATH.relative_to(Path(__file__).resolve().parents[2])}")


if __name__ == "__main__":
    main()
