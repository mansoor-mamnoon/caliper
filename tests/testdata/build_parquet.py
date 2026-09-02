"""Regenerate the Parquet fixtures from their JSON source of truth.

    python testdata/build_parquet.py

The JSON files are the readable, diffable form; the Parquet files exist so
``caliper compare --baseline testdata/base.parquet ...`` (acceptance playbook
#12) and the L0 comparison tests exercise the real Parquet reader. Run this
after editing any ``testdata/*.json``.
"""

from __future__ import annotations

from pathlib import Path

from caliper import Grid

_HERE = Path(__file__).parent


def main() -> None:
    for name in ("base", "slow", "spill"):
        grid = Grid.from_json(_HERE / f"{name}.json")
        grid.to_parquet(_HERE / f"{name}.parquet")
        print(f"wrote {name}.parquet ({len(grid)} rows)")


if __name__ == "__main__":
    main()
