"""Command-line entry point for caliper.

Only ``--version`` and ``--help`` are wired up so far. The measurement,
sweep, compare, selftest, validate, and submit commands are being implemented
incrementally; each will be added here as its module lands.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from caliper.__about__ import __version__

_PLANNED_COMMANDS = (
    "doctor",
    "fingerprint",
    "bench",
    "sweep",
    "compare",
    "selftest",
    "validate",
    "submit",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="caliper",
        description="Correct-by-default GPU kernel benchmarking.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"caliper {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the caliper CLI. Returns a process exit code."""
    parser = _build_parser()
    args, extra = parser.parse_known_args(argv)
    del args

    if extra:
        print(
            f"caliper {__version__}: command {extra[0]!r} is not available yet.\n"
            f"Planned commands: {', '.join(_PLANNED_COMMANDS)}.",
            file=sys.stderr,
        )
        return 2

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
