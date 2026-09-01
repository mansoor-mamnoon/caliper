"""Command-line entry point for caliper.

Wired up so far: ``bench`` (recorded-session path), ``doctor``, ``fingerprint``,
plus ``--version`` / ``--help``. Each command takes ``--json`` for
machine-readable output. The sweep / compare / selftest / validate / submit
commands are added as their supporting code lands.

Exit codes: 0 success, 1 "not fit" (doctor only), 2 usage / runtime error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from caliper import __version__, api


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="caliper",
        description="Correct-by-default GPU kernel benchmarking.",
    )
    parser.add_argument("--version", action="version", version=f"caliper {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_bench = sub.add_parser("bench", help="measure one kernel")
    p_bench.add_argument("target", nargs="?", help="kernel target, e.g. file.py::fn or corpus:o1")
    p_bench.add_argument(
        "--recording", metavar="PATH", help="JSON Lines device recording to replay"
    )
    p_bench.add_argument("--batch", type=int, default=32)
    p_bench.add_argument("--batches", type=int, default=50)
    p_bench.add_argument("--warmup", default="auto", help='"auto" or an integer')
    p_bench.add_argument("--cuda-graph", default="auto", choices=["auto", "on", "off"])
    p_bench.add_argument("--no-flush-l2", dest="flush_l2", action="store_false")
    p_bench.add_argument("--no-lock-clocks", dest="lock_clocks", action="store_false")
    p_bench.add_argument("--json", action="store_true", help="print the Result as JSON")

    p_doc = sub.add_parser("doctor", help="is this machine fit to benchmark?")
    p_doc.add_argument("--recording", metavar="PATH", help="assess a recorded session instead")
    p_doc.add_argument("--json", action="store_true")

    p_fp = sub.add_parser("fingerprint", help="print the machine fingerprint")
    p_fp.add_argument("--recording", metavar="PATH")
    p_fp.add_argument("--json", action="store_true")
    p_fp.add_argument(
        "--check",
        action="store_true",
        help="report completeness instead; exit 1 if a required field is missing",
    )

    return parser


def _cmd_bench(args: argparse.Namespace) -> int:
    if not args.recording:
        print(
            "caliper bench: needs --recording <file> (a recorded device session), or a CUDA "
            f"host for the live launcher. target={args.target!r}",
            file=sys.stderr,
        )
        return 2
    warmup: str | int = args.warmup
    if warmup != "auto":
        try:
            warmup = int(warmup)
        except ValueError:
            print(
                f"caliper bench: --warmup must be 'auto' or an integer, got {args.warmup!r}",
                file=sys.stderr,
            )
            return 2
    try:
        result = api.bench(
            args.target,
            recording=Path(args.recording).read_text(),
            batch=args.batch,
            batches=args.batches,
            warmup=warmup,
            cuda_graph=args.cuda_graph,
            flush_l2=args.flush_l2,
            lock_clocks=args.lock_clocks,
        )
    except (ValueError, OSError) as exc:
        print(f"caliper bench: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(result.to_json())
    else:
        t = result.timing
        print(f"p50 {t['p50_us']:.3f} us/launch  (p10 {t['p10_us']:.3f}, p90 {t['p90_us']:.3f})")
        print(
            f"samples {t['n_samples']}  warmup-trimmed {t['n_warmup_to_steady']}  "
            f"dropped {t['invalidated_samples']}"
        )
        if result.flags:
            print("flags: " + ", ".join(result.flags))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    text = _recording_text(args)
    try:
        report = api.doctor(recording=text)
        rendered = api.doctor_text(recording=text)
    except (ValueError, OSError) as exc:
        print(f"caliper doctor: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2) if args.json else rendered)
    return int(report.get("exit_code", 2))


def _cmd_fingerprint(args: argparse.Namespace) -> int:
    try:
        recording = _recording_text(args)
        if args.check:
            report = api.fingerprint_check(recording=recording)
        else:
            machine = api.fingerprint(recording=recording)
    except (ValueError, OSError) as exc:
        print(f"caliper fingerprint: {exc}", file=sys.stderr)
        return 2

    if args.check:
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            state = "complete" if report["complete"] else "INCOMPLETE"
            print(f"fingerprint: {state}")
            for field in report["missing_required"]:
                print(f"  missing (required)    {field}")
            for field in report["missing_recommended"]:
                print(f"  missing (recommended) {field}")
        return 0 if report["complete"] else 1

    if args.json:
        print(json.dumps(machine, indent=2))
    else:
        for key, value in machine.items():
            print(f"{key:16} {value}")
    return 0


def _recording_text(args: argparse.Namespace) -> str | None:
    return Path(args.recording).read_text() if args.recording else None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the caliper CLI. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "bench":
        return _cmd_bench(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "fingerprint":
        return _cmd_fingerprint(args)
    parser.print_help(sys.stderr)  # pragma: no cover - argparse rejects unknowns first
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
