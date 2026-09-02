"""Command-line entry point for caliper.

Wired up so far: ``bench`` (recorded-session path), ``doctor``, ``fingerprint``,
``selftest``, ``validate``, ``sweep``, ``compare``, plus ``--version`` /
``--help``. Most commands take ``--json`` for machine-readable output. The
submit command is added as its supporting code lands.

Exit codes: 0 success; 1 "not fit" (``doctor``) / "FAIL" (``selftest``) /
"regression" (``compare --fail-on-regression``) / "INVALID" (``validate``);
2 usage / runtime error / "ERROR" (``selftest``, including no device).
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

    p_st = sub.add_parser("selftest", help="run the oracle suite and report")
    p_st.add_argument(
        "--full",
        action="store_true",
        help="also run O5 (cuBLAS) and the nsys cross-check",
    )
    p_st.add_argument("--json", action="store_true", help="print the Appendix-E report as JSON")

    p_val = sub.add_parser("validate", help="check a results file against the schema")
    p_val.add_argument("path", metavar="FILE", help="a .json / .jsonl / .parquet results file")
    p_val.add_argument("--json", action="store_true", help="print the report as JSON")

    p_cmp = sub.add_parser("compare", help="diff two results files for regressions")
    p_cmp.add_argument("--baseline", metavar="FILE", required=True, help="the reference dataset")
    p_cmp.add_argument("--candidate", metavar="FILE", required=True, help="the dataset to check")
    p_cmp.add_argument("--arch", metavar="SM", help="only compare rows on this sm_arch")
    p_cmp.add_argument(
        "--threshold",
        type=float,
        metavar="PCT",
        help="explicit timing noise band in percent (e.g. 10); overrides the "
        "MAD-derived band. A register-spill regression still fails the run.",
    )
    p_cmp.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit 1 if any facet is a timing or register-spill regression",
    )
    p_cmp.add_argument("--json", action="store_true", help="print the full report as JSON")

    p_sw = sub.add_parser("sweep", help="run a sweep spec into a results file")
    p_sw.add_argument("spec", metavar="SPEC", help="a sweep spec YAML file")
    p_sw.add_argument("--recordings", metavar="DIR", help="dir of <cell-key>.jsonl recordings")
    p_sw.add_argument("--parquet", metavar="PATH", help="override the spec's parquet output")
    p_sw.add_argument("--json-out", metavar="PATH", help="also write a JSON results file")
    p_sw.add_argument(
        "--resume", action="store_true", help="continue from the .state.jsonl sidecar"
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


def _cmd_selftest(args: argparse.Namespace) -> int:
    report = api.selftest(full=args.full)
    if args.json:
        print(json.dumps(report, indent=2))  # pure Appendix-E, no extra keys
    else:
        print(f"caliper selftest: {report['result']}  (coverage: {report['coverage']})")
        for check in report["checks"]:
            print(f"  {check['status']:5}  {check['name']:24} {check['detail']}")
        if report["not_validated"]:
            print(f"  not validated here: {', '.join(report['not_validated'])}")
    return api.SELFTEST_EXIT_CODE.get(report["result"], 2)


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        report = api.validate_records(args.path)
    except (ValueError, OSError, ImportError) as exc:
        print(f"caliper validate: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for entry in report["problems"]:
            for problem in entry["problems"]:
                print(f"  row {entry['row']}: {problem}")
        state = "OK" if report["ok"] else "INVALID"
        print(f"{state}: {report['n']} record(s), {report['n_invalid']} invalid")
    return 0 if report["ok"] else 1


def _fmt_deltas(d: dict[str, object]) -> list[str]:
    """The non-zero numeric deltas of a delta-block as ``name +/-N`` strings
    (``+.3g`` for floats, so an occupancy delta reads cleanly)."""
    out: list[str] = []
    for name, value in d.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value != 0:
            out.append(f"{name} {value:+d}")
        elif isinstance(value, float) and value != 0.0:
            out.append(f"{name} {value:+.3g}")
    return out


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        report = api.compare(
            args.baseline,
            args.candidate,
            arch=args.arch,
            threshold=(args.threshold / 100.0) if args.threshold is not None else None,
            fail_on_regression=args.fail_on_regression,
        )
    except (ValueError, OSError, ImportError) as exc:
        print(f"caliper compare: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        s = report["summary"]
        if args.arch is not None and s["facets"] == 0:
            print(
                f"caliper compare: no rows on --arch {args.arch}; nothing compared",
                file=sys.stderr,
            )
        for facet in report["facets"]:
            k = facet["key"]
            name = " ".join(
                str(part)
                for part in (
                    k["kernel"] or "?",
                    k["impl"],
                    k["dtype"],
                    k["shape"],
                    k["layout"],
                    k["arch"],
                )
                if part
            )
            verdict = facet["verdict"]
            delta = facet["delta"]
            band = facet["band"]
            if delta is None:
                print(f"  {verdict:12} {name}  (only on one side)")
            else:
                print(f"  {verdict:12} {name}  {delta * 100:+.1f}% (band +/-{band * 100:.1f}%)")
            interesting = facet["spill_regression"] or verdict in ("regression", "improvement")
            if interesting:
                ptxas = _fmt_deltas(facet["ptxas_delta"])
                if ptxas:
                    print(f"               ptxas: {', '.join(ptxas)}")
                occ = _fmt_deltas(facet["occupancy_delta"])
                if occ:
                    print(f"               occupancy: {', '.join(occ)}")
            if facet["autotune_configs_dropped"]:
                dropped = ", ".join(facet["autotune_configs_dropped"])
                print(f"               autotune configs dropped: {dropped}")
        print(
            f"{s['facets']} facet(s): {s['regressions']} regression(s), "
            f"{s['improvements']} improvement(s), {s['within_noise']} within noise, "
            f"{s['spill_regressions']} spill regression(s), "
            f"{s['configs_dropped']} with dropped configs"
        )
        if s["only_in_baseline"] or s["only_in_candidate"]:
            print(
                f"  unmatched: {s['only_in_baseline']} only in baseline, "
                f"{s['only_in_candidate']} only in candidate"
            )
    if args.fail_on_regression:
        return int(report["exit_code"])
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    try:
        grid = api.sweep(
            Path(args.spec),
            recordings_dir=args.recordings,
            parquet=args.parquet,
            json_out=args.json_out,
            resume=args.resume or None,
        )
    except (ValueError, OSError, NotImplementedError) as exc:
        print(f"caliper sweep: {exc}", file=sys.stderr)
        return 2
    print(f"caliper sweep: {len(grid)} cell(s) measured")
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
    if args.command == "selftest":
        return _cmd_selftest(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "compare":
        return _cmd_compare(args)
    if args.command == "sweep":
        return _cmd_sweep(args)
    parser.print_help(sys.stderr)  # pragma: no cover - argparse rejects unknowns first
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
