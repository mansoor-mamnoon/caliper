# Command line

```
caliper <command> [options]
caliper --version | --help
```

`tests/l0_unit/test_docs_match_code.py` checks this page documents every
subcommand the CLI registers.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | `doctor` not fit · `selftest` FAIL · `validate` INVALID · `compare --fail-on-regression` found a regression |
| `2` | usage / runtime error · `selftest` ERROR (including "no device") |

Most commands accept `--json` for a machine-readable report on stdout.

---

## `caliper bench [target] [--recording PATH] [--batch N] [--batches N] [--warmup W] [--cuda-graph {auto,on,off}] [--no-flush-l2] [--no-lock-clocks] [--json]`

Measure one kernel. Needs `--recording` a JSON Lines device session (the live
launcher is a CUDA-host stub). `target` may be `corpus:o1`..`corpus:o6` or a
`corpus:*` reference kernel; an unknown `corpus:*` target is rejected (exit 2).
`--warmup` is `"auto"` or an integer count of leading samples to trim. Non-JSON
output prints the p50 / p10 / p90, the sample / trim / drop counts, and any
flags.

## `caliper doctor [--recording PATH] [--json]`

Is this machine set up to produce trustworthy numbers? Exit `0` fit / `1` unfit
/ `2` no device. `--recording` assesses a recorded session instead of the live
device. Non-JSON output follows the honest-degradation wording (`FIT TO
BENCHMARK` / `... (reduced confidence)` / not fit).

## `caliper fingerprint [--recording PATH] [--json] [--check]`

Print the machine fingerprint. `--check` reports completeness instead and exits
`1` if a required field is missing.

## `caliper selftest [--full] [--json]`

Run the oracle self-test and print the Appendix-E report. Exit `0` PASS / `1`
FAIL / `2` ERROR. `--full` also runs O5 (cuBLAS) and the `nsys` cross-check.
With no device (and, until the on-device oracle runner lands, with a device but
every oracle skipped) the report is `ERROR`.

## `caliper validate PATH [--json]`

Check a results file (`.json` / `.jsonl` / `.parquet`) against the schema, or a
**bundle directory** against the shared results gate (schema + submission-strict
fields + roofline bound + determinism / calibration / arch consistency). Exit
`0` OK / `1` INVALID / `2` unreadable.

## `caliper sweep SPEC [--recordings DIR] [--parquet PATH] [--json-out PATH] [--resume]`

Expand a sweep spec YAML and run it into a results file, checkpointing after
every cell. `--recordings DIR` supplies one `<cell-key>.jsonl` per cell for the
replay path. `--resume` continues from the `<output>.state.jsonl` sidecar.
`--parquet` / `--json-out` override the spec's `output:` block.

## `caliper compare --baseline FILE --candidate FILE [--arch SM] [--threshold PCT] [--fail-on-regression] [--json]`

Diff two results files facet by facet. `--threshold` is an explicit timing noise
band **in percent** (e.g. `10`); it overrides the MAD-derived band but a
register-spill regression still fails the run. `--arch` restricts the comparison
to rows on that `sm_arch` (and warns on stderr if that matches nothing).
`--fail-on-regression` makes a timing or spill regression exit `1`. Non-JSON
output prints one line per facet (verdict, delta %, band %) plus the `ptxas` /
occupancy deltas and dropped configs for a moved facet.

## `caliper submit FILE... [--out DIR] [--repo DIR] [--dry-run] [--calibration MEASURED_US EXPECTED_US] [--json]`

Build a `caliper-results` submission bundle from one or more results files.
`--out DIR` writes `manifest.json` + `rows.parquet` + `fingerprint.json` there.
`--calibration` records the SKU calibration-GEMM p50 and its expectation in the
manifest. `--repo DIR` (a local `caliper-results` checkout) + no `--dry-run`
commits the bundle to a fresh branch there; push and PR stay manual. Exit `0` /
`2`.
