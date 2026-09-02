# Manual acceptance playbook (L5)

Run this **once per GPU** you can access. Record the results in
`docs/acceptance/reports/<arch>-<host>-<date>.md` from the [template](#report-template).
Tier-1 architectures must be all-PASS to release; Tier-2 archs are filed with
triage notes.

The playbook is split by environment (see `docs/plan.md` §0.5):

- **Playbook A — Colab, every reachable arch** (T4 / A100 / L4, + V100 if it
  appears): steps **1, 2, 3, 4, 5, 6, 7** *(the `ptxas -v` register/occupancy
  check only; the `ncu` sub-check is deferred)*, **8** *(unlocked tier only)*,
  **10, 11, 12, 13, 14**. Step 9 uses the **thermal** throttle fallback on a T4.
  `selftest` reports `coverage: reduced` — expected.
- **Playbook B — golden box, once** (a root A100 instance), plus the H100 pass:
  the full **step 7 with `ncu`** for all five corpus kernels, **step 8 locked
  tier**, **step 9** with `nvidia-smi -pl` power-cap induction, and the FR-4
  clock-lock acceptance. Re-do step 8 locked on the H100 too.

## Steps

| # | Step | Command | Expected result |
|---|------|---------|-----------------|
| 1 | Install clean | `python -m venv v && v/bin/pip install caliper-gpu` | no errors; `caliper --version` prints the released version |
| 2 | Doctor | `caliper doctor --json` | verdict present; every field cross-checks vs `nvidia-smi -q` (fill `docs/checklist_fingerprint.md`) |
| 3 | Selftest | `caliper selftest --full --json > r.json` | exit 0; 9/9 (or 8/8 + SKIP if no `nsys`); `caliper validate` accepts `r.json`'s shape |
| 4 | Duration linearity | `caliper selftest --full` — its O1 check sweeps `target_ns ∈ {1 µs … 10 ms}`; read the report's `o1_duration_linearity` entry | linear-fit slope ∈ [0.97, 1.03] |
| 5 | L2 flush A/B | `caliper bench corpus:o2` at `bytes = L2/2` with `--flush-l2` vs `--no-flush-l2`; then at `L2*4` both ways | small: ≥ 2× GB/s gap; large: < 5% gap |
| 6 | Bandwidth cross-check | build `nvbandwidth`; `./nvbandwidth -t device_to_device_memcpy_read_ce`; compare to O2 at matched size | within 5% |
| 7 | cuBLAS vs `ncu` | `caliper bench corpus:gemm --shape '{M:4096,N:4096,K:4096}' --dtype bf16 --json g.json`; `ncu --set full ...` | duration Δ ≤ 3%; registers exact; occupancy ± 0.05; TFLOP/s Δ ≤ 3% |
| 8 | Reproducibility | 10 fresh-process runs of a ≥ 100 µs kernel | locked CoV(p50) < 2%; unlocked < 5% (≥ 100 µs) and > the locked CoV |
| 9 | Throttle handling | `sudo nvidia-smi -pl <~60% TDP>`; `caliper bench corpus:o3 --json t.json`; restore | `throttle_reasons` non-empty; `invalidated_samples > 0`; run flagged |
| 10 | `do_bench` shim | edit a Triton tutorial to `from caliper import do_bench`; run it | runs unchanged; numbers within 3% of `caliper bench` p50 |
| 11 | Sweep + resume | `caliper sweep <spec> --parquet s.parquet`; kill mid-run; rerun with `--resume` | resumes; final Parquet passes `caliper validate` |
| 12 | Compare catches regression | `caliper compare --baseline tests/testdata/base.parquet --candidate tests/testdata/slow.parquet --fail-on-regression` | exit 1; prints the slowdown **and** the spill delta |
| 13 | Submit dry-run | `caliper submit s.parquet --dry-run --out bundle/` | `bundle/` has `manifest.json` + `rows.parquet` + `fingerprint.json`; `caliper validate bundle/` passes |
| 14 | Negative validate | `caliper validate` on `tests/testdata/over_peak_row.parquet`, `tests/testdata/bundle_missing_field/`, `tests/testdata/bundle_nonreproducing/`, `tests/testdata/bundle_slow_calibration/` | each exits 1 with a specific, correct message |

## Report template

Copy to `docs/acceptance/reports/<arch>-<host>-<date>.md`:

```md
# caliper acceptance report
- Arch / GPU: sm_XX / <card>        - Host: <box>        - Date: 2026-XX-XX
- Tier: 1 | 2                        - caliper: <version> (from PyPI)
- Tools present: nsys <v>, ncu <v>, nvbandwidth <sha>

| Step | Expected | Measured | Pass? | Notes |
|------|----------|----------|-------|-------|
| 1 install clean | released version | | | |
| 2 doctor fields | byte-exact vs nvidia-smi | | | checklist attached |
| 3 selftest --full | 9/9 (or 8/8 + SKIP) | | | report attached |
| 4 O1 linearity | slope ∈ [0.97, 1.03] | | | |
| 5 L2 flush A/B | small ≥ 2×, large < 5% | | | |
| 6 nvbandwidth | ± 5% | | | |
| 7 cuBLAS vs ncu | dur ≤ 3%, regs exact, occ ± 0.05, tflops ≤ 3% | | | table attached |
| 8 reproducibility | locked CoV < 2%, unlocked > locked | | | |
| 9 throttle | flagged + samples dropped | | | |
| 10 do_bench shim | runs, ± 3% | | | |
| 11 sweep + resume | resumes, valid parquet | | | |
| 12 compare regression | exit 1 + spill delta shown | | | |
| 13 submit dry-run | well-formed bundle | | | |
| 14 negative validate | 4/4 rejected correctly | | | |

## Deviations / triage
(none)  |  <describe, link issue>
```
