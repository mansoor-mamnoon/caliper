# Acceptance traceability

Every functional and non-functional requirement (`docs/plan.md` §1.4 / §1.5)
mapped to its evidence. **CI evidence** is what already runs on every push
(`cargo test`, `pytest -m "l0 or l1"`). **On-device evidence** is a step of the
[manual playbook](manual-playbook.md) or a GPU test tier (`l2`/`l3`/`l4`/`l6`),
run on Colab / a rented instance and filed under `reports/`.

Status legend: **CI** = covered by the no-GPU suite · **pending** = needs a GPU
run.

## Functional requirements

| FR | CI evidence | On-device evidence | Status |
|---|---|---|---|
| FR-1 `bench()` core timing | `cargo test pipeline::` (reduce → p10/p50/p90/MAD), `tests/l1_contract/test_bench.py` (replay → populated `Result`, schema-valid) | Playbook step 1, 4; `l2` duration oracle vs `nsys` (NFR-1) | pending |
| FR-2 steady-state warmup | `cargo test warmup::` (golden synthetic sequences), `cargo test --test bench` (`cold_ramp_run_is_trimmed_to_steady_state`) | Playbook step 4; `l2` kernel where `warmup=25` is > 10% low | pending |
| FR-3 arch-aware L2 flush | `cargo test pipeline::` (`flush_buffer_is_near_l2_not_a_fixed_constant`) | Playbook step 5 (flush A/B ratios) | pending |
| FR-4 clock control | `cargo test --test bench` (`a_hard_lock_error_degrades_to_an_unlocked_run`, `unlocked_run_with_throttling_is_flagged_and_cleaned`), `cargo test --test replay` (clock-lock flows) | Playbook step 9 (power-cap throttle) + step 8 locked; golden box | pending |
| FR-5 small-kernel batched | `cargo test graph::`, `cargo test --test bench` (`auto_graph_run_is_tagged_...`, `auto_graph_falls_back_to_eager_...`) | Playbook step 10 write-up delta; `l2` 5 µs oracle (NFR-2) | pending |
| FR-6 `launch_overhead_us` | `cargo test oracles::` (O4 analytic), `cargo test --test bench` (graph vs eager) | `l2` vs `nsys` API-call→kernel-start gap on ≥ 3 archs | pending |
| FR-7 ptxas / cuobjdump parsing | `cargo test ptxas_parse::` + `crates/caliper-core/tests/ptxas/` goldens, `tests/l0_unit/test_ptxas.py` | Playbook step 7 (`regs_per_thread` vs `ncu`, golden box) | pending |
| FR-8 occupancy | `cargo test occupancy::` vs `crates/caliper-core/tests/occupancy/reference.csv`, `tests/l0_unit/test_occupancy.py` | Playbook step 7 (`achieved` vs `ncu` ±0.05, golden box) | pending |
| FR-9 roofline | `cargo test roofline::` (peaks table w/ cited sources, `bound` classification, O2/O3 formulas), `tests/l0_unit/test_roofline.py` | Playbook step 6, 7 (`achieved_gbps` vs `nvbandwidth`, `achieved_tflops` vs `ncu`) | pending |
| FR-10 machine fingerprint | `cargo test fingerprint::`, `tests/l0_unit/test_fingerprint.py`, `tests/l1_contract/test_cli.py::test_fingerprint_*` | Playbook step 2 (byte-exact vs `nvidia-smi -q`, `checklist_fingerprint.md`) on ≥ 3 archs | pending |
| FR-11 `caliper doctor` | `cargo test doctor::`, `tests/l1_contract/test_cli.py` doctor fixtures (fit / constrained / throttling / no-device, exit codes) | Playbook step 2; induced ECC / MIG / persistence / background-load scenarios | pending |
| FR-12 `do_bench` shim | `tests/l1_contract/test_do_bench.py` (Triton-signature parity, quantiles, replay path) | Playbook step 10 (unmodified Triton tutorial, ±3% vs `caliper.bench`) | pending |
| FR-13 `sweep` + spec + `Grid` + `--resume` + autotune cache | `cargo test spec::` + golden `appendix_d`, `cargo test autotune::`, `tests/l1_contract/test_sweep.py`, `tests/l6_e2e/test_autotune_cache.py` (skips off-GPU) | Playbook step 11 (kill + `--resume`); `l6` 20-cell corpus sweep to a valid Parquet | pending |
| FR-14 reference kernel corpus | `cargo test roofline::corpus_spec` (all 5 arms), `tests/l0_unit/test_corpus_kernels.py` (import-guard, roofline math, `NotImplementedError` off-GPU, `TRITON_PIN` + `SOURCE_HASH`) | `l6` all 5 kernels emit valid rows on SM80/86/89/90; `check_numerics` allclose vs baseline | pending |
| FR-15 `caliper compare` | `cargo test thresholds::` (band / verdict / spill / dropped-config), `tests/l0_unit/test_thresholds.py`, `tests/l1_contract/test_cli.py` compare cases | Playbook step 12 (the same fixtures; exit 1 + slowdown + spill delta) | **CI** |
| FR-16 `caliper submit` + results repo + CI | `cargo test submit::` (manifest, `validate_bundle`, all 4 rejection classes, tamper-resistant), `tests/l0_unit/test_submit.py` -- the bundle-gate *logic* | Playbook step 13 (`submit --dry-run` on real rows → `validate bundle/`); the `results-repo/` PR gate activates when that repo is split out | **CI** (logic) |
| FR-17 `caliper selftest` | `cargo test selftest::` (Appendix-E assembly, `PASS`/`FAIL`/`ERROR`, coverage), `tests/l0_unit/test_selftest.py`, `tests/l1_contract/test_cli.py::test_selftest_*` | Playbook step 3, 4; healthy Tier-1 exit 0 + induced-fault exit 1 | pending |
| FR-18 `caliper validate` | `cargo test schema::validate`, `tests/l0_unit/test_submit.py` (over-peak + 3 bad bundles), `tests/l1_contract/test_cli.py` validate cases | Playbook step 14 (4/4 rejected with a specific message) | **CI** |
| FR-19 `caliper fingerprint` | `tests/l1_contract/test_cli.py::test_fingerprint_equals_the_bench_result_machine_block` (asserts the equality FR-19's AC names) | Playbook step 2 | **CI** |

## Non-functional requirements

| NFR | CI evidence | On-device evidence | Status |
|---|---|---|---|
| NFR-1 timing accuracy ≥ 50 µs (≤ 3% vs `nsys`) | — (analytic formulas covered by `cargo test oracles::`) | Playbook step 4, 7 | pending |
| NFR-2 small-kernel ≤ 10% (5 µs oracle, batched) | — | `l2` 5 µs duration oracle | pending |
| NFR-3 bandwidth ≤ 5% vs `nvbandwidth`; compute ≤ 3% vs `ncu` | `cargo test roofline::` (formulas) | Playbook step 6, 7 | pending |
| NFR-4 `regs_per_thread` exact vs `ptxas`; occupancy ±0.05 | `cargo test occupancy::` vs reference table | Playbook step 7 (golden box `ncu` confirm) | pending |
| NFR-5 reproducibility (locked < 2%, unlocked < 5% / < 8%) | — | Playbook step 8 on every Colab arch + golden box + H100 | pending |
| NFR-6 `bench()` wall overhead ≤ 6 s for a 200 µs kernel | — | timed in the acceptance notebook | pending |
| NFR-7 platform matrix (Linux x86-64, CUDA 12.1–12.6, Py 3.10–3.12, SM70–SM120) | CI matrix (Py 3.10/3.11/3.12) | one report per Tier-1 arch under `reports/` | pending |
| NFR-8 no root; NVML user path; graceful degrade | `cargo test bench::` permission-denied → tagged `Result`, not raised | Playbook (run as non-root); step 9 | pending |
| NFR-9 L0 suite: 0 flakes over 100 CI runs | 100-run flake check in CI | — | pending |
| NFR-10 `pip install caliper-gpu` clean; `doctor` runs immediately | — | fresh-Colab install from PyPI at release (`release.yml` DoD) | pending |

## How to close a row

Run [`notebooks/acceptance.ipynb`](../../notebooks/acceptance.ipynb) on the
target GPU (Playbook A) or the golden box (Playbook B). It writes
`reports/<arch>-<host>-<date>.md` and `selftest-<arch>.json`; commit both and
flip the row to the arch it passed on. A deviation is filed as an issue and
noted in the report's *Deviations / triage* block, then tracked to closure in
[`triage.md`](triage.md). Tier-2 archs use the reduced bar in
[`tier2.md`](tier2.md) and are not release-blocking.

When every row here points at a passing test or a filled playbook step, the
release checklist in [`../../RELEASING.md`](../../RELEASING.md) applies.
