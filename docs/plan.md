# caliper — v0.3.0 Build Plan (4 weeks, solo)

A step-by-step plan to build `caliper`, a correct-by-default GPU kernel benchmark
library, to a fully-shipped v0.3.0 in four weeks — engineered so that every claim
it makes is independently verifiable against an external oracle, and so that **you**
can exhaustively validate it on every GPU you can get your hands on.

---

## 0. How to use this document

- **Section 1** is the frozen spec. Nothing here gets cut in the 4 weeks. If time
  pressure hits, use the ranked descope levers in Section 6 — and only those.
- **Section 2** is the architecture, chosen specifically so components are testable
  in isolation without a GPU where possible.
- **Section 3** is the testability system: oracle kernels with first-principles
  expected values, external cross-checks (Nsight Systems / Nsight Compute /
  nvbandwidth / cuBLAS), the `caliper selftest` command, and your per-GPU manual
  acceptance playbook.
- **Section 4** is the day-by-day execution plan. Each task has a Definition of
  Done and a "how to verify" line. Work top to bottom.
- **Section 5** is the single success checklist for the release.
- **Section 7** appendices contain the exact schemas, oracle kernel sketches, and
  report templates you will need.

**Assumptions this plan is built on** (if any is false, adjust before starting):

1. ~Full-time focus for 4 weeks (~120–130 working hours).
2. Dev happens on a **MacBook Pro (Apple Silicon)** for all non-GPU code, and on
   **Google Colab** (A100 / L4 / T4) for GPU validation. **No local CUDA.** The
   Mac cannot run CUDA / Triton / `nsys` / `ncu` / NVML — this is accounted for in
   the plan (§0.5), not a gap.
3. **$100** cloud budget, spent on targeted rentals at acceptance (Week 4):
   primarily a root "golden box" (a cheap Lambda A100) for the clock-lock +
   `ncu` + power-cap validation Colab can't do, plus one H100 pass. See §0.5 and
   Appendix H.
4. You are comfortable with CUDA C, Triton, Python packaging, and pytest.

**Week 3 is the crunch week** (reference kernel corpus) — and Colab session drops
make it riskier. Protect it: land `sweep --resume` early, checkpoint every cell.

**Read §0.5 before anything else** — it is the resource-adjusted execution model
and it overrides specific rows elsewhere in this document.

---

## 0.5 Resource-adjusted execution model (READ FIRST — overrides where noted)

**Your resources:** MacBook Pro (Apple Silicon GPU — Metal only, no CUDA), Google
Colab subscription (T4 / A100-40GB / L4 depending on availability), $100 cloud
budget.

**Consequence:** the *product scope (FR-1…FR-19) does not change.* What changes is
**where you build** and **how each requirement is validated.** Two facts drive
everything:

1. **The MacBook cannot run CUDA, Triton, `ptxas`, `nsys`, `ncu`, or NVML.** It is
   your **pure-core dev machine**: everything under `_internal/` (stats, warmup,
   roofline, ptxas_parse, occupancy, thresholds, schema, spec, fingerprint_model),
   all L0 tests, all L1 fixture tests, the CLI, docs, packaging, and CI-CPU config
   are written and tested here with zero GPU. That is ~40% of the code and 100% of
   the deterministic test surface.
2. **Colab cannot lock GPU clocks, cannot grant Nsight Compute counter access,
   cannot set a power cap, and has no persistent runner.** Colab validates
   everything *else*: all timing oracles, the `nsys` timing cross-check (NFR-1
   intact), `ptxas -v` parsing, the occupancy API, cuBLAS/roofline, `sweep`, the
   corpus, `compare`, the e2e pipeline, and the **unlocked-clocks** reproducibility
   tier.

The four things Colab can't validate — **(a) FR-4 clock-lock path, (b) the `ncu`
cross-check, (c) O6 power-cap throttle induction, (d) locked-clocks
reproducibility** — are validated **once** on a rented root "golden box," and are
arch-independent logic (not per-GPU behavior), so a single thorough pass is sound.

### The build loop (Mac ⇄ Colab)

```
Mac:    edit caliper-core (+ tests) → cargo test && pytest -m "l0 or l1" → commit/push
Colab (notebooks/dev.ipynb):  git pull → pip install -e . → pytest -m "l2 or l6" → caliper selftest
```

- `make sync` = push branch + print the one-liner to paste into Colab.
- `notebooks/` (committed — and doubles as user-facing "run caliper on Colab" docs,
  a real adoption channel since most people first meet a GPU through Colab):
  `dev.ipynb` (pull/install/L2+L6), `selftest.ipynb`, `corpus_sweep.ipynb`
  (sweep → push rows), `acceptance.ipynb` (the Colab half of the playbook).
- Colab Pro+ background execution runs `corpus_sweep.ipynb` unattended; on plain
  Pro keep sessions < 90 min and checkpoint (`sweep --resume` earns its keep).

### Colab GPU reachability (plan around this)

| Colab GPU | Arch | corpus dtypes usable | Notes |
|-----------|------|----------------------|-------|
| T4 | SM75 | fp32, fp16 (no bf16 TC, no fp8) | free tier; throttles easily → the thermal-throttle O6 fallback |
| A100-40GB | SM80 | + bf16, tf32 | Pro/Pro+; the workhorse |
| L4 | SM89 | + fp8_e4m3 / e5m2 | Pro/Pro+; datacenter Ada |
| V100 | SM70 | fp32, fp16 | sometimes; Tier-2 |

H100 (SM90), consumer 3090/4090/5090, and any AMD card are **not** on Colab.

### The $100, allocated (spent at acceptance — Week 4)

| Spend | GPU / where | Hours | Purpose (closes) |
|-------|-------------|-------|------------------|
| ~$10 | **Lambda Cloud on-demand A100-40GB** (golden box — reliably grants root + performance counters + `nvidia-smi -lgc`) | 8 | **FR-4 lock path, `ncu` L3 (all 5 corpus kernels), O6 power-cap throttle, locked-clocks NFR-5, O7 calibration.** |
| ~$16 | **H100** (SM90), RunPod | 8 | The Hopper Tier-1 arch; 2nd `nsys` cross-check arch; corpus rows. |
| ~$5 | **RTX 4090** (consumer SM89), Vast — *verify it grants clock-lock + counters first* | ~8 | Consumer-Ada datapoint; a 2nd locked-clock check. |
| ~$8 | **RTX 3090** (SM86) spot, Vast | ~1 | Ampere-consumer cell (Tier-2 → Tier-1 if it passes). |
| ~$10 | **MI300X** (CDNA3), RunPod | ~1 | Tier-2 "runs & emits a valid row" ROCm check. |
| **~$51** | **reserve** | — | Re-runs after bug fixes, a 2nd H100 pass, spillover. Do **not** pre-spend. |

If you are on Colab **Pro** (not Pro+): consider spending ~$50 of the budget on
**one month of Pro+** for the build month (reliable A100 + background execution),
compressing rentals to golden-box + H100 only (~$26). Log the choice in
`docs/DESCOPE.md`.

### What this changes elsewhere in the doc

| Section | Override |
|---------|----------|
| **§4 throughout** | "the dev GPU" / "on the dev GPU" now means **a Colab runtime (A100 unless noted)**. Any step needing `ncu`, clock locking, or a power cap is **deferred to the Week-4 golden-box pass**. |
| §1.3 NFR-4 | Registers validated against **`ptxas -v`** (the compiler's own static report) on every arch; **confirmed** against `ncu` on the golden box only. Achieved occupancy: vs the reference model on Colab; vs `ncu` on the golden box. |
| §1.3 NFR-5 | Colab archs use the **unlocked** tier (CoV < 5% for ≥ 100 µs, < 8% for 10–100 µs). Locked tier (< 2%) validated on the golden box + H100 + any Colab arch where NVML locking happens to succeed. |
| §1.7 | Each Tier-1 arch tagged **Colab-full** or **rental-pass** — see the revised matrix. |
| §3.1 / §3.3 | Each cross-check tagged **[Colab]** (every arch) or **[golden box]** (once). O6 gets a **thermal-throttle** fallback on T4. |
| §3.6 | `ci-gpu` = `notebooks/dev.ipynb` run on Colab on demand (schedulable on Pro+), **not** a self-hosted GitHub runner. |
| §3.7 | Split into **Playbook A (Colab, every arch)** and **Playbook B (golden box, once)**. |
| §4 Week 4 | D4 = Colab acceptance passes (T4/A100/L4) + golden-box pass + H100 pass; D5 = 3090/MI300X best-effort + release. |
| §6 | New top risks: Colab session limits, no clock lock, budget overrun. |
| Appendix H | Replaced. |

### Honest-degradation behavior (this is a feature — ship it)

`caliper doctor` and `caliper selftest` must detect a constrained environment
(clock lock denied, counters restricted, `nvidia-smi -pl` denied, ephemeral
session) and say so plainly:

```
environment: CONSTRAINED (Colab-like)
  · GPU clock locking: DENIED by driver → results tagged `clocks-unlocked`, reduced confidence
  · performance counters: RESTRICTED → ncu-class metrics unavailable; using ptxas + occupancy API
  · verdict: FIT TO BENCHMARK (reduced confidence) — good for relative comparisons on this box,
    not for cross-machine absolute claims
```

`selftest` report `coverage` becomes `reduced`, with an explicit
`not_validated: ["clock_lock", "ncu_crosscheck", "powercap_throttle"]` list. A
`reduced` selftest is still a **PASS** if every runnable check passes — it just
advertises its own limits. caliper is then trustworthy *about its own
trustworthiness*, which is on-brand.

---

## 0.6 Language & implementation policy (READ FIRST — overrides where noted)

**All internal logic is Rust. C++/CUDA only where the GPU toolchain forces it.
Python only for the public surface.** This supersedes any "`_internal/*.py`"
phrasing elsewhere in the document — those modules are Rust modules in
`crates/caliper-core`.

| Concern | Language | Where |
|---|---|---|
| Result schema, statistics (percentiles, MAD, CoV), steady-state detection, roofline model, `ptxas`/`cuobjdump` parsing, occupancy model, regression threshold model, sweep-spec expansion, fingerprint shaping, `validate` | **Rust** | `crates/caliper-core` (pure `rlib`, `#![forbid(unsafe_code)]`, `cargo test`) |
| Python bindings for the core | **Rust (PyO3)** | `crates/caliper-ffi` → built by **maturin** into `caliper._core` |
| Kernel launch, CUDA events & graphs, L2-flush buffer, NVML clock lock / throttle reasons, device info, occupancy-API call | **Rust** via `cudarc` + `nvml-wrapper` (FFI) | `crates/caliper-gpu` (feature-gated; only compiles where a CUDA driver is present) |
| The on-device oracle kernels (O1–O7) and any unavoidable CUDA-*runtime* interop | **CUDA C++** (`.cu`), built with `cc`/`nvcc` | `crates/caliper-gpu/kernels/` |
| Public API (`bench`, `sweep`, `compare` entry points), CLI, the Triton `do_bench` shim, `sweep` orchestration glue, Parquet I/O wiring, notebooks, packaging | **Python** | `python/caliper/` |

**Consequences for the rest of the plan:**

- **L0 = `cargo test` + the Python tests that exercise the core through the
  bindings.** Both run with no GPU, both gate merge in `ci-cpu`. The "pure core is
  100% testable without hardware" property is unchanged; it now has two test
  layers instead of one.
- **Ports** (§2.3): the `real`/`fixture`/`record` split still applies, but to the
  Rust GPU layer. Fixtures are recorded device responses replayed in Rust unit
  tests; `caliper-core` itself has no ports because it touches no hardware.
- **Wheels are shipped.** `pip install caliper-gpu` installs a prebuilt wheel with
  the compiled core — **users and Colab do not need a Rust toolchain.** Only
  building from source (`pip install -e .`) needs `rustup`.
- **CI** (§3.6) gains a Rust lane: `cargo fmt --check`, `cargo clippy -D
  warnings`, `cargo test`.
- Day tasks in §4 that name `_internal/foo.py` mean `crates/caliper-core/src/foo.rs`
  plus a thin `caliper-ffi` export and, where a Python-facing helper is useful, a
  wrapper in `python/caliper/`. The task intent (what to build, its acceptance
  criteria) is unchanged.

---

## 1. Product spec — v0.3.0 (frozen)

### 1.1 What caliper v0.3.0 is / is not

**Is:**

- A Python library + CLI that measures a single GPU kernel (Triton, raw CUDA,
  CUDA-graph-captured, or a `torch` callable) *correctly by default*: steady-state
  warmup, arch-aware L2 flushing, clock locking, throttle detection & sample
  invalidation, wall-vs-GPU-event time separation, launch-overhead isolation,
  small-kernel batched measurement, a latency distribution (not a mean), and a
  roofline / occupancy / register-spill readout.
- A `sweep` runner that produces a schema-stable Parquet dataset over a
  kernel × dtype × shape × layout × autotune-config matrix.
- A `compare` command that flags variance-aware performance regressions between
  two datasets and attributes them (ptxas/occupancy delta, autotune-config drop).
- A `submit` flow + `caliper-results` repo + PR-validation CI for community
  benchmark rows.
- A `selftest` command that validates a caliper install against on-device oracles.

**Is not (explicitly out of scope for v0.3.0 — these are v0.4+):**

- The `caliper-hud` website / dashboard / regression-watch bot.
- `ptxas` / CUDA / driver *version* axes in `sweep` (v0.3 sweeps the **Triton
  version** axis only; other toolchain fields are *recorded* but not swept).
- SM100 (B200), CDNA4, Windows, Jetson, multi-GPU / NCCL, ARM hosts.
- PTX/SASS structural diffing beyond the numeric ptxas fields.
- `git bisect` driver.
- First-class ROCm (v0.3 requirement is only "runs and produces a valid row on
  MI300X"; accuracy targets are not enforced on CDNA).
- Any hosted service, auth, or database (results are flat Parquet in git).

### 1.2 Functional requirements

Each FR has an ID, a statement, and **acceptance criteria (AC)** that are
objectively checkable. "nsys" = Nsight Systems kernel duration; "ncu" = Nsight
Compute; tolerances are defined in §1.3.

| ID | Requirement | Acceptance criteria |
|----|-------------|---------------------|
| **FR-1** | `bench()` core timing: CUDA-event GPU time as p10/p50/p90 + MAD, plus wall p50; configurable sample count; returns a `Result` (Appendix B). | On the duration oracle at 200 µs: `p50` within NFR-1 of nsys; `mad_us` populated; two fresh-process runs have CoV(p50) < NFR-5. `Result` validates against schema. |
| **FR-2** | Steady-state warmup: `warmup="auto"` ramps until the running p50 stabilizes within a tolerance; reports `n_warmup_to_steady`. Fixed `warmup=N` also supported. | (L0) Deterministic decision on synthetic timing sequences (golden tests). (L2) On a kernel where `warmup=25` is >10% below nsys, `warmup="auto"` is within NFR-1. `n_warmup_to_steady` is reported and > 25 in that case. |
| **FR-3** | Arch-aware L2 flush: flush buffer sized from the queried device L2 size; executed between samples; toggle `--flush-l2/--no-flush-l2`. | Buffer bytes == queried L2 size ± one allocation granularity. On the stream oracle at a transfer that fits in L2: GB/s with flush-off ≥ 2× GB/s with flush-on. At 4× L2 size: the two differ by < 5%. |
| **FR-4** | Clock control: lock SM+mem clocks via NVML; poll `nvmlDeviceGetCurrentClocksThrottleReasons` before/during/after; drop and count throttled samples; when locking is unavailable, tag `clocks-unlocked` and never present the run as trusted. | With power limit lowered to force throttle: `throttle_reasons` non-empty, `invalidated_samples > 0`, run flagged. With clocks locked vs default on the same kernel: CoV(p50) strictly lower locked. Permission-denied path returns a tagged `Result`, does not raise. |
| **FR-5** | Small-kernel batched measurement: time N back-to-back launches between two events; optional CUDA-graph capture; auto-engage when single-launch time < threshold. | On the 5 µs duration oracle: batched `p50` within NFR-2 of the analytic value. A single-synced measurement of the same kernel is demonstrably ≥ 8× too high (documented in the writeup). Graph vs non-graph `launch_overhead_us` differ in the expected direction. |
| **FR-6** | `launch_overhead_us`: isolated via the single-instruction oracle and the graph-replay delta. | Within NFR of the nsys "API call → kernel start" gap on ≥ 3 archs. Under graph replay the measured per-kernel launch cost drops below 1 µs. |
| **FR-7** | ptxas/cuobjdump parsing: `regs_per_thread`, static & dynamic smem, spill load/store bytes, local & stack bytes. | (L0) Golden-file tests pass for 6 real `ptxas -v` captures (with-spill, no-spill, dynamic-smem, multiple kernels per module, HIP `-v`, and a malformed sample that must raise a typed error). (L2) `regs_per_thread` exactly matches ncu on all 5 corpus kernels. |
| **FR-8** | Occupancy: theoretical via the CUDA occupancy API; achieved reported alongside. | (L0) Theoretical matches a checked-in reference table (CUDA occupancy calculator values) for 10 (arch, regs, smem, block) tuples. (L2) `achieved_occupancy` within NFR-4 of ncu on all 5 corpus kernels. |
| **FR-9** | Roofline: per-arch, dtype-aware tensor-core / FP peaks table + measured HBM bandwidth; emits `achieved_tflops`, `achieved_gbps`, `arithmetic_intensity`, ridge point, and `bound ∈ {compute, memory, latency, unknown}`. | fma oracle ≥ 90% of the documented FP32 FMA peak and classified `compute`. stream oracle ≥ 70% of measured HBM and classified `memory`. cuBLAS 4096³ bf16: `achieved_tflops` within NFR-3 of ncu. Every peaks-table cell has a cited source in code. |
| **FR-10** | Machine fingerprint: full field set (Appendix B, `machine`), attached to every `Result`. | Every field byte-exact against `nvidia-smi -q` / NVML on ≥ 3 archs (checklist). Identical across repeated runs on the same box. |
| **FR-11** | `caliper doctor`: fitness verdict + per-check detail + `--json` + exit codes (0 fit / 1 not fit / 2 error). | Correctly flags each induced scenario: unlockable clocks, active throttle, ECC on, MIG enabled, persistence off, background GPU load above threshold. Exit codes correct for each. |
| **FR-12** | `do_bench` shim: `from caliper import do_bench` with a Triton-compatible signature (`fn, warmup, rep, grad_to_none, quantiles, return_mode`), returns milliseconds. | Drops into an unmodified Triton `03-matrix-multiplication.py` tutorial with only the import changed; runs; result within NFR-1 of `caliper.bench(...).p50`. |
| **FR-13** | `sweep()` + YAML spec + `Grid` + `.to_parquet()/.to_json()` + `--resume`; each autotune config timed separately; per-config timings cached keyed by `{sku, driver, cuda, ptxas, triton, torch, kernel_source_hash}`. | Runs a 3-kernel × 3-dtype × 5-shape matrix to a schema-valid Parquet. Killed mid-run and restarted with `--resume`: completes without redoing finished cells. Adding one autotune config re-times only that config (cache hit on the rest). |
| **FR-14** | Reference kernel corpus v0: `gemm`, `attention_fwd`, `attention_bwd`, `rmsnorm`, `softmax` — each with a Triton implementation **and** a vendor baseline (cuBLAS / cuDNN / `F.scaled_dot_product_attention`). Named shape libraries: `llm-7b`, `llm-70b`, `square-pow2`, `prime-odd`. | All 5 kernels + baselines run and produce valid rows on SM80/86/89/90. Each kernel's source is pinned to an upstream commit and content-hashed. Shape libraries resolve to documented (M,N,K)/(B,H,S,D) tuples. |
| **FR-15** | `caliper compare`: variance-aware threshold (per-facet noise band from historical MAD, or explicit `--threshold PCT`); prints per-facet delta + ptxas/occupancy delta + "autotune configs dropped" flag; `--fail-on-regression` → exit 1. | Detects an injected 10% slowdown and an injected register-spill regression; the spill delta is shown. Does **not** fire on a within-noise difference. Exit code contract holds. |
| **FR-16** | `caliper submit` + `caliper-results` repo + validation CI: bundle = rows + fingerprint + calibration-kernel result + caliper version; `--dry-run` builds the bundle without pushing; CI validates schema + roofline sanity bounds + determinism repeat + calibration-kernel clock check. | `--dry-run` emits a well-formed bundle. `caliper validate` **rejects**: a row claiming > 100% of dtype peak, a row missing a required field, a bundle whose determinism repeat disagrees > tolerance, a bundle whose calibration kernel is > X% slow. A clean bundle passes CI. |
| **FR-17** | `caliper selftest [--full] [--report PATH]`: runs the oracle suite + doctor invariants + a short reproducibility check on the current device; `--full` additionally cross-checks against `nsys` if present. Exit 0 all-pass / 1 any-fail. | On a healthy Tier-1 GPU: exit 0, report validates against the selftest schema (Appendix E). With an induced fault (throttle): the relevant check fails and exit is 1. |
| **FR-18** | `caliper validate <PATH...>`: standalone schema + sanity check for any Parquet/JSON caliper output. Exit 0 / 1. | Passes on all corpus outputs; fails with a specific message on each of the FR-16 rejection cases. |
| **FR-19** | `caliper fingerprint [--json]`: prints the machine fingerprint block alone. | Output equals the `machine` block embedded in a `bench` `Result` on the same box. |

### 1.3 Non-functional requirements (accuracy & quality targets)

| ID | Target |
|----|--------|
| **NFR-1** | Timing accuracy, kernels ≥ 50 µs: `|caliper.p50 − nsys.dur| / nsys.dur ≤ 0.03`. |
| **NFR-2** | Small-kernel accuracy, 5 µs duration oracle, batched mode: relative error ≤ 0.10. |
| **NFR-3** | Bandwidth: stream oracle `achieved_gbps` within 5% of `nvbandwidth` device-to-device for the same transfer size. Compute: cuBLAS 4096³ `achieved_tflops` within 3% of ncu; fma oracle within 5% of the documented FMA-peak fraction. |
| **NFR-4** | `regs_per_thread` exactly equals the **`ptxas -v`** static report on every arch (the compiler's own number); additionally **confirmed** to equal `ncu` `launch__registers_per_thread` on the golden box. `achieved_occupancy` within ±0.05 of the reference occupancy model on every arch; confirmed within ±0.05 of `ncu` `sm__warps_active` on the golden box. |
| **NFR-5** | Reproducibility. **Locked-clocks tier** — CoV(p50) < 2% across 10 fresh-process runs (kernel ≥ 100 µs) — validated on the golden box + H100. **Unlocked-clocks tier** (Colab default) — CoV(p50) < 5% for ≥ 100 µs, < 8% for 10–100 µs — validated on every Colab arch. On unlocked runs, `selftest` records the observed SM-clock spread. Where both are measured, unlocked CoV > locked CoV. |
| **NFR-6** | `bench()` wall-time overhead: a 200 µs kernel completes a default measurement in ≤ 6 s. |
| **NFR-7** | Platform: Linux x86-64; CUDA 12.1–12.6; Python 3.10–3.12; NVIDIA SM70–SM120. ROCm 6.x = "runs & emits a valid row" only. |
| **NFR-8** | No root required for any core path. Clock locking uses the NVML user path; degrades gracefully (tagged, not fatal) when the driver denies it. |
| **NFR-9** | The L0 (pure, no-GPU) test suite has zero flakes over 100 consecutive CI runs. |
| **NFR-10** | `pip install caliper-gpu` in a fresh venv on a CUDA box succeeds with no manual steps; `caliper doctor` runs immediately after. |

### 1.4 Public Python API contract

```python
# caliper/__init__.py  — the entire stable surface for v0.3.0

def bench(
    fn: Callable[[], Any],
    *,
    flush_l2: bool = True,
    lock_clocks: bool = True,
    warmup: Literal["auto"] | int = "auto",
    min_samples: int = 100,
    max_samples: int = 2000,
    cuda_graph: Literal["auto", "on", "off"] = "auto",
    repeat: int = 1,                 # fresh measurement passes; >1 adds cross-pass CoV
    roofline: RooflineSpec | None = None,   # dtype + flop/byte counts for achieved-% math
    label: KernelLabel | None = None,       # name/impl/dtype/shape/layout for the row
) -> Result: ...

def sweep(
    target: Callable | str,          # callable factory, or "corpus:gemm", or "file.py::sym"
    *,
    shapes: str | list,
    dtypes: list[str],
    layouts: list[str] = ["row"],
    **bench_kwargs,
) -> Grid: ...

# do_bench shim — signature-compatible with triton.testing.do_bench
def do_bench(fn, warmup=25, rep=100, grad_to_none=None,
             quantiles=None, return_mode="mean") -> float | list[float]: ...

# Data types (frozen dataclasses; see Appendix B/C)
class Result: ...        # .p50_us, .mad_us, .achieved_tflops, .ptxas, .machine, .flags, .to_dict()
class Grid:  ...          # .rows: list[Result]; .to_parquet(path); .to_json(path); .filter(**)
class RooflineSpec: ...   # dtype, flops, bytes_hbm  (either provided or inferred for corpus kernels)
class KernelLabel: ...
```

**Stability rule:** anything importable from `caliper/__init__.py` is covered by
semver from v0.3.0. Everything under `caliper._internal.*` is private.

### 1.5 CLI contract

All commands: `--help`, `--version`, `--log-level`, machine-readable output via
`--json` where noted. Exit codes are part of the contract and are asserted in CI.

| Command | Purpose | Key flags | Exit codes |
|---------|---------|-----------|------------|
| `caliper doctor` | Is this machine fit to benchmark? | `--json`, `--strict` (warnings → non-zero) | 0 fit / 1 not fit / 2 error |
| `caliper fingerprint` | Print the machine fingerprint block | `--json` | 0 / 2 |
| `caliper bench <target>` | Measure one kernel / a shape list | `--shapes`, `--dtype`, `--layout`, `--flush-l2/--no-flush-l2`, `--lock-clocks/--no-lock-clocks`, `--warmup`, `--min-samples`, `--cuda-graph`, `--repeat`, `--json`, `--parquet` | 0 / 2 |
| `caliper sweep <spec.yaml>` | Run a matrix | `--parquet`, `--resume`, `--dry-run` | 0 / 2 |
| `caliper compare` | Regression diff of two datasets | `--baseline`, `--candidate`, `--arch`, `--threshold`, `--fail-on-regression`, `--json` | 0 ok / 1 regression / 2 error |
| `caliper selftest` | Validate this install against on-device oracles | `--full`, `--report`, `--json` | 0 pass / 1 fail / 2 error |
| `caliper validate <path...>` | Schema + sanity check any caliper output | `--json` | 0 valid / 1 invalid / 2 error |
| `caliper submit <path...>` | Build & (optionally) push a results bundle | `--dry-run`, `--repo`, `--out` | 0 / 2 |

`<target>` grammar: `path/to/file.py::symbol` | `corpus:<name>` | `-` (read a spec
from stdin).

### 1.6 Data schemas

- **`Result` / row schema** — Appendix B (JSON) and Appendix C (flat Parquet row).
  `schema_version` is embedded; `caliper validate` is the reference implementation.
- **`sweep` spec** — Appendix D.
- **`selftest` report** — Appendix E.

### 1.7 Platform support matrix

| Arch | Rep GPU | Tier | Validation method (see §0.5) |
|------|---------|------|------------------------------|
| SM75 | T4 | 1 | **Colab-full** (fp32/fp16 dtypes only) |
| SM80 | A100-40GB | 1 | **Colab-full** (the workhorse arch) |
| SM89 | L4 (Colab) + RTX 4090 (golden/rental) | 1 | **Colab-full** on L4; rental adds lock/`ncu`/throttle on a 4090 |
| SM90 | H100 | 1 | **rental-pass** (~$16, one pass at acceptance) |
| SM86 | RTX 3090 | 2 → 1 if it passes | **rental-pass** (~$8 spot, best-effort) |
| SM70 | V100 | 2 | Colab if it appears |
| CDNA3 | MI300X | 2 | **rental-pass** (~$10) — "runs & emits a valid row" only |
| Apple | M-series GPU | N/A | no CUDA backend; the Mac is the pure-core dev host, not a target |
| — | B200 / CDNA4 / Windows / Jetson / multi-GPU | out of scope | — |

**Colab-full** = every NFR except clock-lock and `ncu` is enforced on this arch,
on Colab. The clock-lock path, the `ncu` cross-check, and the power-cap throttle
are validated **once** on the golden box (Lambda A100) and are arch-independent
logic. Tier-1 = NFR targets enforced + a filed all-PASS acceptance report. Tier-2
= `selftest` run & filed, failures triaged, not release-blocking.

---

## 2. Architecture for testability

### 2.1 Layering — Rust core, bindings, Python shell, GPU layer

```
        ┌───────────────────────────────────────────────┐
        │  python/caliper/cli.py — arg parse, exit codes  │   thin
        ├───────────────────────────────────────────────┤
        │  python/caliper/api.py — bench/sweep/compare    │   Python
        │  + do_bench shim, Parquet wiring, orchestration │   shell
        ├───────────────────────────────────────────────┤
        │  caliper._core  (crates/caliper-ffi, PyO3)      │   thin binding
        ├───────────────┬───────────────────────────────┤
        │  crates/caliper-gpu   │   crates/caliper-core   │
        │  Rust: cudarc +       │   Rust, #![forbid(      │
        │  nvml-wrapper for      │   unsafe_code)]:        │
        │  launch / events /     │   schema.rs  stats.rs   │
        │  graphs / clocks /     │   warmup.rs  roofline.rs│  cargo test
        │  throttle / devinfo    │   ptxas_parse.rs        │  = L0
        │  + kernels/*.cu        │   occupancy.rs          │
        │  (CUDA C++, O1–O7)     │   thresholds.rs  spec.rs │
        │  PORTS: real|fixture|  │   fingerprint.rs        │  no GPU,
        │  record               │   validate              │  no I/O
        └───────────────────────┴───────────────────────┘
```

**Rules that make it testable:**

1. **All measurement math is pure Rust in `caliper-core`.** Given a slice of
   sample times, `stats` returns p10/p50/p90/MAD. Given a timing sequence,
   `warmup` returns a steady-state index. Given (flops, bytes, seconds, arch,
   dtype), `roofline` returns achieved %, AI, ridge point, bound. Given two
   datasets + a noise model, `thresholds` returns regression verdicts. `caliper-core`
   has no GPU dependency and no `unsafe` → the entire numerical heart is covered
   by `cargo test` and again by Python tests through the bindings, with zero
   hardware.
2. **Every hardware/tool interaction is behind a narrow port in `caliper-gpu`**
   with three implementations: `real` (production), `fixture` (replays a recorded
   capture — used in Rust unit tests and L1 CI), `record` (wraps `real`, writes a
   capture file — used to generate fixtures). See §2.3. `caliper-core` has no
   ports because it touches no hardware.
3. **Deterministic seams.** The warmup detector, the sample invalidator (samples +
   throttle events → kept/dropped), the autotune-config cache key builder, and the
   schema validator are all pure functions with golden-file tests.
4. **One schema, one validator.** Every command's output is a `Result`/row.
   `caliper validate` is the single source of truth and is itself unit-tested
   against a corpus of valid and each class of invalid document.
5. **Oracles ship in the package** (`caliper.corpus.oracles`) so `selftest` is
   self-contained and every result is reproducible by anyone.

### 2.2 Module map + per-module contract & test strategy

Rust modules live in `crates/caliper-core/src/` unless noted; each gets a thin
`caliper-ffi` export. Python modules live in `python/caliper/`.

| Module | Lang | Responsibility | Independent test strategy |
|--------|------|----------------|---------------------------|
| `schema.rs` | Rust | `Record` schema, canonical JSON, lenient parse, `validate()` | `cargo test` + Python round-trip: valid corpus passes, one case per invalid class fails |
| `stats.rs` | Rust | p10/p50/p90, MAD, CoV, cross-pass aggregation | `cargo test`: property tests (monotonicity, known inputs), golden vectors |
| `warmup.rs` | Rust | steady-state detection from a timing stream | `cargo test`: synthetic sequences (ramp, spike, flat, noisy) → expected index; golden files |
| `roofline.rs` | Rust | achieved TFLOP/s & GB/s, AI, ridge, bound; per-arch peaks table | `cargo test`: table has a cited source per cell; math vs hand-computed cases; oracle values reproduced |
| `ptxas_parse.rs` | Rust | parse `ptxas -v` / `cuobjdump` / HIP `-v` | `cargo test`: 6 checked-in real captures → expected structs; malformed → typed error |
| `occupancy.rs` | Rust | theoretical occupancy model | `cargo test`: 10 (arch, regs, smem, block) tuples vs reference table |
| `thresholds.rs` | Rust | noise band from MAD history; regression verdict | `cargo test`: injected slowdown → fires; within-noise → silent; spill delta surfaced |
| `spec.rs` | Rust | parse & expand `sweep` YAML; shape libraries | `cargo test`: spec → expanded cell list; shape libs → documented tuples; bad spec → error |
| `fingerprint.rs` | Rust | shape/serialise the machine fingerprint (no collection) | `cargo test`: round-trip; field-completeness check |
| `caliper-gpu: launcher.rs` | Rust (`cudarc`) | launch a callable, capture CUDA events, graph capture, batched mode | L1: fixture replay; L2: oracle kernels |
| `caliper-gpu: clock.rs` | Rust (`nvml-wrapper`) | lock/unlock clocks, read clocks, throttle reasons | L1: fixture replay of NVML calls; L2: induced throttle |
| `caliper-gpu: device_info.rs` | Rust | L2 size, SM count, VRAM, PCIe, ECC, MIG, BAR1, driver/runtime versions | L1: fixture; manual byte-exact check vs `nvidia-smi -q` |
| `caliper-gpu: module_probe.rs` | Rust + shell-out | run `ptxas -v` / `cuobjdump`, call the occupancy API | L1: fixture; L2: vs ncu (golden box) |
| `caliper-gpu: kernels/*.cu` | CUDA C++ | oracle kernels O1–O7 (Appendix A) | L2: the oracle suite |
| `python/caliper/toolchain.py` | Python | detect Triton/torch/CUDA/ptxas versions | L1: fixture; L2: vs `pip show` / `nvcc --version` |
| `python/caliper/results.py` | Python | write Parquet/JSON (pyarrow), build submit bundle | L0/L1: schema-valid output via the Rust validator; bundle structure |
| `python/caliper/corpus/` | Python + Triton | reference kernels + vendor baselines | L2/L6: run on Tier-1 archs; source pinned + hashed |
| `python/caliper/api.py` | Python | `bench` / `sweep` / `compare` orchestration; `do_bench` shim | L6: end-to-end on GPU; L1 with fixture GPU layer for control flow |
| `python/caliper/cli.py` | Python | arg parsing, output formatting, exit codes | L0: exit-code table; L6: subprocess smoke tests |
| `python/caliper/selftest.py` | Python | run oracle suite + doctor + repro + optional nsys | L2/L3: on real GPUs |

### 2.3 Ports: fixture / record mechanism

Each port is a Rust trait in `caliper-gpu`. The `real` impl does the work. The
`record` impl wraps `real` and appends every call + return value (or error) to
`crates/caliper-gpu/fixtures/<port>/<name>.jsonl`. The `fixture` impl replays a
named capture and asserts call order. Selection is via a Cargo feature / env var
(`CALIPER_GPU_PORTS=record|fixture|real`).

- Generate/refresh fixtures on a real GPU: `CALIPER_GPU_PORTS=record cargo test -p
  caliper-gpu <name>` then commit the new `.jsonl`.
- CI (no GPU) builds `caliper-gpu` with the `fixture` feature → L1 exercises real
  control flow against recorded hardware responses. `caliper-core` needs none of
  this — it never touches hardware.
- A fixture staleness check (L1): every fixture records the caliper version and
  arch it came from; CI warns if a fixture is > 2 minor versions old.

---

## 3. Test strategy (the testability system)

### 3.1 Test taxonomy

| Level | Name | Needs GPU? | Where it runs | Gate |
|-------|------|------------|---------------|------|
| **L0** | Unit — `cargo test` (Rust core) + Python tests through the bindings | No | CI on every push (`ci-cpu`), both lanes | Blocks merge |
| **L1** | Contract — `caliper-gpu` ports built with the `fixture` feature | No | CI on every push (Rust lane) | Blocks merge |
| **L2** | Oracle (first-principles on-device) | Yes | `notebooks/dev.ipynb` on Colab, after each GPU-affecting push | Blocks release |
| **L3** | Cross-tool: `nsys` / `ptxas -v` / `nvbandwidth` / cuBLAS on Colab (every arch); `ncu` on the golden box (once) | Yes + tools | Colab notebooks + the golden-box pass | Blocks release (Tier 1) |
| **L4** | Reproducibility (multi-run variance) | Yes | Colab = unlocked tier; golden box + H100 = locked tier | Blocks release |
| **L5** | Manual acceptance playbook (A = Colab, B = golden box) | Yes | You: Playbook A per Colab arch + Playbook B once | Blocks release (Tier 1) |
| **L6** | End-to-end pipeline (`sweep`→`validate`→`compare`→`submit --dry-run`) | Yes | `ci-gpu` + you | Blocks release |

### 3.2 Oracle kernel suite — first-principles expected values

Full source sketches in Appendix A. Each oracle is a kernel whose *true* behavior
is knowable without trusting caliper, so it pins one measurement path.

| Oracle | What it pins | Analytic expectation | Pass tolerance |
|--------|--------------|----------------------|----------------|
| **O1 `busy(target_ns)`** — spins on `clock64()` until `target_ns` of locked-clock cycles elapse | The timing path itself (duration ↔ wall-clock) | `p50 ≈ target_ns` for a sweep `target_ns ∈ {1µs … 10ms}`; linear fit slope ∈ [0.97, 1.03], intercept ≈ `launch_overhead_us` | slope within 3%; per-point within 3% for ≥ 50 µs, 10% below |
| **O2 `stream_triad(n_bytes)`** — grid-stride `a = b + s*c` over `n_bytes` | GB/s computation + L2 flush correctness | `achieved_gbps = 3 * n_bytes / p50`; at large `n_bytes` this is 70–90% of measured HBM; flush-on vs flush-off diverge ≥ 2× when `n_bytes < L2`, converge < 5% when `n_bytes ≥ 4·L2` | vs `nvbandwidth` d2d ± 5% (NFR-3) |
| **O3 `fma_peak(n_iters)`** — register-resident FMA loop, no memory traffic, high ILP | TFLOP/s computation + compute-bound roofline branch | `achieved_tflops = 2 * threads * n_iters * ILP / p50` ≈ 90–98% of documented FP32 FMA peak; `bound == "compute"` | ≥ 90% of documented peak; classified `compute` |
| **O4 `one_op()`** — kernel with a single instruction | `launch_overhead_us` | `p50 ≈` pure launch+teardown (~3–10 µs eager); under graph replay per-launch cost < 1 µs | within 20% of nsys API→start gap |
| **O5 cuBLAS GEMM** (via cuBLAS handle, not a caliper kernel) | The whole stack against a trusted third-party kernel | `achieved_tflops` matches ncu for the same call; matches published cuBLAS efficiency for the arch | vs ncu ± 3% (NFR-3); vs nsys duration ± 3% (NFR-1) |
| **O6 `throttle_bait()`** — sustained high-power FMA to trip a lowered power cap | Throttle detection + sample invalidation | With `nvidia-smi -pl <low>`: `throttle_reasons` includes `SW_POWER_CAP` or `HW_*`; `invalidated_samples > 0`; run flagged | detection rate 100% over 5 runs |
| **O7 `calibration_gemm`** — a fixed small GEMM with a known locked-clock time per SKU (checked-in table) | The community-submission trust gate | On locked clocks, `p50` within X% of the table value for that SKU | ± 8% → `verified`; else `clocks-suspect` |

`caliper selftest` runs O1–O4, O6, O7 and the reproducibility check; `--full`
adds O5 and the nsys cross-check for O1/O2/O5.

### 3.3 External cross-checks — exact procedure & tolerances

| Tool | Where | What you compare | Tolerance |
|------|-------|------------------|-----------|
| **Nsight Systems** (`nsys`) | **[Colab]** every arch + golden box | `nsys profile --stats=true` kernel `Duration` (avg) vs caliper `p50_us` for O1@200µs, O2@1GB, O5 | ≤ 3% (NFR-1) |
| **Nsight Compute** (`ncu`) | **[golden box only]** — Colab denies counters | `gpu__time_duration`, `launch__registers_per_thread`, `sm__warps_active`, `sm__throughput` vs caliper `p50` / `regs_per_thread` / `achieved_occupancy` / `achieved_tflops` for all 5 corpus kernels | duration ≤ 3%, registers exact, occ ± 0.05, tflops ≤ 3% |
| **`ptxas -v`** (register / smem / spill ground truth) | **[Colab]** every arch | the compiler's own static usage report vs `Result.ptxas` | exact |
| **nvbandwidth** | **[Colab]** every arch + golden box | `device_to_device_memcpy_read_ce` GB/s vs O2 `achieved_gbps` at matched size | ≤ 5% (NFR-3) |
| **cuBLAS** (O5) | **[Colab]** every arch | `achieved_tflops` vs documented cuBLAS efficiency for the arch; vs `ncu` on the golden box | ≤ 3% |
| **`nvidia-smi -q`** | **[Colab]** every arch | every `machine` fingerprint field | byte-exact |
| **Thermal-throttle O6 fallback** | **[Colab]** T4 | sustained O3 for ~3 min on a T4 until `SW_THERMAL` / `HW_THERMAL` appears → assert detection + sample drop | detection 100% / 5 runs |
| **Power-cap O6** | **[golden box only]** | `nvidia-smi -pl <~60% TDP>` → assert `SW_POWER_CAP` detected + samples dropped | detection 100% / 5 runs |
| **BabelStream** (optional) | any | second opinion on O2 sustained bandwidth | ≤ 8% |

Automated by: `caliper selftest [--full]`, `tests/l3/test_vs_*.py`, and
`tests/l1/test_fingerprint_fields.py` + the manual `checklist_fingerprint.md`.

### 3.4 `caliper selftest` — spec

```
$ caliper selftest --full --report selftest-<host>-<arch>.json
caliper selftest 0.3.0  ·  RTX 4090 (SM89)  ·  driver 550.90  ·  CUDA 12.4
[ 1/9] doctor invariants ............................. PASS
[ 2/9] O1 duration oracle  (1µs–10ms sweep) .......... PASS  slope=1.006  intercept=6.2µs
[ 3/9] O2 stream triad  (L2-flush A/B + GB/s) ........ PASS  912 GB/s (90.5% HBM)  flush Δ=3.7×
[ 4/9] O3 fma peak ................................... PASS  81.9 TFLOP/s (95.2% FP32 FMA)
[ 5/9] O4 launch overhead ............................ PASS  6.1µs eager / 0.4µs graph
[ 6/9] O6 throttle detection (induced) ............... PASS  SW_POWER_CAP flagged, 214 samples dropped
[ 7/9] O7 calibration gemm .......................... PASS  p50 1.812ms vs table 1.79ms (+1.2%)
[ 8/9] reproducibility (10× fresh process) ........... PASS  CoV(p50)=0.9%
[ 9/9] cross-check vs nsys (O1,O2,O5) ................ PASS  max Δ=2.1%
RESULT: PASS (9/9)   report → selftest-box-sm89.json
```

- Machine-readable report validates against Appendix E.
- Every check emits `{name, status, measured, expected, tolerance, detail}`.
- `--full` requires `nsys` on PATH; without it check 9 is `SKIP` (not `FAIL`) and
  the overall result notes reduced coverage.
- On a constrained host (Colab): check 6 uses the **thermal** fallback (T4) or is
  `SKIP` (A100/L4 — no power-cap control); check 8 runs the **unlocked** tier;
  the header prints `environment: CONSTRAINED` and the report sets
  `coverage: reduced` with `not_validated: ["clock_lock","ncu_crosscheck","powercap_throttle"]`.
  A `reduced` run with every non-SKIP check passing is still `RESULT: PASS`.
- Non-zero exit if any non-SKIP check fails.

### 3.5 Reproducibility protocol (L4)

**Colab (unlocked tier — every arch):**
1. `caliper bench corpus:gemm --shapes '{M:4096,N:4096,K:4096}' --dtype bf16
   --json run_<i>.json` in **10 fresh subprocesses** (a Colab kernel restart per
   run is fine).
2. `caliper validate --repro run_*.json` computes CoV(p50) and asserts the
   **unlocked** NFR-5 tier (< 5% for ≥ 100 µs).
3. Interleave test: `A,B,A,B,...` vs `A×10 then B×10` for two kernels → p50 of
   each agrees within noise (proves no ordering/thermal contamination).

**Golden box + H100 (locked tier — once):**
4. `caliper doctor` confirms clocks lockable; repeat steps 1–2 locked → assert the
   **locked** NFR-5 tier (< 2%).
5. Repeat once unlocked on the same box → assert unlocked CoV > locked CoV (proves
   locking does something).

### 3.6 CI design

| Workflow | Trigger | Runner | Runs |
|----------|---------|--------|------|
| `ci-cpu.yml` — **rust lane** | every push / PR | GitHub-hosted | `cargo fmt --all --check`, `cargo clippy --all-targets --all-features -D warnings`, `cargo test --all` (this is **L0** for the core and **L1** for `caliper-gpu` built with the `fixture` feature). Rust cache via `Swatinem/rust-cache`. **Blocks merge.** |
| `ci-cpu.yml` — **python lane** (matrix 3.10/3.11/3.12) | every push / PR | GitHub-hosted | install Rust toolchain, `pip install -e ".[dev]"` (maturin builds `caliper._core`), `ruff check` + `ruff format --check`, `mypy`, `pytest -m "l0 or l1"` (Python tests through the bindings), `maturin build`. < 5 min. **Blocks merge.** |
| `notebooks/dev.ipynb` ("ci-gpu") | you run it after a GPU-affecting push; Pro+ can schedule it | **Colab** (A100 preferred) | `git pull` → `pip install -e .[dev]` → **L2** oracle suite + **L4** unlocked-repro + **L6** e2e + `caliper selftest`. Paste the pass/fail tail into the PR. |
| `notebooks/corpus_sweep.ipynb` | at acceptance, per Colab arch | **Colab** | `caliper selftest --full` + a 30-cell (or larger, Pro+ background) corpus `sweep` → push rows to `caliper-results` staging. |
| golden-box + H100 passes | once, at acceptance (Week 4) | **rented root instances** | Playbook B (§3.7): `ncu` L3, FR-4 lock, O6 power-cap, locked NFR-5; then a corpus sweep → `caliper-results`. |
| `release.yml` | tag `v*` | GitHub-hosted | build sdist+wheel, Test PyPI → PyPI, attach every `selftest-*.json` + the golden-box `ncu` report to the GitHub Release. |

### 3.7 Your exhaustive manual acceptance playbook (L5)

Run this **once per GPU** you can access. Record results in
`docs/acceptance/reports/<arch>-<host>-<date>.md` from the template in Appendix F.
Tier-1 archs must be all-PASS to release; Tier-2 archs are filed with triage notes.

**Split by environment (see §0.5):**

- **Playbook A — Colab, every reachable arch (T4 / A100 / L4, + V100 if it
  appears):** steps 1, 2, 3 (`selftest` reports `coverage: reduced` — expected),
  4, 5, 6, 7 *(the `ptxas -v` register/occupancy check only; the `ncu` sub-check
  is deferred)*, 8 *(unlocked tier only)*, 10, 11, 12, 13, 14. Step 9 uses the
  **thermal** throttle fallback on a T4.
- **Playbook B — golden box, once (Lambda A100 root instance), plus the H100
  pass:** the full step 7 **with `ncu`** for all 5 corpus kernels, step 8
  **locked tier**, step 9 with **`nvidia-smi -pl`** power-cap induction, and
  FR-4's clock-lock acceptance. The H100 rental is a normal root instance too, so
  re-do step 8 locked there as well.

| # | Step | Command | Expected result | Pass? |
|---|------|---------|-----------------|-------|
| 1 | Install clean | `python -m venv v && v/bin/pip install caliper-gpu` | no errors; `caliper --version` prints v0.3.0 | |
| 2 | Doctor | `caliper doctor --json` | verdict present; every field cross-checks vs `nvidia-smi -q` (fill the fingerprint checklist) | |
| 3 | Selftest | `caliper selftest --full --report r.json` | exit 0; 9/9 (or 8/8 + SKIP if no nsys); `r.json` validates | |
| 4 | Duration linearity | `caliper selftest --full` (its O1 check sweeps `target_ns ∈ {1µs … 10ms}` and runs `check_o1_linearity`) — see the `selftest` report's `o1_duration_linearity` entry | linear fit slope ∈ [0.97,1.03]; report attached | |
| 5 | L2 flush A/B | `caliper bench corpus:o2 --bytes $((L2/2)) --flush-l2` vs `--no-flush-l2`; then `--bytes $((L2*4))` both ways | small: ≥ 2× GB/s gap; large: < 5% gap | |
| 6 | Bandwidth cross-check | build `nvbandwidth`, run `./nvbandwidth -t device_to_device_memcpy_read_ce`; compare to O2 at matched size | within 5% | |
| 7 | cuBLAS vs ncu | `caliper bench corpus:gemm --shapes '{M:4096,N:4096,K:4096}' --dtype bf16 --json g.json`; `ncu --set full -k <k> ...` | duration Δ ≤ 3%; registers exact; occupancy ± 0.05; TFLOP/s Δ ≤ 3% (fill the table) | |
| 8 | Reproducibility | run §3.5 steps 1–4 | locked CoV < 2%; unlocked CoV > locked | |
| 9 | Throttle handling | `sudo nvidia-smi -pl <~60% TDP>`; `caliper bench corpus:o3 --json t.json`; restore `-pl` | `throttle_reasons` non-empty; `invalidated_samples > 0`; run flagged | |
| 10 | do_bench shim | edit Triton `03-matrix-multiplication.py` to `from caliper import do_bench`; run it | runs unchanged; numbers within 3% of `caliper.bench` p50 | |
| 11 | Sweep + resume | `caliper sweep examples/mini.yaml --parquet s.parquet`; `kill` mid-run; rerun with `--resume` | resumes; final Parquet passes `caliper validate` | |
| 12 | Compare catches regression | `caliper compare --baseline testdata/base.parquet --candidate testdata/slow.parquet --fail-on-regression` | exit 1; prints the slowdown **and** the spill delta | |
| 13 | Submit dry-run | `caliper submit s.parquet --dry-run --out bundle/` | bundle has rows + fingerprint + calibration result + version; `caliper validate bundle/` passes | |
| 14 | Negative validate | `caliper validate testdata/over_peak_row.parquet` etc. (4 bad fixtures) | each fails with a specific, correct message | |

### 3.8 Test artifacts & locations

```
tests/
  l0_unit/            # Python tests that exercise the Rust core via the bindings
  l2_oracle/          # on-device oracle drivers (call into caliper-gpu)
  l3_crosstool/       # vs nsys / ncu / nvbandwidth
  l4_repro/
  l6_e2e/
  testdata/           # tiny parquet/json: valid + one per invalid class + injected-regression pair
crates/caliper-core/  # #[cfg(test)] unit tests + tests/ ; golden/ = parser inputs + expected structs
crates/caliper-gpu/
  fixtures/           # recorded port captures (committed), replayed by fixture-feature tests
docs/acceptance/
  manual-playbook.md
  checklist_fingerprint.md
  reports/            # one filled report per GPU (committed)
selftest-reports/     # selftest-*.json from every arch (attached to the release)
```

---

## 4. Week-by-week execution plan

Effort is in focused hours (h). Budget ≈ 124 h. Each task: **Do**, **DoD**
(definition of done), **Verify**.

> **Language note (see §0.6):** where a task says `_internal/foo.py`, build it as
> `crates/caliper-core/src/foo.rs` (pure Rust, `cargo test`) + a thin
> `caliper-ffi` export + a Python wrapper in `python/caliper/` only if callers
> need one. Where a task says `ports/*.py`, build it in `crates/caliper-gpu`
> (Rust, `cudarc`/`nvml-wrapper`). `.cu` oracle/reference kernels live in
> `crates/caliper-gpu/kernels/`. The acceptance criteria are unchanged.

### Week 1 — Timing core, environment, `doctor`, oracles (≈ 31 h)

**Goal:** a trustworthy `caliper.bench()` for kernels ≥ 50 µs and a `doctor`,
each proven against O1/O2/O4 and (manually) against `nsys`.

#### W1D1 — Repo & CI skeleton (5 h)
- **T1 (2h)** Scaffold: `pyproject.toml` (package name `caliper-gpu`, import
  `caliper`), ruff + mypy config, `src/` layout, `pytest` markers `l0 l1 l2 l3 l4
  l6`, `CALIPER_PORTS` env switch. **DoD:** `pip install -e .[dev]` works; `pytest
  -m l0` runs (0 tests ok). **Verify:** clean `pip install` in a throwaway venv.
- **T2 (2h)** `ci-cpu.yml`: ruff, mypy, `pytest -m "l0 or l1"`, build. **DoD:**
  green on an empty PR. **Verify:** the Actions run is green < 5 min.
- **T3 (1h)** `_internal/schema.py` stub + `Result` dataclass (Appendix B fields,
  all `Optional` initially) + `to_dict()`. **DoD:** round-trips through
  dict/JSON. **Verify:** `tests/l0_unit/test_schema_roundtrip.py`.

#### W1D2 — Pure core: stats + warmup (6 h)
- **T1 (3h)** `_internal/stats.py`: p10/p50/p90, MAD, CoV, cross-pass merge.
  **DoD:** matches hand-computed golden vectors; property tests (order-invariance,
  p50 of constant = that constant). **Verify:** `pytest -m l0 -k stats`.
- **T2 (3h)** `_internal/warmup.py`: `steady_state_index(times, tol)`; `"auto"` vs
  fixed. **DoD:** golden decisions for ramp / spike / flat / noisy synthetic
  sequences; reproduces the "25 iterations is too few" failure on a ramp fixture.
  **Verify:** `pytest -m l0 -k warmup`.

#### W1D3 — Ports: launcher + clock + device_info (7 h)
- **T1 (3h)** `ports/kernel_launcher.py` `real`: launch a `Callable`, CUDA-event
  timing, N-in-a-row batched mode, CUDA-graph capture path. `record`/`fixture`
  wrappers. **DoD:** returns raw sample times for a trivial kernel. **Verify:**
  ad-hoc script on the dev GPU.
- **T2 (2h)** `ports/gpu_clock.py` `real` (pynvml): lock/unlock SM+mem, read
  clocks, read throttle-reason bitmask → list of names. **DoD:** lock then read
  shows locked freq; unlock restores. **Verify:** compare to `nvidia-smi -q -d
  CLOCK` before/after.
- **T3 (2h)** `ports/device_info.py` `real`: L2 bytes, SM count, VRAM, PCIe
  gen/width, ECC, MIG, BAR1, driver/runtime/NVML versions. **DoD:** populates the
  `machine` block. **Verify:** eyeball vs `nvidia-smi -q`.

#### W1D4 — `bench()` v0 + L2 flush + clock/throttle integration (7 h)
- **T1 (3h)** `api.bench()`: warmup → locked-clock samples → flush between samples
  → stats → `Result` (timing fields + `machine` + `clocks` + `flags`). **DoD:**
  returns a populated `Result` for a Triton kernel. **Verify:** manual.
- **T2 (2h)** `_internal` L2 flush buffer sized from `device_info.l2_bytes`;
  `--flush-l2/--no-flush-l2`. **DoD:** buffer bytes == L2 size ± granularity.
  **Verify:** `tests/l2_oracle/test_l2_flush.py` (needs O2 → may land W1D5).
- **T3 (2h)** Throttle polling + sample invalidation (pure `invalidate(samples,
  throttle_events)` in core; wiring in shell). `clocks-unlocked` tag path.
  **DoD:** induced throttle drops samples. **Verify:** O6 (W1D5) / playbook #9.

#### W1D5 — Oracles O1–O4, O6 + `doctor` + week gate (6 h)
- **T1 (3h)** `corpus/oracles.py`: O1 `busy`, O2 `stream_triad`, O3 `fma_peak`,
  O4 `one_op`, O6 `throttle_bait` (Appendix A). `corpus:o1..o4` CLI targets.
  **DoD:** each runs; O1 sweep is monotonic. **Verify:**
  `tests/l2_oracle/test_o1..o4.py` on the dev GPU.
- **T2 (2h)** `caliper doctor` + `caliper fingerprint`: checks for lockable
  clocks, active throttle, ECC, MIG, persistence, background load; exit codes.
  **DoD:** flags each induced scenario. **Verify:** playbook #2 partial.
- **T3 (1h)** `cli.py` `bench` wired to `api.bench`; `--json` output. Create
  `notebooks/dev.ipynb` (git pull → `pip install -e .[dev]` → `pytest -m "l2 or
  l6"` → tail failures) and `make sync`.
  **Week-1 gate demo (on Colab — A100 or T4):** `caliper doctor` + `caliper bench
  corpus:o1 --recording <captured>.jsonl` (a single point); the O1 *sweep* +
  `check_o1_linearity` runs under `caliper selftest --full` (W2D5). Run `nsys
  profile --stats=true` on O1@200µs and O2@1GB and record Δ (must be ≤ 3% →
  NFR-1). Commit `crates/caliper-gpu/fixtures/*` captured via
  `CALIPER_GPU_PORTS=record` **from the Colab session** (the Mac cannot record
  them).

**Week-1 acceptance gate (all must hold):**
- [ ] FR-1, FR-3, FR-5, FR-6, FR-11, FR-19 meet their AC on a Colab A100.
- [ ] FR-4: the **`clocks-unlocked` tagged path** works on Colab (lock denied →
      handled gracefully, run flagged, `environment: CONSTRAINED` reported). The
      lock *path itself* is a Week-4 golden-box item.
- [ ] O1 linear-fit slope ∈ [0.97, 1.03]; O1@200µs and O2@1GB within 3% of `nsys`
      on Colab.
- [ ] `ci-cpu` green (from the Mac); L0+L1 for stats/warmup/schema/ports.
- [ ] Port fixtures committed (captured on Colab via `CALIPER_PORTS=record`).

### Week 2 — Analysis layer, fingerprint, `do_bench`, `selftest` (≈ 30 h)

**Goal:** every non-timing number caliper prints (registers, occupancy, TFLOP/s,
GB/s, roofline bound) is proven against `ncu` / `nvbandwidth`; `selftest` exists.

#### W2D1 — ptxas / cuobjdump parsing (6 h)
- **T1 (3h)** `_internal/ptxas_parse.py`: parse `ptxas -v` (regs, static/dynamic
  smem, spill loads/stores, local, stack), multi-kernel modules, `cuobjdump
  -res-usage`, HIP `-v`. **DoD:** 6 golden captures → expected structs; malformed
  → `PtxasParseError`. **Verify:** `pytest -m l0 -k ptxas`.
- **T2 (2h)** `ports/module_probe.py` `real`: compile a Triton kernel with
  `TRITON_ALWAYS_COMPILE`/dump, run `ptxas -v`, capture. **DoD:** returns the
  struct for a corpus kernel. **Verify (Colab):** `regs_per_thread` matches the
  value Triton reports in the compiled-kernel metadata (`.n_regs`). `ncu`
  confirmation → Week-4 golden box.
- **T3 (1h)** Wire `ptxas` into `Result.ptxas`. **DoD:** field populated in a real
  `bench`. **Verify:** playbook #7 (partial).

#### W2D2 — Occupancy + roofline math (6 h)
- **T1 (2h)** `_internal/occupancy.py`: theoretical occupancy model. **DoD:** 10
  reference tuples match the checked-in table. **Verify:** `pytest -m l0 -k occ`.
- **T2 (1h)** `ports/module_probe` occupancy via CUDA occupancy API (real
  achieved reported alongside). **DoD:** value present. **Verify (Colab):**
  theoretical occupancy equals the CUDA occupancy API's own return; recompute the
  arch max-warps math by hand. `ncu` achieved-occupancy confirmation → Week 4.
- **T3 (3h)** `_internal/roofline.py`: per-arch dtype-aware peaks table (SM70/75/
  80/86/89/90/120 + CDNA3; bf16/fp16/fp8 TC dense, fp32 FMA, HBM GB/s), **each
  cell with a `# source:` comment**; `achieved_tflops`, `achieved_gbps`, AI,
  ridge, `bound`. **DoD:** O2/O3 analytic values reproduced; hand-computed cases
  pass. **Verify:** `pytest -m l0 -k roofline` + O2/O3 on GPU.

#### W2D3 — Fingerprint model + machine block completion (5 h)
- **T1 (2h)** `_internal/fingerprint_model.py`: full field set (Appendix B),
  serialization, completeness assertion. **DoD:** round-trips; missing field →
  error. **Verify:** `pytest -m l0 -k fingerprint`.
- **T2 (2h)** `ports/toolchain.py`: detect Triton/torch/CUDA/ptxas/nvcc versions.
  **DoD:** matches `pip show` / `nvcc --version`. **Verify:** L1 fixture + manual.
- **T3 (1h)** `checklist_fingerprint.md` (every field ↔ `nvidia-smi -q` line).
  **DoD:** doc committed. **Verify:** dry-run the checklist on a Colab A100.

#### W2D4 — `do_bench` shim + roofline for corpus + graph mode polish (7 h)
- **T1 (3h)** `caliper.do_bench` (Triton-compatible signature; `quantiles`,
  `return_mode`, `grad_to_none`). **DoD:** unmodified Triton
  `03-matrix-multiplication.py` runs with only the import changed. **Verify:**
  playbook #10 on a Colab A100.
- **T2 (2h)** `RooflineSpec` inference for corpus kernels (flop/byte counts per
  kernel & shape). **DoD:** `bench(corpus:gemm)` reports `achieved_tflops` +
  `bound`. **Verify (Colab):** cuBLAS GEMM `achieved_tflops` within 3% of the
  documented A100 bf16 cuBLAS efficiency. `ncu` confirmation → Week 4.
- **T3 (2h)** CUDA-graph capture hardening: `cuda_graph="auto"` engages below a
  measured single-launch threshold; graph vs eager `launch_overhead_us`. **DoD:**
  O4 shows eager ~µs, graph < 1 µs. **Verify:** `tests/l2_oracle/test_o4.py`.

#### W2D5 — `selftest` + `ci-gpu` + week gate (6 h)
- **T1 (3h)** `selftest.py`: run O1–O4, O6, O7 + reproducibility + (`--full`) nsys
  cross-check; emit report (Appendix E) with `coverage: full|reduced` +
  `not_validated`; exit codes. **DoD:** exit 0 on a Colab A100 with
  `coverage: reduced`; report validates. **Verify:** `caliper selftest --full`.
- **T2 (2h)** Finalize `notebooks/dev.ipynb` + add `notebooks/selftest.ipynb`
  (runs `caliper selftest --full`, saves the report artifact). Document the
  "push → run `dev.ipynb` on Colab → paste tail into PR" loop in `CONTRIBUTING.md`.
  **DoD:** one full L2 + L4(unlocked) + selftest cycle green on a Colab A100.
- **T3 (1h)** `corpus/oracles.py` O7 `calibration_gemm` + the per-SKU table (seed
  with the Colab A100 value; other SKUs filled at acceptance). **DoD:** O7 runs.
  **Verify:** selftest check 7.

**Week-2 acceptance gate:**
- [ ] FR-7, FR-8, FR-9, FR-10, FR-12, FR-17, FR-19 meet AC on a Colab A100.
- [ ] `regs_per_thread` exact vs **`ptxas -v` / Triton `.n_regs`**;
      `achieved_occupancy` within ±0.05 of the reference model; O3 ≥ 90% FMA peak;
      O2 within 5% of `nvbandwidth`; cuBLAS TFLOP/s within 3% of documented arch
      efficiency. (`ncu` confirmation of all four → Week-4 golden box.)
- [ ] `caliper selftest --full` on a Colab A100: exit 0, `coverage: reduced` with
      the correct `not_validated` list; report committed.
- [ ] The push → `dev.ipynb` → PR loop is documented and works.

### Week 3 — `sweep`, reference corpus, `compare` (≈ 34 h — crunch)

**Goal:** the dataset-producing and regression-detecting half of the product.

#### W3D1 — Spec + shape libraries + `Grid` (6 h)
- **T1 (3h)** `_internal/spec.py`: parse `sweep` YAML (Appendix D), expand to a
  cell list, dedupe, `--resume` state file. **DoD:** golden spec → golden cell
  list; bad spec → typed error. **Verify:** `pytest -m l0 -k spec`.
- **T2 (2h)** Shape libraries `llm-7b`, `llm-70b`, `square-pow2`, `prime-odd` →
  documented (M,N,K)/(B,H,S,D) tuples. **DoD:** each resolves; documented in
  `docs/shapes.md`. **Verify:** `pytest -m l0 -k shapes`.
- **T3 (1h)** `Grid` + `.to_parquet()/.to_json()/.filter()`. **DoD:** writes a
  schema-valid Parquet. **Verify:** `caliper validate`.

#### W3D2 — `sweep()` orchestration + autotune-per-config cache (7 h)
- **T1 (4h)** `api.sweep()`: iterate cells, call `bench` per (kernel, dtype,
  shape, layout, autotune-config), collect `Grid`, checkpoint after each cell,
  `--resume`. **DoD:** 3×3×5 matrix → Parquet; kill + `--resume` completes.
  **Verify:** playbook #11 on a Colab A100 (genuinely kill the runtime mid-sweep).
- **T2 (3h)** Autotune-config cache: key `{sku, driver, cuda, ptxas, triton,
  torch, kernel_source_hash, config_hash}`; per-config timing stored; adding a
  config re-times only it. **DoD:** cache hit/miss test passes. **Verify:**
  `tests/l6_e2e/test_autotune_cache.py`.

#### W3D3 — Corpus kernels: gemm, rmsnorm, softmax (7 h)
- **T1 (3h)** `corpus/kernels/gemm.py`: Triton GEMM (pinned upstream commit +
  content hash) + cuBLAS baseline; `RooflineSpec`. **DoD:** runs on a Colab A100;
  baseline present; valid rows. **Verify:** `achieved_tflops` vs documented A100
  bf16 cuBLAS efficiency; `ncu` confirmation → Week 4.
- **T2 (2h)** `rmsnorm.py`: Triton fwd + `torch` baseline. **DoD:** runs; valid
  row. **Verify:** `caliper validate`.
- **T3 (2h)** `softmax.py`: Triton + `torch` baseline. **DoD:** runs; valid row.
  **Verify:** `caliper validate`.

#### W3D4 — Corpus kernels: attention fwd + bwd (7 h)
- **T1 (4h)** `attention_fwd.py`: Triton (FlashAttention-style, pinned) +
  `F.scaled_dot_product_attention` baseline; causal + GQA; head_dim 64/128.
  **DoD:** runs and emits valid rows on Colab **A100 (bf16) and L4 (fp8)**.
  **Verify:** `caliper validate` + an `allclose` vs the SDPA baseline. (`ncu`
  confirmation → Week-4 golden box.)
- **T2 (3h)** `attention_bwd.py`: Triton bwd + SDPA-backward baseline. **DoD:**
  runs; valid row. **Verify:** `caliper validate` + a sanity `allclose` vs the
  baseline output.

#### W3D5 — `compare` + thresholds + e2e + week gate (7 h)
- **T1 (3h)** `_internal/thresholds.py`: per-facet noise band from MAD history;
  regression verdict; ptxas/occupancy delta; "autotune configs dropped" detector.
  **DoD:** injected 10% slowdown fires; injected spill regression fires with the
  delta shown; within-noise stays silent. **Verify:** `pytest -m l0 -k
  threshold` with `testdata/{base,slow,spill}.parquet`.
- **T2 (2h)** `caliper compare` CLI: `--baseline/--candidate/--arch/--threshold/
  --fail-on-regression/--json`; exit codes. **DoD:** playbook #12 passes.
  **Verify:** subprocess test.
- **T3 (2h)** `tests/l6_e2e/test_pipeline.py`: `sweep` → `validate` → `compare` →
  `submit --dry-run` on a tiny matrix; add to `notebooks/dev.ipynb`. **DoD:**
  green on a Colab A100.

**Week-3 acceptance gate:**
- [ ] FR-13, FR-14, FR-15 meet AC on Colab **A100 + L4** (and T4 for the
      fp16-only subset).
- [ ] All 5 corpus kernels + baselines run and emit valid rows on A100 and L4;
      sources pinned + hashed. (SM90 / SM86 coverage lands in Week-4 rentals.)
- [ ] `sweep --resume` proven by genuinely killing a Colab runtime mid-sweep;
      autotune cache proven.
- [ ] `compare` catches both injected regression classes; silent on noise.
- [ ] `notebooks/dev.ipynb` e2e (L6) green on Colab.

### Week 4 — `submit`, results repo, the writeup, docs, release, acceptance (≈ 31 h)

**Goal:** ship v0.3.0 with the community loop, the launch artifact, the
golden-box pass (the four Colab-unreachable validations), and filed acceptance
reports for every Tier-1 GPU.

#### W4D1 — `submit` + `caliper-results` repo + validation CI (6 h)
- **T1 (3h)** `caliper submit`: build bundle (rows + fingerprint + O7 calibration
  result + version + manifest), `--dry-run`, `--repo`, opens a PR (or writes the
  branch). **DoD:** playbook #13 passes. **Verify:** `caliper validate bundle/`.
- **T2 (2h)** `caliper-results` repo: `results/<arch>/<kernel>/<toolchain-hash>/`
  layout, `schema/`, `SUBMITTING.md`, `.github/workflows/validate.yml` (schema +
  roofline bounds + determinism repeat + calibration check + dedupe + tier
  labelling). **DoD:** a valid bundle PR passes; each bad fixture fails.
  **Verify:** open a self-PR with `testdata/`.
- **T3 (1h)** `caliper validate` finalized as the shared validator (used by both
  repos). **DoD:** playbook #14 (4 negative cases) passes.

#### W4D2 — The launch writeup: "Your Triton benchmark is lying to you" (6 h)
- **T1 (4h)** Reproducible experiment: on ≥ 2 archs (**Colab A100 + Colab L4**;
  add the golden-box A100 if time), table of `do_bench` (default `warmup=25`) vs
  `caliper` vs `nsys` for: (a) a fast kernel (< 20 µs) — show the per-iter-sync
  inflation; (b) a kernel where 25-iter warmup underestimates by > 15%; (c) an
  L2-resident kernel with/without flush. Scripts in `examples/`, numbers
  regenerated by `make writeup-data`. **DoD:** every number reproducible by a
  reader with one Colab cell. **Verify:** run `make writeup-data` in two fresh
  Colab runtimes; diff against committed CSV within tolerance.
- **T2 (2h)** Write `docs/why-do_bench-misleads.md` with the tables, the issue
  links (Triton #2306/#1252/#404/#2832, flashinfer-bench #195), and a "reproduce
  this" section. **DoD:** renders; links resolve.

#### W4D3 — Docs (6 h)
- **T1 (2h)** `README.md` per the positioning in the research doc; install, an
  **"Open in Colab" button** (many readers' only GPU), 30-second example, the
  `do_bench` shim, the misleads-table teaser, the selftest badge, "Submit your
  GPU". **DoD:** a new reader can run `caliper bench` from the linked Colab
  notebook with no local setup. **Verify:** open it in a fresh Colab runtime.
- **T2 (2h)** `docs/api.md` (every public symbol + the `Result`/row schema),
  `docs/cli.md` (every command + exit codes), `docs/shapes.md`. **DoD:** matches
  the frozen §1.4/§1.5. **Verify:** a doctest-style check that signatures match.
- **T3 (2h)** `CONTRIBUTING.md` (add an arch profile / a kernel / a backend / run
  the playbook), `docs/acceptance/manual-playbook.md` (§3.7 + Appendix F).
  **DoD:** committed.

#### W4D4 — Acceptance: Colab archs + golden box + H100 (8 h)
- **T1 (2.5h)** **Colab passes.** In `notebooks/acceptance.ipynb`, run Playbook A
  (§3.7) + `caliper selftest --full` + a 30-cell corpus `sweep` on **T4, then
  A100, then L4** (fresh runtime each). File
  `docs/acceptance/reports/<arch>-colab-<date>.md`, commit each
  `selftest-<arch>.json`, push corpus rows to `caliper-results` staging.
  **DoD:** Playbook A all-PASS on the 3 Colab archs (deviations filed).
- **T2 (3.5h)** **Golden-box pass.** Rent the **Lambda A100 root instance** (~8 h,
  ~$10). Confirm `nvidia-smi -lgc` + `ncu` work in the first 5 min (else switch
  provider from the reserve). `pip install` from Test PyPI. Run **Playbook B**:
  full `ncu` L3 for all 5 corpus kernels, FR-4 clock-lock acceptance, O6
  power-cap throttle, locked-tier NFR-5. Commit the `ncu` report +
  `selftest-goldenbox.json`. Fill the O7 calibration-GEMM table for this SKU.
  **DoD:** the four Colab-unreachable items (§0.5) all PASS; NFR-4/NFR-5
  locked-tier recorded.
- **T3 (2h)** **H100 pass.** Rent an H100 for ~4 h (~$16). Playbook A + locked
  step 8 + `caliper selftest --full` + a 30-cell corpus sweep. File the report,
  commit `selftest-sm90.json`, push rows. **DoD:** SM90 Tier-1 report all-PASS.

#### W4D5 — Triage, Tier-2 best-effort, release (5 h)
- **T1 (2h)** Triage every deviation from W4D4; fix and re-run the affected
  playbook step (use the ~$51 reserve for a re-rental if a fix needs the golden
  box or H100 again). **DoD:** all Tier-1 reports all-PASS.
- **T2 (1.5h)** Tier-2 best-effort within remaining budget: ~$8 spot **RTX 3090**
  (SM86) → Playbook A; ~$10 **MI300X** → `caliper doctor` + one corpus kernel +
  `caliper validate` (the ROCm "runs & emits a valid row" bar only). File reports
  with triage notes; not release-blocking.
- **T3 (1.5h)** `release.yml`: tag `v0.3.0`; build; Test PyPI → PyPI; GitHub
  Release with every `selftest-*.json` + the golden-box `ncu` report + the
  writeup. **DoD:** `pip install caliper-gpu` from PyPI in a **fresh Colab
  runtime** → `caliper doctor` runs and correctly reports `environment:
  CONSTRAINED` (NFR-10).

**Week-4 acceptance gate = the release checklist in Section 5.**

---

## 5. Success criteria — v0.3.0 Definition of Done

Release only when **every** box is checked.

**Functional**
- [ ] FR-1 … FR-19 each meet their acceptance criteria, evidenced by a passing
      test or a filled playbook step (map each FR → its evidence in
      `docs/acceptance/traceability.md`).

**Accuracy (NFR) — recorded in each arch's report**
- [ ] NFR-1 timing ≤ 3% vs `nsys` — on **every** arch (T4, A100, L4, H100,
      golden-box A100).
- [ ] NFR-2 small-kernel ≤ 10% (O1@5µs batched) — every arch.
- [ ] NFR-3 GB/s ≤ 5% vs `nvbandwidth`; O3 ≥ 90% FMA peak — every arch. cuBLAS
      TFLOP/s ≤ 3% vs documented arch efficiency on every arch, **≤ 3% vs `ncu`
      on the golden box**.
- [ ] NFR-4 registers exact vs `ptxas -v` on every arch; **exact vs `ncu` on the
      golden box**. Achieved occupancy ± 0.05 vs the reference model everywhere;
      ± 0.05 vs `ncu` on the golden box.
- [ ] NFR-5 unlocked CoV(p50) < 5% (≥ 100 µs) on every Colab arch; **locked
      CoV < 2% on the golden box + H100**; unlocked CoV > locked where both measured.
- [ ] NFR-6 200 µs kernel measured in ≤ 6 s — every arch.

**Testability**
- [ ] `caliper selftest --full` exits 0 on T4, A100, L4 (`coverage: reduced`,
      correct `not_validated` list), H100, and the golden-box A100
      (`coverage: full`). All reports committed and attached to the release.
- [ ] The golden-box `ncu` L3 report (5 corpus kernels: duration, registers,
      occupancy, tflops within tolerance) committed and attached.
- [ ] L0 + L1 green in `ci-cpu`; 100-run flake check on L0 = 0 flakes (NFR-9).
- [ ] L2 + L4(unlocked) + L6 green via `notebooks/dev.ipynb` on Colab.
- [ ] `docs/acceptance/reports/` has a filled, all-PASS report for each Tier-1
      arch and a filed (triaged) report for each reachable Tier-2 arch.
- [ ] Every negative case in `caliper validate` (over-peak row, missing field,
      non-reproducing bundle, slow calibration kernel) is rejected with a
      specific message (test + playbook #14).

**Community loop**
- [ ] `caliper submit --dry-run` bundle is well-formed and passes `caliper validate`.
- [ ] `caliper-results` validation CI accepts a clean bundle and rejects each bad
      fixture.
- [ ] ≥ 3,000 schema-valid rows from ≥ 4 archs (T4, A100, L4, H100) merged to
      `caliper-results` main. (Use Pro+ background execution for the large sweeps;
      `--resume` across sessions.)

**Distribution & docs**
- [ ] `pip install caliper-gpu` from PyPI works with zero manual steps in a fresh
      **Colab runtime**; `caliper doctor` runs immediately and prints
      `environment: CONSTRAINED` correctly (NFR-10).
- [ ] README, `docs/api.md`, `docs/cli.md`, `docs/shapes.md`,
      `docs/why-do_bench-misleads.md`, `CONTRIBUTING.md`, `SUBMITTING.md` complete
      and accurate to the frozen spec.
- [ ] The writeup's every number regenerates via `make writeup-data` within
      tolerance on a fresh box.
- [ ] `v0.3.0` tagged; GitHub Release published with selftest reports + writeup.

---

## 6. Risk register & contingency descope levers

**Contingency only.** Do not pre-emptively cut. If, at a week gate, you are
> 1 day behind, pull the highest-numbered lever that unblocks you and log it in
`docs/DESCOPE.md`.

| # | Lever (what you'd defer to v0.3.1) | Why it's the least damaging | What you must NOT cut |
|---|-----------------------------------|-----------------------------|-----------------------|
| 1 | `attention_bwd` corpus kernel (keep fwd) | bwd is the hardest kernel; fwd still proves the attention path | gemm, rmsnorm, softmax, attention_fwd |
| 2 | CDNA3 in the peaks table + "runs on MI300X" | ROCm is already best-effort in the spec | SM70–SM120 peaks table |
| 3 | `--resume` for `sweep` | a killed sweep can be re-run from scratch on small matrices | `sweep` producing a valid Parquet |
| 4 | The SM86 (3090) + MI300X Tier-2 rental passes | Tier-2 is best-effort by definition; the 4 Tier-1 archs (T4/A100/L4/H100) are untouched | the golden-box pass and the 4 Tier-1 archs |
| 5 | Cross-pass `--repeat > 1` CoV | single-pass p50 + separate L4 protocol still covers reproducibility | the L4 reproducibility protocol |
| 6 | `prime-odd` shape library (keep llm-7b, llm-70b, square-pow2) | non-aligned stress is a nice-to-have signal | the three LLM/square shape libs |

**Non-negotiable core (never cut):** `bench()` correctness on ≥ 50 µs kernels,
O1/O2/O3/O4 oracles, the `nsys` / `ptxas -v` / `nvbandwidth` cross-checks on every
arch + the one-time `ncu` confirmation on the golden box, `selftest`, `doctor`
(incl. honest-degradation reporting), the schema + `validate`, and filed Tier-1
acceptance reports. Without these the project is "another `do_bench`," which is the
one outcome to avoid.

**Top execution risks:**

- **Colab session limits / disconnects mid-sweep.** *Mitigation:* `sweep
  --resume` is a Week-3 hard requirement and is genuinely exercised (not just
  unit-tested); `corpus_sweep.ipynb` checkpoints every cell; use Pro+ background
  execution for anything > 60 min. Keeping each `bench` under 6 s (NFR-6) means a
  drop costs one cell, not one run.
- **Colab cannot lock clocks or grant `ncu` counters.** *Mitigation:* planned
  for, not a surprise — the `clocks-unlocked` path, `ptxas -v` register ground
  truth, the occupancy reference model, and the O3/cuBLAS TFLOP/s oracles all work
  without either. The lock path, `ncu` cross-check, power-cap throttle, and
  locked-repro tier are batched into one golden-box rental (§0.5, W4D4-T2). If
  that rental *also* denies clock locking (some providers do), switch provider
  from the reserve; if still denied, ship v0.3.0 with FR-4's lock path marked
  "implemented, validated only via graceful degradation" in `docs/DESCOPE.md` and
  make it the first v0.3.1 item.
- **`ncu` permission wall (`ERR_NVGPUCTRPERM`).** Colab always hits this.
  *Mitigation:* `ncu` is used **only** on the rented root golden box (Lambda A100
  — set `NVreg_RestrictProfilingToAdminUsers=0` if needed, or it's already
  permissive); every other arch relies on `nsys` + `ptxas -v` + `nvbandwidth` +
  the oracles, none of which need elevated privilege.
- **$100 is tight; a mid-build bug forces a re-rental.** *Mitigation:* the ~$51
  reserve exists for exactly this. Do all cheap/free validation on Colab **first**
  and only rent once the corpus + oracles are known-good on Colab, so rentals are
  confirmation passes, not debugging sessions.
- **Week 3 overrun.** *Mitigation:* levers 1 and 3 are both in Week 3's scope;
  start `attention_*` on D4 with the baseline first so a partial kernel still
  yields a comparison row.
- **Triton API churn breaking corpus kernels.** *Mitigation:* pin Triton exactly
  in `pyproject.toml` extras; pin each kernel to an upstream commit + hash.
- **Writing CUDA/ports "blind" on the Mac.** *Mitigation:* keep `ports/*/real`
  thin and mechanical; push the logic into the pure core; the Colab `dev.ipynb`
  round-trip is fast enough that a compile error costs minutes, not hours.

---

## 7. Appendices

### Appendix A — Oracle kernel sketches

```cuda
// O1: calibrated duration. Locked clocks ⇒ cycles↔ns known exactly.
__global__ void busy(unsigned long long target_cycles) {
    unsigned long long t0 = clock64();
    while (clock64() - t0 < target_cycles) { /* volatile spin */ }
}
// host: target_cycles = target_ns * sm_clock_hz / 1e9

// O2: streaming triad, known bytes moved = 3 * n (read b, read c, write a).
template<typename T> __global__ void triad(T* a, const T* b, const T* c, T s, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)
        a[i] = b[i] + s * c[i];
}
// achieved_gbps = 3 * n * sizeof(T) / p50_seconds

// O3: register-resident FMA peak, no memory traffic, unrolled for ILP.
__global__ void fma_peak(float* sink, int iters) {
    float x0=threadIdx.x, x1=1.1f, x2=2.2f, x3=3.3f, a=0.9f, b=1.0001f;
    #pragma unroll 1
    for (int i=0;i<iters;i++){ x0=x0*a+b; x1=x1*a+b; x2=x2*a+b; x3=x3*a+b; }
    if (x0==-1.0f) sink[threadIdx.x]=x0+x1+x2+x3;   // keep it live
}
// flops = 2 (FMA) * 4 (ILP lanes) * iters * total_threads

// O4: one instruction ⇒ measures launch+teardown.
__global__ void one_op(int* p){ if (threadIdx.x==0xffff) p[0]=1; }

// O6: sustained high-power FMA to trip a lowered power cap (reuse O3 with big iters, many blocks).
// O7: a fixed cuBLAS GEMM (M=N=K per the per-SKU calibration table).
```

### Appendix B — `Result` / JSON schema (v1, abbreviated)

```jsonc
{
  "schema_version": "1",
  "caliper_version": "0.3.0",
  "measured_at": "2026-09-20T14:03:11Z",
  "host_id_class": "sha256:...",              // salted, non-identifying
  "kernel": { "name": "matmul_kernel", "impl": "triton",
              "source_hash": "sha256:...", "autotune_config": {"BLOCK_M":128, "...":"..."},
              "dtype": "bf16", "shape": {"M":4096,"N":4096,"K":4096}, "layout": "row" },
  "timing": { "p10_us": 241.0, "p50_us": 243.2, "p90_us": 250.1, "mad_us": 1.4,
              "wall_p50_us": 254.0, "launch_overhead_us": 6.1,
              "n_samples": 300, "n_warmup_to_steady": 187, "invalidated_samples": 0,
              "cross_pass_cov": null },
  "roofline": { "achieved_tflops": 565.0, "roofline_pct": 0.86, "achieved_gbps": null,
                "arithmetic_intensity": 682.0, "ridge_point": 1560.0, "bound": "compute",
                "baseline_pct": 0.96, "baseline": "cublas" },
  "ptxas": { "regs_per_thread": 168, "smem_static_bytes": 99328, "smem_dynamic_bytes": 0,
             "spill_loads_bytes": 0, "spill_stores_bytes": 0, "local_bytes": 0, "stack_bytes": 0 },
  "occupancy": { "theoretical": 0.25, "achieved": 0.247, "active_warps_per_sm": 16, "waves": 2.0 },
  "clocks": { "sm_mhz": 2520, "mem_mhz": 10501, "locked": true, "lock_method": "nvml" },
  "throttle_reasons": [],
  "machine": { "gpu_name": "NVIDIA GeForce RTX 4090", "sm_arch": "sm_89", "vram_mib": 24564,
               "sm_count": 128, "l2_bytes": 75497472, "bar1_mib": 32768,
               "driver": "550.90.07", "cuda_runtime": "12.4", "cuda_driver": "12.4",
               "nvml_version": "12.550.90", "ecc": false, "mig": "disabled",
               "persistence_mode": true, "pcie_gen": 4, "pcie_width": 16,
               "toolkit": { "triton": "3.2.0", "torch": "2.6.0", "ptxas": "12.4.131", "nvcc": "12.4.131" } },
  "flags": []                                  // e.g. ["clocks-unlocked","throttled-samples-dropped"]
}
```

Required fields, tolerance rules, and each invalid class are enumerated in
`caliper-results/schema/` and enforced by `caliper validate`.

### Appendix C — Parquet row

One row per measurement = `Result` flattened with dotted names
(`timing.p50_us`, `ptxas.regs_per_thread`, `machine.sm_arch`, …). Partition
columns: `machine.sm_arch`, `kernel.name`, `toolchain_hash`
(= sha256 of the sorted `machine.toolkit` map + `machine.driver`).

### Appendix D — `sweep` spec YAML

```yaml
schema_version: 1
target: corpus:gemm            # or  path/to/kernels.py::my_kernel
dtypes:   [bf16, fp16, fp8_e4m3]
layouts:  [row, col]
shapes:   llm-7b               # named library, or an inline list of {M,N,K}
bench:
  warmup: auto
  min_samples: 200
  flush_l2: true
  lock_clocks: true
  cuda_graph: auto
autotune: from_kernel          # use the kernel's own configs; time each separately
output:
  parquet: results/gemm-sweep.parquet
  resume: true
```

### Appendix E — `selftest` report schema (abbreviated)

```jsonc
{
  "schema_version": "1", "caliper_version": "0.3.0",
  "machine": { "...": "see Appendix B" },
  "result": "PASS",                         // PASS | FAIL | ERROR
  "coverage": "full",                       // full | reduced (no nsys)
  "checks": [
    { "name": "o1_duration_linearity", "status": "PASS",
      "measured": {"slope": 1.006, "intercept_us": 6.2},
      "expected": {"slope_range": [0.97, 1.03]}, "tolerance": "3%", "detail": "..." },
    { "name": "vs_nsys", "status": "SKIP", "detail": "nsys not on PATH" }
  ]
}
```

### Appendix F — Manual acceptance report template

```md
# caliper v0.3.0 — acceptance report
- Arch / GPU: sm_89 / RTX 4090          - Host: <box>          - Date: 2026-09-2x
- Tier: 1                                - caliper: 0.3.0 (from PyPI)
- Tools present: nsys 2026.x, ncu 2026.x, nvbandwidth <sha>

| Step | Expected | Measured | Pass? | Notes |
|------|----------|----------|-------|-------|
| 1 install clean | v0.3.0 | v0.3.0 | ✅ | |
| 2 doctor fields | byte-exact vs nvidia-smi | (attach checklist) | ✅ | |
| 3 selftest --full | 9/9 | 9/9 | ✅ | report attached |
| 4 O1 linearity | slope∈[0.97,1.03] | 1.006 | ✅ | plot attached |
| 5 L2 flush A/B | small ≥2×, large <5% | 3.7× / 2.1% | ✅ | |
| 6 nvbandwidth | ±5% | +2.8% | ✅ | |
| 7 cuBLAS vs ncu | dur≤3%, regs exact, occ±0.05, tflops≤3% | (attach table) | ✅ | |
| 8 reproducibility | locked CoV<2%, unlocked>locked | 0.9% / 3.4% | ✅ | |
| 9 throttle | flagged + samples dropped | SW_POWER_CAP, 214 | ✅ | |
| 10 do_bench shim | runs, ±3% | +1.1% | ✅ | |
| 11 sweep+resume | resumes, valid parquet | | ✅ | |
| 12 compare regression | exit 1 + spill delta shown | | ✅ | |
| 13 submit dry-run | well-formed bundle | | ✅ | |
| 14 negative validate | 4/4 rejected correctly | | ✅ | |

## NFR results
NFR-1 …%  NFR-2 …%  NFR-3 …%  NFR-4 …  NFR-5 …%  NFR-6 …s

## Deviations / triage
(none)  |  <describe, link issue>
```

### Appendix G — Per-arch peaks table: sources to cite in code

For each `(arch, dtype)` cell in `roofline.py`, cite one of: the NVIDIA
architecture whitepaper (Ampere GA100 / Ada AD102 / Hopper GH100 / Blackwell),
the "Dissecting the NVIDIA <arch> Architecture via Microbenchmarking" paper for
that gen, or a measured value from `caliper selftest` on that SKU (mark measured
cells `# source: measured, selftest-<sku>.json`). HBM GB/s cells should prefer the
**measured** O2 sustained value over the datasheet, with the datasheet in a
comment.

### Appendix H — Environment & hardware plan (resource-adjusted)

**Pure-core dev host — MacBook Pro (Apple Silicon).**
- Rust toolchain via [rustup](https://rustup.rs) (`rustc` + `cargo` + `clippy` +
  `rustfmt`); Python 3.11 via `uv` or `pyenv`; `pip install -e ".[dev]"` (maturin
  builds `caliper._core`).
- Runs: **all of `caliper-core` (`cargo test` = L0)**, `caliper-gpu` built with
  the `fixture` feature (L1), the Python bindings + tests, the CLI, docs,
  packaging, `ci-cpu` config. No CUDA, no Triton, no GPU — by design. The
  CUDA-touching Rust (`caliper-gpu` `real`) and the `.cu` kernels only *compile*
  where a CUDA toolkit is present (Colab / golden box).
- `make sync` pushes the branch and prints the Colab bootstrap one-liner.

**GPU validation host — Google Colab (primary).**
- Target runtimes: **A100-40GB** (workhorse), **L4** (fp8 arch), **T4**
  (fp16/fp32 + the thermal-throttle O6 fallback), **V100** if it appears (Tier-2).
- **Colab does not need Rust** for a released build — `pip install caliper-gpu`
  pulls a prebuilt wheel with the compiled core. For a from-source dev run the
  first notebook cell adds it:
  `!curl -sSf https://sh.rustup.rs | sh -s -- -y && source $HOME/.cargo/env && pip -q install -e . && (apt-get -qq install -y nsight-systems-cli 2>/dev/null || true); caliper doctor`
  (`nsys` installs cleanly; `ptxas` / `cuobjdump` are already present; `ncu` will
  fail to profile — expected). Build `nvbandwidth` once per runtime (~2 min) or
  vendor a prebuilt binary in `notebooks/bin/`.
- Notebooks (committed): `dev.ipynb`, `selftest.ipynb`, `corpus_sweep.ipynb`,
  `acceptance.ipynb`. Pro+ background execution for sweeps > 60 min.
- Colab **cannot**: lock clocks, grant `ncu` counters, set a power cap, host a
  persistent runner. Everything that needs those is on the golden box.

**Paid rentals — $100 budget, spent at acceptance (Week 4).**

| Spend | GPU | ~Rate | Hours | Purpose |
|-------|-----|-------|-------|---------|
| ~$10 | **Lambda Cloud on-demand A100-40GB** (golden box — reliably root + counters + `nvidia-smi -lgc`) | ~$1.29/h | 8 | FR-4 clock-lock, `ncu` L3 (5 kernels), O6 power-cap, locked NFR-5, O7 calibration. |
| ~$16 | **H100** (SM90), RunPod | ~$2/h | 8 | The Hopper Tier-1 arch; 2nd `nsys` arch; corpus rows. |
| ~$5 | **RTX 4090** (consumer SM89), Vast — *verify clock-lock + counters first* | ~$0.5/h | ~8 | Consumer-Ada datapoint; a 2nd locked-clock check. |
| ~$8 | **RTX 3090** (SM86) spot, Vast | ~$0.25/h | ~1 | Ampere-consumer cell (best-effort). |
| ~$10 | **MI300X** (CDNA3), RunPod | ~$2.5/h | ~1 | ROCm "runs & emits a valid row" (Tier-2). |
| **~$51** | **reserve** | — | — | Re-rentals after fixes; a 2nd H100 / golden-box pass; spillover. Do not pre-spend. |

Providers: **Lambda** (recommended for the golden box — known-permissive), RunPod
(H100, MI300X), Vast.ai (cheap community 4090/3090 — verify the instance grants
clock-lock + counter access before relying on it). **In the first 5 minutes of
the golden-box rental, confirm `sudo nvidia-smi -lgc <freq>` and a trivial `ncu`
run both succeed; if not, kill it and try another provider.**

**If you only have Colab Pro (not Pro+):** consider spending ~$50 of the budget on
one month of **Pro+** for the build month (reliable A100 + background execution),
compressing rentals to golden-box + H100 only (~$26). Log the choice in
`docs/DESCOPE.md`.

**CI:** `ci-cpu` runs on GitHub-hosted runners (driven from the Mac). There is no
persistent `ci-gpu` runner — `notebooks/dev.ipynb` on Colab is the GPU CI, run on
demand after GPU-affecting pushes (schedulable on Pro+).
