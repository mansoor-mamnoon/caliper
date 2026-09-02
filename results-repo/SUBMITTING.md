# Submitting your GPU's numbers

Thank you for adding to the corpus. Every submission is a small, self-describing
bundle; the PR gate is the same `caliper validate` you can run locally.

## 1. Measure

On the machine you want to profile:

```bash
pip install caliper-gpu
caliper selftest --full --json > selftest.json      # confirms the box is fit
caliper sweep corpus.yaml --parquet rows.parquet    # or your own kernels
```

`selftest --full` also prints the O7 calibration-GEMM result -- note its `p50_us`
and the expected value for your SKU (`expected_p50_us` in the report).

## 2. Build the bundle

```bash
caliper submit rows.parquet --out bundle/ --calibration <measured_us> <expected_us>
caliper validate bundle/          # must print OK
```

`caliper submit` writes `bundle/manifest.json`, `bundle/rows.parquet`, and
`bundle/fingerprint.json`. The manifest records your caliper version, the
architecture, the toolchain hash, the clock-lock tier, and -- when present -- a
determinism-repeat summary and the calibration check.

## 3. Open the PR

Copy `bundle/` to `results/<sm_arch>/<toolchain-hash-16>/` and open a PR. If you
have a local checkout of this repo:

```bash
caliper submit rows.parquet --repo /path/to/caliper-results --calibration <m> <e>
```

writes the bundle onto a fresh branch there; push it and open the PR yourself.

## What the gate checks (and rejects)

| Rejection | Cause |
|---|---|
| `missing required field <path>` | a row without `measured_at`, `kernel.name`, `kernel.dtype`, `timing.p50_us`, `timing.n_samples`, or `machine.sm_arch` |
| `roofline.roofline_pct ... claims more than 1.05 of peak` | an impossible efficiency (bad FLOP count or a mislabelled peak) |
| `determinism repeat CoV ...% exceeds the ...% tolerance` | repeated measurements of one facet disagree past the tier's CoV bound (2% locked, 5% unlocked) |
| `calibration GEMM p50 is ...% of expected` | the SKU calibration GEMM ran more than 8% off its expectation -- the clocks were probably not what the fingerprint says |
| `fingerprint.sm_arch ... != manifest arch` / `row ... machine.sm_arch ...` | the bundle mixes architectures |
| `manifest tier ... != the rows' tier` / `manifest kernels ...` | a hand-edited manifest that disagrees with the rows (the gate recomputes both) |
| `row N: exact duplicate of an earlier row` | the same measurement appears twice in one bundle |
| any `caliper validate` schema problem | a malformed record (bad percentile ordering, out-of-range occupancy, ...) |
