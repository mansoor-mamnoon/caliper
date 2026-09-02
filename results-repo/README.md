# caliper-results (scaffold)

The community results repository for [caliper](https://github.com/mansoor-mamnoon/caliper).
This directory is the seed that gets split into its own `caliper-results` repo;
it is kept here so the layout, the submission guide, and the validation
workflow evolve together with the tool.

## Layout

```
results/<sm_arch>/<toolchain-hash-16>/
    manifest.json      # caliper submit output; see caliper_core::submit::Manifest
    rows.parquet       # one row per measurement (Appendix C flat schema)
    fingerprint.json   # the machine block, must agree with every row's machine
```

`<toolchain-hash-16>` is the first 16 hex chars of the Appendix-C partition key
(`sha256` of the sorted `machine.toolkit` map + `machine.driver`).

The layout has no per-`<kernel>` segment: one `caliper submit` bundle spans
every kernel a submitter swept, so a bundle directory is keyed by `(arch,
toolchain)` and the kernel breakdown lives inside `manifest.json`.

## Submitting

See [`SUBMITTING.md`](SUBMITTING.md). In short: `caliper submit <your.parquet>
--out bundle/`, check it with `caliper validate bundle/`, then open a PR that
adds it under `results/`.

## Validation

Every PR runs [`.github/workflows/validate.yml`](.github/workflows/validate.yml):
`caliper validate` on each changed bundle directory. That gate is exactly the
one `caliper validate <dir>` runs locally -- schema validity, the
submission-strict field / roofline checks, the bundle-level determinism,
calibration, and arch-consistency checks, and a within-bundle exact-duplicate
row check. The manifest's own verdicts are not trusted: the tier, kernel list,
and the determinism / calibration outcomes are recomputed from `rows`.

**Cross-bundle dedupe** -- rejecting a PR whose `(facet, arch, toolchain_hash)`
already exists under `results/` -- is not implemented yet; it lands with the
first real merges, once there is a corpus to dedupe against.
