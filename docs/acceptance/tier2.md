# Tier-2 acceptance (best-effort)

Tier-2 architectures are exercised when budget allows. They are **not
release-blocking**: a Tier-2 report is filed with triage notes
([`triage.md`](triage.md)), and an open deviation on a Tier-2 arch can be waived
for the release with a reason.

See `docs/plan.md` §1.3 for the full tier table. The reachable Tier-2 targets
for this release:

| Arch | Rep GPU | Bar |
|------|---------|-----|
| SM86 | RTX 3090 (consumer Ampere) | full [Playbook A](manual-playbook.md) |
| CDNA3 | MI300X (ROCm) | `caliper doctor` + **one** corpus kernel + `caliper validate` |

## SM86 — RTX 3090

Run Playbook A exactly as for a Colab arch (steps 1–8, 10–14; step 9 uses the
thermal-throttle fallback). Consumer cards often can't lock clocks -- if
`caliper doctor` reports clocks unlockable, step 8 records the **unlocked** tier
only and that is expected, not a deviation. File
`reports/sm86-<host>-<date>.md`.

## CDNA3 — MI300X

The ROCm bar is only *"caliper runs and emits a valid row"*:

1. `caliper doctor --json` — a verdict is produced; the AMD fields are populated
   (no hard error, no CUDA assumption).
2. One corpus kernel end to end, e.g.
   `caliper sweep examples/acceptance-sweep.yaml --parquet rows-mi300x.parquet`
   reduced to a single cell, **or** `caliper bench corpus:gemm` on a recorded
   session — whichever the ROCm launcher supports at the time.
3. `caliper validate rows-mi300x.parquet` — the row passes the schema.

Anything past that (roofline accuracy, occupancy, `ncu`/`rocprof` cross-checks)
is out of scope for this release. File `reports/mi300x-<host>-<date>.md` with a
one-line status and a triage note for whatever didn't work.

## Report template (Tier-2)

```md
# caliper acceptance report (Tier-2, best-effort)
- Arch / GPU: sm_XX / <card>     - Host: <box>     - Date: 2026-XX-XX
- Tier: 2 (not release-blocking) - caliper: <version>

| Step | Expected | Measured | Pass? | Notes |
|------|----------|----------|-------|-------|
| ...  |          |          |       |       |

## Deviations / triage
(none)  |  <describe, link issue, mark waived-for-release if so>
```
