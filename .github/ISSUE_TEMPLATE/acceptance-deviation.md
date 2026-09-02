---
name: Acceptance deviation
about: A manual-playbook step whose measured result didn't meet the expected one
title: "[acceptance] sm_XX: step N — <short description>"
labels: acceptance
---

<!-- One issue per deviating step. Also add a row to docs/acceptance/triage.md. -->

## Where

- **Arch / GPU:** sm_XX / <card>
- **Tier:** 1 (release-blocking) | 2 (best-effort)
- **Host:** <box> (Colab / Lambda / Vast / …)
- **caliper version:** <version> · **commit:** <sha>
- **Playbook step:** <number> — <name>

## What happened

- **Expected:** <from the playbook's Expected column>
- **Measured:** <what actually came back — paste the number / output>
- **Command:** <exact command run>

## Evidence

<!-- attach the report .md, selftest-*.json, ncu table, nsys/nvbandwidth output -->

## Triage

- **Suspected cause:**
- **Fix:** <PR / commit>
- **Re-run:** <result of re-running just this step on the same arch>
