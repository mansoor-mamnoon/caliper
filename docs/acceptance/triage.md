# Acceptance triage log

Every deviation from an expected result in the [manual playbook](manual-playbook.md)
-- on any arch, Tier-1 or Tier-2 -- is recorded here, fixed or waived, and the
affected step is re-run before a row in
[`traceability.md`](traceability.md) flips to PASS.

## The loop

1. **File.** A playbook step whose *Measured* column doesn't meet *Expected* is
   a deviation. Open an issue with the
   [acceptance-deviation template](../../.github/ISSUE_TEMPLATE/acceptance-deviation.md)
   and add a row to the log below (status `open`).
2. **Classify.**
   - *Tier-1 arch* (T4, A100, L4, H100, golden-box A100): **release-blocking.**
   - *Tier-2 arch* (see [`tier2.md`](tier2.md)): **not** release-blocking; it is
     filed, triaged, and may be waived for the release with a note.
3. **Fix.** Land the change on `main`; `ci-cpu` must stay green.
4. **Re-run.** Re-run only the affected playbook step (and `caliper selftest
   --full`) on the arch that deviated. Use the rental reserve if the fix needs
   the golden box or an H100 again.
5. **Close.** Update the report's *Deviations / triage* block with the outcome,
   set the log row to `fixed` (or `waived`, Tier-2 only), and link the commit.
   Flip the `traceability.md` row once its evidence is a clean re-run.

A release goes out only when every Tier-1 row here is `fixed` and every Tier-2
row is `fixed` or `waived` (`RELEASING.md`, "Before you tag").

## Log

| ID | Arch | Tier | Playbook step | Deviation | Issue | Fix commit | Status |
|----|------|------|---------------|-----------|-------|-----------|--------|
| _(none yet -- the on-device passes haven't run)_ | | | | | | | |

Status: `open` · `fixed` · `waived` (Tier-2 only, with a reason in the notes).
