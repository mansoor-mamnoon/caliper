# Releasing caliper

A release is a git tag `vX.Y.Z`. Pushing that tag runs
[`.github/workflows/release.yml`](.github/workflows/release.yml), which builds
the sdist and the Linux wheels, publishes to Test PyPI, then -- after a manual
approval -- to PyPI, and opens a GitHub Release with the acceptance evidence
attached.

## Before you tag

The tag is only pushed once **every box in `docs/plan.md` §5 (Definition of
Done) is checked**. In particular:

- [ ] `ci-cpu` is green on the commit you're about to tag (L0 + L1, all
      Python versions, the Rust suite with and without `--all-features`).
- [ ] `docs/acceptance/reports/` has a filled, all-PASS report for every Tier-1
      arch (T4, A100, L4, H100, golden-box A100) and a filed report for each
      reachable Tier-2 arch. See [`docs/acceptance/tier2.md`](docs/acceptance/tier2.md).
- [ ] Every deviation logged in [`docs/acceptance/triage.md`](docs/acceptance/triage.md)
      is closed (Tier-1) or explicitly waived as non-blocking (Tier-2).
- [ ] `docs/acceptance/traceability.md` has no `pending` row -- every FR / NFR
      points at a passing test or a filled playbook step.
- [ ] The golden-box `ncu` L3 report is committed under
      `docs/acceptance/reports/`.
- [ ] `CHANGELOG.md` has everything since the last release under
      `## [Unreleased]`.

## Cutting the release

1. **Bump the version.** Edit `version` in the workspace `Cargo.toml` (the
   Python package version is derived from it by maturin). Use
   [semantic versioning](https://semver.org/); the first public release is
   `0.3.0`.
2. **Update the changelog.** Rename `## [Unreleased]` to
   `## [X.Y.Z] - YYYY-MM-DD` and add a fresh empty `## [Unreleased]` above it.
3. **Commit** those two edits together: `Release vX.Y.Z`.
4. **Tag and push:**
   ```bash
   git tag vX.Y.Z
   git push origin main vX.Y.Z
   ```
   The workflow refuses to publish if the tag doesn't match the `Cargo.toml`
   version, so step 1 and the tag must agree.
5. **Approve the PyPI step.** The `test-pypi` job runs automatically; the `pypi`
   job waits on the `pypi` GitHub Environment -- approve it once the Test PyPI
   artifacts look right (`pip install -i https://test.pypi.org/simple/
   caliper-gpu` in a throwaway venv).

Both PyPI uploads use [trusted publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC) -- there are no API tokens to manage, but the project must be registered
as a trusted publisher on both PyPI and Test PyPI, tied to this workflow and the
`pypi` / `test-pypi` environments.

## After the release

- [ ] In a **fresh Colab runtime**: `pip install caliper-gpu` with no other
      steps, then `caliper doctor` -- it must run immediately and print
      `environment: CONSTRAINED` (NFR-10).
- [ ] The GitHub Release lists the wheels, the sdist, every
      `selftest-*.json`, the golden-box `ncu` report, and the do_bench writeup.
- [ ] Announce, then start the next `## [Unreleased]` cycle.

## A rehearsal without tagging

Run the workflow from the Actions tab (`workflow_dispatch`). It builds the
artifacts and pushes them to Test PyPI (`skip-existing` is on), and stops there
-- the `pypi` and `github-release` jobs are tag-only.
