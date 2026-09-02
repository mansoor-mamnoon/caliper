# selftest-reports

`caliper selftest --full --json` output, one `selftest-<arch>.json` per
architecture the acceptance playbook has run on (`notebooks/acceptance.ipynb`
writes these; commit them here). `.github/workflows/release.yml` attaches every
file matching `selftest-*.json` to the GitHub Release, and refuses to cut a
tagged release if this directory is empty.

The filled human-readable reports live in `docs/acceptance/reports/`.
