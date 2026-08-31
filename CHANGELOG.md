# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from the first
tagged release onward.

## [Unreleased]

### Added
- Project scaffold: `src/` layout, packaging (`caliper-gpu`), Ruff + Mypy
  configuration, and pytest markers (`l0`/`l1`/`l2`/`l3`/`l4`/`l6`).
- `caliper._internal.schema`: the `Result` record and its nested sections
  (`KernelLabel`, `Timing`, `Roofline`, `Ptxas`, `Occupancy`, `Clocks`,
  `Machine`, `Toolkit`), with `to_dict()` / `from_dict()` and a JSON round-trip
  test.
- Minimal `caliper` CLI entry point (`--version`, `--help`).
- Continuous integration for the no-GPU test surface (Ruff, Mypy, `l0`/`l1`
  tests, package build) across Python 3.10-3.12.
