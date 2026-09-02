"""L0: the release path is wired up and honest.

`.github/workflows/release.yml` must build an sdist + wheels, go through Test
PyPI before PyPI, and cut a GitHub Release; `RELEASING.md` must gate the tag on
the plan's Definition of Done; and the crate version must agree with the
newest released section of the changelog (and stay at the pre-release `0.0.1`
until the first release is cut).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.l0

REPO = Path(__file__).resolve().parents[2]
WF = (REPO / ".github" / "workflows" / "release.yml").read_text()
RELEASING = (REPO / "RELEASING.md").read_text()
CARGO = (REPO / "Cargo.toml").read_text()
CHANGELOG = (REPO / "CHANGELOG.md").read_text()
TRACE = (REPO / "docs" / "acceptance" / "traceability.md").read_text()


def _workflow() -> Any:
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(WF)
    # YAML 1.1 parses a bare `on:` key as the boolean True
    doc["on"] = doc.get("on", doc.get(True))
    return doc


def _crate_version() -> str:
    # the workspace version -- the anchored `version = "..."` line, not
    # `rust-version` and not a dependency's inline `version = "1"`.
    m = re.search(r'(?m)^version = "([^"]+)"', CARGO)
    assert m, 'no `version = "..."` line in Cargo.toml'
    return m.group(1)


def _released_versions() -> list[str]:
    # `## [X.Y.Z]` changelog headings, newest first; `## [Unreleased]` excluded.
    return re.findall(r"(?m)^## \[(\d+\.\d+\.\d+)\]", CHANGELOG)


def test_release_workflow_triggers_on_version_tags() -> None:
    on = _workflow()["on"]
    assert "workflow_dispatch" in on
    assert on["push"]["tags"] == ["v*"]


def test_release_workflow_builds_sdist_and_wheels() -> None:
    steps = _workflow()["jobs"]["build"]["steps"]
    commands = {s.get("with", {}).get("command") for s in steps if isinstance(s, dict)}
    assert "sdist" in commands
    assert "build" in commands
    # the wheel step names each supported interpreter
    wheel = next(s for s in steps if s.get("with", {}).get("command") == "build")
    args = wheel["with"]["args"]
    for py in ("python3.10", "python3.11", "python3.12"):
        assert py in args


def test_release_goes_through_test_pypi_before_pypi() -> None:
    jobs = _workflow()["jobs"]
    assert "test.pypi.org/legacy/" in WF
    # build -> test-pypi -> pypi -> github-release, in that order
    assert jobs["test-pypi"]["needs"] == "build"
    assert jobs["pypi"]["needs"] == "test-pypi"
    assert jobs["github-release"]["needs"] == "pypi"
    # PyPI + GitHub Release are tag-only, not on workflow_dispatch
    assert "startsWith(github.ref, 'refs/tags/')" in jobs["pypi"]["if"]
    assert "startsWith(github.ref, 'refs/tags/')" in jobs["github-release"]["if"]


def test_release_creates_a_github_release_with_acceptance_evidence() -> None:
    jobs = _workflow()["jobs"]
    assert "github-release" in jobs
    assert "selftest-*.json" in WF
    assert re.search(r"\*ncu\*\.md", WF), "release notes should attach the ncu report"
    assert "why-do_bench-misleads.md" in WF


def test_release_workflow_checks_the_tag_matches_the_crate_version() -> None:
    # the guard both reads an anchored `^version` line and compares to the tag
    assert "does not match Cargo.toml version" in WF
    assert re.search(r"\^version = ", WF)
    assert "GITHUB_REF_NAME#v" in WF


def test_releasing_doc_gates_the_tag_on_the_definition_of_done() -> None:
    assert "§5" in RELEASING or "Definition of Done" in RELEASING
    assert "traceability.md" in RELEASING
    assert "triage.md" in RELEASING
    assert "tier2.md" in RELEASING
    # the version-bump step and the post-release check
    assert "Cargo.toml" in RELEASING
    assert "CONSTRAINED" in RELEASING


def test_crate_version_matches_the_changelog() -> None:
    version = _crate_version()
    released = _released_versions()
    # there is always an open section to collect the next release's notes
    assert "## [Unreleased]" in CHANGELOG
    if released:
        # newest released heading must be the version the repo currently carries
        assert version == released[0], (
            f"Cargo.toml is {version} but the newest CHANGELOG release is {released[0]}"
        )
    else:
        # nothing released yet -- stay on the pre-release version
        assert version == "0.0.1", f"Cargo.toml is {version} with no released CHANGELOG section"


def test_triage_and_tier2_docs_exist_and_are_linked() -> None:
    assert (REPO / "docs" / "acceptance" / "triage.md").exists()
    assert (REPO / "docs" / "acceptance" / "tier2.md").exists()
    assert (REPO / ".github" / "ISSUE_TEMPLATE" / "acceptance-deviation.md").exists()
    assert "triage.md" in TRACE
    assert "tier2.md" in TRACE
