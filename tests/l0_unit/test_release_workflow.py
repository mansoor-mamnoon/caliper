"""L0: the release path is wired up and honest.

`.github/workflows/release.yml` must build an sdist + wheels, go through Test
PyPI before PyPI, and cut a GitHub Release; `RELEASING.md` must gate the tag on
the plan's Definition of Done; and the repo must not carry a version number
ahead of an actual release.
"""

from __future__ import annotations

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


def test_release_workflow_triggers_on_version_tags() -> None:
    on = _workflow()["on"]
    assert "workflow_dispatch" in on
    assert on["push"]["tags"] == ["v*"]


def test_release_workflow_builds_sdist_and_wheels() -> None:
    jobs = _workflow()["jobs"]
    build = jobs["build"]
    steps = "\n".join(str(s) for s in build["steps"])
    assert "command: sdist" in WF or "sdist" in steps
    assert "command: build" in WF or "maturin" in steps.lower()
    # wheels for the supported interpreter range
    for py in ("3.10", "3.11", "3.12"):
        assert f"python{py}" in WF


def test_release_goes_through_test_pypi_before_pypi() -> None:
    jobs = _workflow()["jobs"]
    assert "test.pypi.org/legacy/" in WF
    # the real-PyPI job must depend (transitively) on the Test PyPI one
    assert jobs["pypi"]["needs"] == "test-pypi"
    assert jobs["test-pypi"]["needs"] == "build"
    # PyPI + GitHub Release are tag-only, not on workflow_dispatch
    assert "startsWith(github.ref, 'refs/tags/')" in jobs["pypi"]["if"]
    assert "startsWith(github.ref, 'refs/tags/')" in jobs["github-release"]["if"]


def test_release_creates_a_github_release_with_acceptance_evidence() -> None:
    jobs = _workflow()["jobs"]
    assert "github-release" in jobs
    assert "selftest-*.json" in WF
    assert "ncu" in WF
    assert "why-do_bench-misleads.md" in WF


def test_release_workflow_checks_the_tag_matches_the_crate_version() -> None:
    assert "does not match Cargo.toml version" in WF


def test_releasing_doc_gates_the_tag_on_the_definition_of_done() -> None:
    assert "§5" in RELEASING or "Definition of Done" in RELEASING
    assert "traceability.md" in RELEASING
    assert "triage.md" in RELEASING
    assert "tier2.md" in RELEASING
    # the version-bump step and the post-release check
    assert "Cargo.toml" in RELEASING
    assert "CONSTRAINED" in RELEASING


def test_repo_version_is_not_ahead_of_a_release() -> None:
    # while the changelog still has an open [Unreleased] section, the crate
    # version stays at the pre-release 0.0.1 -- we don't stamp 0.3.0 until the
    # release is actually cut (RELEASING.md step 1 + 2, done together).
    assert "## [Unreleased]" in CHANGELOG
    assert 'version = "0.0.1"' in CARGO


def test_triage_and_tier2_docs_exist_and_are_linked() -> None:
    assert (REPO / "docs" / "acceptance" / "triage.md").exists()
    assert (REPO / "docs" / "acceptance" / "tier2.md").exists()
    assert (REPO / ".github" / "ISSUE_TEMPLATE" / "acceptance-deviation.md").exists()
    assert "triage.md" in TRACE
    assert "tier2.md" in TRACE
