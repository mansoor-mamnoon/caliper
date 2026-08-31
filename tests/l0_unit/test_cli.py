"""L0: the CLI entry point handles --version, --help, and unknown commands."""

from __future__ import annotations

import pytest

from caliper import __version__
from caliper.cli import main

pytestmark = pytest.mark.l0


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_args_prints_help_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: caliper" in capsys.readouterr().out


def test_unknown_command_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench"]) == 2
    assert "not available yet" in capsys.readouterr().err
