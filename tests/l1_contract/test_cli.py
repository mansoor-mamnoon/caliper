"""L1: the CLI subcommands over recorded sessions (moved from l0 now that they
drive the device layer)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from caliper import __version__
from caliper.cli import main

pytestmark = pytest.mark.l1

BENCH = Path(__file__).resolve().parents[2] / "crates" / "caliper-gpu" / "fixtures" / "bench"
DOCTOR = Path(__file__).resolve().parents[2] / "crates" / "caliper-gpu" / "fixtures" / "doctor"


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_args_prints_help_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: caliper" in capsys.readouterr().out


def test_unknown_command_is_rejected_by_argparse(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["wat"])
    assert exc.value.code == 2


def test_bench_without_a_recording_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "corpus:o1"]) == 2
    assert "needs --recording" in capsys.readouterr().err


def test_bench_replays_a_recording(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["bench", "k", "--recording", str(BENCH / "happy.jsonl"), "--batches", "40"])
    assert code == 0
    assert "p50" in capsys.readouterr().out


def test_bench_corpus_target_resolves(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "bench",
            "corpus:o1",
            "--recording",
            str(BENCH / "oracle_o1.jsonl"),
            "--batches",
            "40",
            "--json",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["kernel"]["name"] == "oracle:busy"


def test_bench_unknown_corpus_target_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["bench", "corpus:o9", "--recording", str(BENCH / "oracle_o1.jsonl")])
    assert code == 2
    assert "unknown corpus target" in capsys.readouterr().err


def test_bench_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["bench", "k", "--recording", str(BENCH / "happy.jsonl"), "--batches", "40", "--json"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "1"
    assert 198.0 < payload["timing"]["p50_us"] < 202.0


def test_bench_bad_warmup_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "k", "--recording", str(BENCH / "happy.jsonl"), "--warmup", "soon"]) == 2
    assert "warmup" in capsys.readouterr().err


def test_doctor_fit_recording_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["doctor", "--recording", str(DOCTOR / "fit.jsonl")])
    assert code == 0
    assert "FIT TO BENCHMARK" in capsys.readouterr().out


def test_doctor_constrained_text_matches_the_honest_degradation_spec(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["doctor", "--recording", str(DOCTOR / "constrained.jsonl")])
    assert code == 0
    out = capsys.readouterr().out
    assert "FIT TO BENCHMARK (reduced confidence)" in out
    assert "constrained (Colab-like)" in out
    assert "clocks-unlocked" in out


def test_doctor_throttling_recording_exits_one(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "--recording", str(DOCTOR / "throttling.jsonl")]) == 1


def test_doctor_no_device_recording_exits_two() -> None:
    assert main(["doctor", "--recording", str(DOCTOR / "no_device.jsonl")]) == 2


def test_doctor_from_env_without_a_gpu_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    # No CUDA build feature -> open_from_env yields "no device".
    assert main(["doctor"]) == 2


def test_doctor_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["doctor", "--recording", str(DOCTOR / "constrained.jsonl"), "--json"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["environment"] == "constrained"
    assert report["verdict"] == "fit"


def test_fingerprint_from_a_recording(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["fingerprint", "--recording", str(DOCTOR / "fit.jsonl"), "--json"])
    assert code == 0
    machine = json.loads(capsys.readouterr().out)
    assert machine["sm_arch"] == "sm_89"


def test_fingerprint_from_env_without_a_gpu_exits_two() -> None:
    assert main(["fingerprint"]) == 2


def test_fingerprint_check_on_a_complete_recording(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["fingerprint", "--recording", str(DOCTOR / "fit.jsonl"), "--check", "--json"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["complete"] is True
    assert report["missing_required"] == []


def test_validate_a_clean_parquet(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("pyarrow")
    from caliper import Grid, Result

    p = tmp_path / "g.parquet"
    Grid([Result.default().to_dict()]).to_parquet(p)
    code = main(["validate", str(p), "--json"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True and report["n"] == 1


def test_validate_a_bad_json_file_exits_one(tmp_path: Path) -> None:
    from caliper import Result

    rec = Result.default().to_dict()
    rec["timing"]["p50_us"] = -1.0
    p = tmp_path / "g.json"
    p.write_text(json.dumps([rec]))
    assert main(["validate", str(p)]) == 1


def test_selftest_without_a_gpu_is_an_error_report(capsys: pytest.CaptureFixture[str]) -> None:
    from caliper import _core

    code = main(["selftest", "--full", "--json"])
    assert code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["result"] == "ERROR"
    assert report["coverage"] == "reduced"
    assert "exit_code" not in report  # the --json body is pure Appendix E
    assert set(report["not_validated"]) == {"clock_lock", "ncu_crosscheck", "powercap_throttle"}
    assert _core.validate_selftest_json(json.dumps(report)) == []
