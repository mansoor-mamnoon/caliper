"""The public measurement entry point.

``bench()`` drives the device layer through one measurement and returns a
:class:`~caliper.Result`. Today only the recorded-session path is wired up: pass
``fixture=`` a path to a JSON Lines device recording, or ``recording=`` its text.
The on-device launcher (real CUDA events, clock locking) is implemented and
validated on a CUDA host.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from caliper import _core
from caliper._record import Result

_GRAPH_MODES = ("auto", "on", "off")


def _warmup_plan(warmup: str | int, window: int, tol: float, min_warm: int) -> dict[str, Any]:
    opts = {"window": window, "tol": tol, "min_warm": min_warm}
    if warmup == "auto":
        return {"fixed": None, "opts": opts}
    if isinstance(warmup, bool) or not isinstance(warmup, int):
        raise ValueError(f"warmup must be 'auto' or an int, got {warmup!r}")
    if warmup < 0:
        raise ValueError(f"a fixed warmup must be non-negative, got {warmup}")
    return {"fixed": warmup, "opts": opts}


def _graph_mode(cuda_graph: str | bool) -> str:
    if isinstance(cuda_graph, bool):
        return "on" if cuda_graph else "off"
    if cuda_graph not in _GRAPH_MODES:
        raise ValueError(f"cuda_graph must be one of {_GRAPH_MODES} or a bool, got {cuda_graph!r}")
    return cuda_graph


def bench(
    target: Any = None,
    *,
    fixture: str | Path | None = None,
    recording: str | None = None,
    kernel_key: str = "kernel",
    kernel_impl: str | None = None,
    dtype: str | None = None,
    batch: int = 32,
    batches: int = 50,
    cuda_graph: str | bool = "auto",
    flush_l2: bool = True,
    lock_clocks: bool = True,
    sm_mhz: int | None = None,
    mem_mhz: int | None = None,
    warmup: str | int = "auto",
    warmup_window: int = 20,
    warmup_tol: float = 0.02,
    warmup_min: int = 30,
) -> Result:
    """Measure one kernel and return its :class:`Result`.

    Parameters
    ----------
    target:
        A kernel to measure directly. Not supported yet (needs the on-device
        launcher); pass ``fixture`` / ``recording`` instead.
    fixture:
        Path to a JSON Lines recording of a device session.
    recording:
        The text of such a recording (alternative to ``fixture``).
    warmup:
        ``"auto"`` for steady-state detection, or an ``int`` to trim exactly
        that many leading samples.
    cuda_graph:
        ``"auto"`` / ``"on"`` / ``"off"`` (or a bool).
    """
    if recording is None and fixture is None:
        raise NotImplementedError(
            "caliper.bench() currently runs from a recorded device session; pass "
            f"fixture=<path> or recording=<text>. (target={target!r} needs the "
            "on-device launcher, which runs on a CUDA host.)"
        )
    if recording is None:
        assert fixture is not None
        recording = Path(fixture).read_text()

    if isinstance(target, str) and target.startswith("corpus:"):
        resolved = _core.resolve_corpus_target(target)
        if resolved is None:
            names = ", ".join(t[0] for t in _core.corpus_targets())
            raise ValueError(f"unknown corpus target {target!r}; available: {names}")
        kernel_key = resolved

    opts = {
        "kernel_key": kernel_key,
        "kernel_impl": kernel_impl,
        "dtype": dtype,
        "batch": batch,
        "batches": batches,
        "cuda_graph": _graph_mode(cuda_graph),
        "flush_l2": flush_l2,
        "lock_clocks": lock_clocks,
        "clock_target": {"sm_mhz": sm_mhz, "mem_mhz": mem_mhz},
        "warmup": _warmup_plan(warmup, warmup_window, warmup_tol, warmup_min),
    }
    return Result.from_json(_core.bench_replay(recording, json.dumps(opts)))


def _read(fixture: str | Path | None, recording: str | None) -> str | None:
    if recording is not None:
        return recording
    if fixture is not None:
        return Path(fixture).read_text()
    return None


def doctor(
    *,
    fixture: str | Path | None = None,
    recording: str | None = None,
) -> dict[str, Any]:
    """Assess whether this machine is fit to benchmark.

    With no ``fixture`` / ``recording`` this probes the backend selected by
    ``CALIPER_GPU_PORTS`` (default: the real device, which is "no device found"
    on a build without CUDA). The returned dict has ``verdict``, ``environment``,
    ``checks``, ``notes``, and ``exit_code``.
    """
    text = _read(fixture, recording)
    raw = _core.doctor_from_env() if text is None else _core.doctor_replay(text)
    report: dict[str, Any] = json.loads(raw)
    return report


def doctor_text(
    *,
    fixture: str | Path | None = None,
    recording: str | None = None,
) -> str:
    """The :func:`doctor` report rendered for a terminal (the canonical format)."""
    text = _read(fixture, recording)
    if text is None:
        return _core.doctor_render_from_env()
    return _core.doctor_render_replay(text)


def fingerprint(
    *,
    fixture: str | Path | None = None,
    recording: str | None = None,
) -> dict[str, Any]:
    """The machine fingerprint (GPU, driver, toolchain, ...).

    Sources the same way as :func:`doctor`. Raises ``ValueError`` when there is
    no device.
    """
    text = _read(fixture, recording)
    raw = _core.fingerprint_from_env() if text is None else _core.fingerprint_replay(text)
    machine: dict[str, Any] = json.loads(raw)
    return machine
