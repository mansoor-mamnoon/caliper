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
    shape: dict[str, int] | None = None,
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

    roofline_spec: dict[str, Any] | None = None
    if isinstance(target, str) and target.startswith("corpus:"):
        resolved = _core.resolve_corpus_target(target)
        if resolved is None:
            names = ", ".join(t[0] for t in _core.corpus_targets())
            raise ValueError(f"unknown corpus target {target!r}; available: {names}")
        kernel_key = resolved
        spec_json = _core.corpus_roofline_spec(kernel_key, json.dumps(shape or {}), dtype)
        if spec_json is not None:
            roofline_spec = json.loads(spec_json)

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
        "roofline": roofline_spec,
    }
    return Result.from_json(_core.bench_replay(recording, json.dumps(opts)))


def _read(fixture: str | Path | None, recording: str | None) -> str | None:
    if recording is not None:
        return recording
    if fixture is not None:
        return Path(fixture).read_text()
    return None


_RETURN_MODES = ("min", "max", "mean", "median")


def _reduce_times_ms(
    times_ms: list[float], quantiles: list[float] | None, return_mode: str
) -> float | list[float]:
    """Reduce a raw millisecond sample vector the way Triton's ``do_bench``
    does, using ``caliper._core`` for the percentile math."""
    if quantiles is not None:
        return list(_core.quantiles(times_ms, [float(q) for q in quantiles]))
    summary = _core.summarize(times_ms)
    return {
        "mean": summary["mean"],
        "median": summary["p50"],
        "min": summary["min"],
        "max": summary["max"],
    }[return_mode]


def _do_bench_live(
    fn: Any,
    warmup_ms: float,
    rep_ms: float,
    grad_to_none: Any,
    quantiles: list[float] | None,
    return_mode: str,
) -> float | list[float]:
    """Time a live callable with CUDA events, à la ``triton.testing.do_bench``.
    Needs PyTorch + a CUDA device; raises ``NotImplementedError`` otherwise."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch not installed on the dev box
        raise NotImplementedError(
            "caliper.do_bench() on a live callable needs PyTorch with CUDA. Pass "
            "fixture=<path> or recording=<text> for the no-GPU path."
        ) from exc
    if not torch.cuda.is_available():  # pragma: no cover - no CUDA on the dev box
        raise NotImplementedError(
            "caliper.do_bench() on a live callable needs a CUDA device. Pass "
            "fixture=<path> or recording=<text> for the no-GPU path."
        )

    # pragma: no cover below - exercised on a CUDA host (playbook #10), not in CI
    fn()
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end.record()
    torch.cuda.synchronize()
    estimate_ms = max(start.elapsed_time(end) / 5.0, 1e-6)

    n_warmup = max(1, int(warmup_ms / estimate_ms))
    n_repeat = max(1, int(rep_ms / estimate_ms))
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    for _ in range(n_warmup):
        fn()
    for i in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        cache.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)]
    return _reduce_times_ms(times_ms, quantiles, return_mode)


def do_bench(
    fn: Any = None,
    warmup: float = 25,
    rep: float = 100,
    grad_to_none: Any = None,
    quantiles: list[float] | None = None,
    fast_flush: bool = True,
    return_mode: str = "mean",
    *,
    fixture: str | Path | None = None,
    recording: str | None = None,
    kernel_key: str = "kernel",
    dtype: str | None = None,
    batch: int = 32,
    batches: int = 50,
) -> float | list[float]:
    """Triton-compatible ``do_bench``: time ``fn`` and return the result in
    **milliseconds**.

    The signature matches ``triton.testing.do_bench`` so an unmodified Triton
    script can swap the import. ``return_mode`` is one of ``"min"`` / ``"max"``
    / ``"mean"`` / ``"median"``; ``quantiles`` (e.g. ``[0.5, 0.2, 0.8]``)
    overrides it and returns a list.

    Two modes:

    * a live ``fn`` (no ``fixture`` / ``recording``) is timed with CUDA events,
      like Triton's own ``do_bench`` -- needs PyTorch + a CUDA device.
    * ``fixture=`` / ``recording=`` replays a recorded device session, so the
      shim works with no GPU.

    ``fast_flush`` is accepted for compatibility (the live path always clears an
    L2-sized buffer between reps). For the rigorous, clock-locked measurement
    use :func:`bench` instead.
    """
    if return_mode not in _RETURN_MODES:
        raise ValueError(f"return_mode must be one of {_RETURN_MODES}, got {return_mode!r}")
    del fast_flush  # accepted for Triton parity; the live path always flushes

    text = _read(fixture, recording)
    if text is None:
        return _do_bench_live(fn, warmup, rep, grad_to_none, quantiles, return_mode)

    del fn, warmup, rep, grad_to_none  # a recording fixes the samples; see bench()
    opts = json.dumps(
        {"kernel_key": kernel_key, "dtype": dtype, "batch": batch, "batches": batches}
    )

    if quantiles is not None:
        us = _core.bench_replay_quantiles(text, opts, [float(q) for q in quantiles])
        return [v / 1000.0 for v in us]

    r = Result.from_json(_core.bench_replay(text, opts))
    picked = {
        "mean": r.mean_us,
        "median": r.p50_us,
        "min": r.min_us,
        "max": r.max_us,
    }[return_mode]
    if picked is None:  # pragma: no cover - reduce always fills these
        raise ValueError(f"the record has no {return_mode} timing")
    return picked / 1000.0


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


def _fingerprint_json(fixture: str | Path | None, recording: str | None) -> str:
    text = _read(fixture, recording)
    return _core.fingerprint_from_env() if text is None else _core.fingerprint_replay(text)


def fingerprint(
    *,
    fixture: str | Path | None = None,
    recording: str | None = None,
) -> dict[str, Any]:
    """The machine fingerprint (GPU, driver, toolchain, ...).

    Sources the same way as :func:`doctor`. Raises ``ValueError`` when there is
    no device.
    """
    machine: dict[str, Any] = json.loads(_fingerprint_json(fixture, recording))
    return machine


def fingerprint_check(
    *,
    fixture: str | Path | None = None,
    recording: str | None = None,
) -> dict[str, Any]:
    """Completeness of the machine fingerprint.

    Returns ``{"complete": bool, "missing_required": [...],
    "missing_recommended": [...]}``. A fingerprint has to be complete for a
    result to be comparable across machines.
    """
    report: dict[str, Any] = json.loads(
        _core.fingerprint_check(_fingerprint_json(fixture, recording))
    )
    return report


SELFTEST_EXIT_CODE = {"PASS": 0, "FAIL": 1, "ERROR": 2}


def selftest(*, full: bool = False) -> dict[str, Any]:
    """Run the oracle self-test against the ``CALIPER_GPU_PORTS`` backend.

    Returns the Appendix-E report as a dict (no extra keys -- map ``result``
    through :data:`SELFTEST_EXIT_CODE` for a process exit code). With no CUDA
    device this is the ``ERROR`` / no-device report. ``full`` also runs O5
    (cuBLAS) and the ``nsys`` cross-check; the on-device oracle runner itself
    lands on a CUDA host, so until it does a device-present run is still an
    ``ERROR`` (every oracle skipped).
    """
    report: dict[str, Any] = json.loads(_core.selftest_from_env(full))
    return report


def toolchain() -> dict[str, str | None]:
    """Detected local kernel toolchain (Triton, PyTorch, ``nvcc``, ``ptxas``).

    Each value is a version string, or ``None`` when that tool or package is not
    installed here.
    """
    from caliper import _toolchain

    return _toolchain.detect()
