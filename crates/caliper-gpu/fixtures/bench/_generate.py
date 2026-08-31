#!/usr/bin/env python3
"""Regenerate the bench replay fixtures. Deterministic; stdlib only.

Run from anywhere:  python3 crates/caliper-gpu/fixtures/bench/_generate.py

Each fixture is a full `bench()` session in call order:
    device_info.snapshot -> gpu_clock.lock -> gpu_clock.throttle_reasons (before)
    -> kernel_launcher.time_batches -> gpu_clock.throttle_reasons (after)
    -> gpu_clock.read -> gpu_clock.unlock*
(*unlock only when the lock succeeded).
"""

from __future__ import annotations

import json
import math
import pathlib

HERE = pathlib.Path(__file__).parent
HEADER = "# caliper-fixture v=0.0.1 arch=sm_89 (synthetic, see _generate.py)"

MACHINE = {
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "sm_arch": "sm_89",
    "vram_mib": 24564,
    "sm_count": 128,
    "l2_bytes": 75497472,
    "bar1_mib": 32768,
    "driver": "550.90.07",
    "cuda_runtime": "12.4",
    "cuda_driver": "12.4",
    "nvml_version": "12.550.90",
    "ecc": False,
    "mig": "disabled",
    "persistence_mode": True,
    "pcie_gen": 4,
    "pcie_width": 16,
    "toolkit": {"triton": "3.2.0", "torch": "2.6.0", "ptxas": "12.4.131", "nvcc": "12.4.131"},
}


def ok(port: str, method: str, args: object, ret: object) -> str:
    return json.dumps(
        {"port": port, "method": method, "args": args, "ret": {"Ok": ret}},
        separators=(",", ":"),
    )


def err(port: str, method: str, args: object, ret: object) -> str:
    return json.dumps(
        {"port": port, "method": method, "args": args, "ret": {"Err": ret}},
        separators=(",", ":"),
    )


def write(name: str, lines: list[str]) -> None:
    (HERE / name).write_text(HEADER + "\n" + "\n".join(lines) + "\n")


def snapshot() -> str:
    return ok("device_info", "snapshot", None, MACHINE)


def time_batches(
    n: int,
    gpu: list[float],
    wall: list[float],
    throttled: list[bool],
    reasons: list[str],
    kernel_key: str = "kernel",
) -> str:
    return ok(
        "kernel_launcher",
        "time_batches",
        {"kernel_key": kernel_key, "batch": 32, "batches": n, "use_graph": False},
        {
            "gpu_us": gpu,
            "wall_us": wall,
            "batch": 32,
            "throttled": throttled,
            "throttle_reasons": reasons,
        },
    )


def read(sm: int, locked: bool) -> str:
    return ok(
        "gpu_clock",
        "read",
        None,
        {
            "sm_mhz": sm,
            "mem_mhz": 10501,
            "locked": locked,
            "lock_method": "nvml" if locked else None,
        },
    )


LOCK_ARGS = {"sm_mhz": None, "mem_mhz": None}


def poll(reasons: list[str]) -> str:
    return ok("gpu_clock", "throttle_reasons", None, reasons)


def probe(kernel_key: str = "kernel", regs: int = 168, smem: int = 99328) -> str:
    kernel = {
        "name": kernel_key,
        "target": "sm_89",
        "ptxas": {
            "regs_per_thread": regs,
            "smem_static_bytes": smem,
            "spill_stores_bytes": 0,
            "spill_loads_bytes": 0,
            "stack_bytes": 0,
        },
    }
    return ok("module_probe", "probe", {"kernel_key": kernel_key}, [kernel])


def probe_unavailable(kernel_key: str = "kernel") -> str:
    return err(
        "module_probe", "probe", {"kernel_key": kernel_key}, {"Unsupported": "no ptxas on PATH"}
    )


def probe_hard_error(kernel_key: str = "kernel") -> str:
    return err(
        "module_probe", "probe", {"kernel_key": kernel_key}, {"Cuda": "CUDA_ERROR_INVALID_IMAGE"}
    )


def _k(name: str, regs: int, smem: int = 0) -> dict:
    return {
        "name": name,
        "target": "sm_89",
        "ptxas": {
            "regs_per_thread": regs,
            "smem_static_bytes": smem,
            "spill_stores_bytes": 0,
            "spill_loads_bytes": 0,
            "stack_bytes": 0,
        },
    }


def probe_multi(kernel_key: str) -> str:
    # Two kernels; the named one carries a distinctive register count so the
    # pick_ptxas name match (not just "first") is observable.
    kernels = [_k("epilogue_helper", 32), _k(kernel_key, 200, 40960)]
    return ok("module_probe", "probe", {"kernel_key": kernel_key}, kernels)


def happy() -> None:
    n = 40
    gpu = [round(6400.0 + 3.0 * math.sin(i / 2.0), 3) for i in range(n)]
    wall = [round(g + 320.0 + 2.0 * math.cos(i / 3.0), 3) for i, g in enumerate(gpu)]
    write(
        "happy.jsonl",
        [
            snapshot(),
            ok("gpu_clock", "lock", LOCK_ARGS, "Locked"),
            poll([]),
            time_batches(n, gpu, wall, [], []),
            ok("gpu_clock", "throttle_reasons", None, []),
            probe("kernel"),
            read(2520, True),
            ok("gpu_clock", "unlock", None, None),
        ],
    )


def unlocked_throttled() -> None:
    n = 40
    gpu = [6400.0] * n
    gpu[10] = gpu[11] = 20000.0
    wall = [round(g + 320.0, 3) for g in gpu]
    throttled = [i in (10, 11) for i in range(n)]
    write(
        "unlocked_throttled.jsonl",
        [
            snapshot(),
            ok("gpu_clock", "lock", LOCK_ARGS, "Denied"),
            poll(["SW_POWER_CAP"]),
            time_batches(n, gpu, wall, throttled, ["SW_POWER_CAP"]),
            ok("gpu_clock", "throttle_reasons", None, ["SW_POWER_CAP"]),
            probe("kernel"),
            read(2415, False),
        ],
    )


def cold_ramp() -> None:
    ramp = [round(6400.0 + 5000.0 * math.exp(-i / 7.0), 3) for i in range(30)]
    flat = [6400.0] * 40
    gpu = ramp + flat
    wall = [round(g + 320.0, 3) for g in gpu]
    write(
        "cold_ramp.jsonl",
        [
            snapshot(),
            ok("gpu_clock", "lock", LOCK_ARGS, "Locked"),
            poll([]),
            time_batches(len(gpu), gpu, wall, [], []),
            ok("gpu_clock", "throttle_reasons", None, []),
            probe("kernel"),
            read(2520, True),
            ok("gpu_clock", "unlock", None, None),
        ],
    )


def lock_error() -> None:
    # A hard NVML permission error from lock() must degrade to an unlocked,
    # tagged run -- not raise. No unlock call (the lock never succeeded).
    n = 40
    gpu = [6400.0] * n
    wall = [round(g + 320.0, 3) for g in gpu]
    write(
        "lock_error.jsonl",
        [
            snapshot(),
            err("gpu_clock", "lock", LOCK_ARGS, {"PermissionDenied": "NVML_ERROR_NO_PERMISSION"}),
            poll([]),
            time_batches(n, gpu, wall, [], []),
            ok("gpu_clock", "throttle_reasons", None, []),
            probe("kernel"),
            read(2415, False),
        ],
    )


def trailing_call() -> None:
    # A well-formed happy session with one extra recorded call at the end;
    # run_replay must reject it as "recording fully consumed".
    n = 40
    gpu = [6400.0] * n
    wall = [round(g + 320.0, 3) for g in gpu]
    write(
        "trailing_call.jsonl",
        [
            snapshot(),
            ok("gpu_clock", "lock", LOCK_ARGS, "Locked"),
            poll([]),
            time_batches(n, gpu, wall, [], []),
            ok("gpu_clock", "throttle_reasons", None, []),
            probe("kernel"),
            read(2520, True),
            ok("gpu_clock", "unlock", None, None),
            ok("gpu_clock", "unlock", None, None),  # <- leftover
        ],
    )


def oracle_o1() -> None:
    # A clean locked run of the O1 `busy` oracle at ~200 us/launch, keyed so
    # `caliper bench corpus:o1 --recording bench/oracle_o1.jsonl` matches.
    n = 40
    gpu = [round(6400.0 + 3.0 * math.sin(i / 2.0), 3) for i in range(n)]
    wall = [round(g + 320.0, 3) for g in gpu]
    write(
        "oracle_o1.jsonl",
        [
            snapshot(),
            ok("gpu_clock", "lock", LOCK_ARGS, "Locked"),
            poll([]),
            time_batches(n, gpu, wall, [], [], kernel_key="oracle:busy"),
            ok("gpu_clock", "throttle_reasons", None, []),
            probe("oracle:busy"),
            read(2520, True),
            ok("gpu_clock", "unlock", None, None),
        ],
    )


def ptxas_unavailable() -> None:
    n = 40
    gpu = [6400.0] * n
    wall = [round(g + 320.0, 3) for g in gpu]
    write(
        "ptxas_unavailable.jsonl",
        [
            snapshot(),
            ok("gpu_clock", "lock", LOCK_ARGS, "Locked"),
            poll([]),
            time_batches(n, gpu, wall, [], []),
            ok("gpu_clock", "throttle_reasons", None, []),
            probe_unavailable("kernel"),
            read(2520, True),
            ok("gpu_clock", "unlock", None, None),
        ],
    )


def multi_kernel_probe() -> None:
    n = 40
    gpu = [6400.0] * n
    wall = [round(g + 320.0, 3) for g in gpu]
    write(
        "multi_kernel_probe.jsonl",
        [
            snapshot(),
            ok("gpu_clock", "lock", LOCK_ARGS, "Locked"),
            poll([]),
            time_batches(n, gpu, wall, [], []),
            ok("gpu_clock", "throttle_reasons", None, []),
            probe_multi("kernel"),
            read(2520, True),
            ok("gpu_clock", "unlock", None, None),
        ],
    )


def probe_hard_error_fixture() -> None:
    n = 40
    gpu = [6400.0] * n
    wall = [round(g + 320.0, 3) for g in gpu]
    write(
        "probe_hard_error.jsonl",
        [
            snapshot(),
            ok("gpu_clock", "lock", LOCK_ARGS, "Locked"),
            poll([]),
            time_batches(n, gpu, wall, [], []),
            ok("gpu_clock", "throttle_reasons", None, []),
            probe_hard_error("kernel"),
        ],
    )


if __name__ == "__main__":
    for f in (
        happy,
        unlocked_throttled,
        cold_ramp,
        lock_error,
        trailing_call,
        oracle_o1,
        ptxas_unavailable,
        multi_kernel_probe,
        probe_hard_error_fixture,
    ):
        f()
    print("wrote 9 bench fixtures")
