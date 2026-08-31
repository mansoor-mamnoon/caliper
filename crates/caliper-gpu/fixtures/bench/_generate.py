#!/usr/bin/env python3
"""Regenerate the bench replay fixtures. Deterministic; stdlib only.

Run from anywhere:  python3 crates/caliper-gpu/fixtures/bench/_generate.py
"""

from __future__ import annotations

import json
import math
import pathlib

HERE = pathlib.Path(__file__).parent

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


def line(port: str, method: str, args: object, ret_ok: object) -> str:
    return json.dumps(
        {"port": port, "method": method, "args": args, "ret": {"Ok": ret_ok}},
        separators=(",", ":"),
    )


def write(name: str, lines: list[str]) -> None:
    (HERE / name).write_text("\n".join(lines) + "\n")


def happy() -> None:
    # 40 batches of 32 launches, ~200 us/launch => ~6400 us/batch, +320 us wall.
    n = 40
    gpu = [round(6400.0 + 3.0 * math.sin(i / 2.0), 3) for i in range(n)]
    wall = [round(g + 320.0 + 2.0 * math.cos(i / 3.0), 3) for i, g in enumerate(gpu)]
    write(
        "happy.jsonl",
        [
            line("device_info", "snapshot", None, MACHINE),
            line("gpu_clock", "lock", {"sm_mhz": None, "mem_mhz": None}, "Locked"),
            line(
                "kernel_launcher",
                "time_batches",
                {"kernel_key": "kernel", "batch": 32, "batches": n, "use_graph": False},
                {
                    "gpu_us": gpu,
                    "wall_us": wall,
                    "batch": 32,
                    "throttled": [],
                    "throttle_reasons": [],
                },
            ),
            line(
                "gpu_clock",
                "read",
                None,
                {"sm_mhz": 2520, "mem_mhz": 10501, "locked": True, "lock_method": "nvml"},
            ),
            line("gpu_clock", "unlock", None, None),
        ],
    )


def unlocked_throttled() -> None:
    # Lock is denied; two batches are throttled and should be dropped.
    n = 40
    gpu = [6400.0] * n
    gpu[10] = gpu[11] = 20000.0  # throttled batches run long
    wall = [round(g + 320.0, 3) for g in gpu]
    throttled = [i in (10, 11) for i in range(n)]
    write(
        "unlocked_throttled.jsonl",
        [
            line("device_info", "snapshot", None, MACHINE),
            line("gpu_clock", "lock", {"sm_mhz": None, "mem_mhz": None}, "Denied"),
            line(
                "kernel_launcher",
                "time_batches",
                {"kernel_key": "kernel", "batch": 32, "batches": n, "use_graph": False},
                {
                    "gpu_us": gpu,
                    "wall_us": wall,
                    "batch": 32,
                    "throttled": throttled,
                    "throttle_reasons": ["SW_POWER_CAP"],
                },
            ),
            line(
                "gpu_clock",
                "read",
                None,
                {"sm_mhz": 2415, "mem_mhz": 10501, "locked": False, "lock_method": None},
            ),
        ],
    )


def cold_ramp() -> None:
    # 30 hot batches decaying, then 40 flat; warm-up trimming should kick in.
    ramp = [round(6400.0 + 5000.0 * math.exp(-i / 7.0), 3) for i in range(30)]
    flat = [6400.0] * 40
    gpu = ramp + flat
    wall = [round(g + 320.0, 3) for g in gpu]
    write(
        "cold_ramp.jsonl",
        [
            line("device_info", "snapshot", None, MACHINE),
            line("gpu_clock", "lock", {"sm_mhz": None, "mem_mhz": None}, "Locked"),
            line(
                "kernel_launcher",
                "time_batches",
                {"kernel_key": "kernel", "batch": 32, "batches": len(gpu), "use_graph": False},
                {
                    "gpu_us": gpu,
                    "wall_us": wall,
                    "batch": 32,
                    "throttled": [],
                    "throttle_reasons": [],
                },
            ),
            line(
                "gpu_clock",
                "read",
                None,
                {"sm_mhz": 2520, "mem_mhz": 10501, "locked": True, "lock_method": "nvml"},
            ),
            line("gpu_clock", "unlock", None, None),
        ],
    )


if __name__ == "__main__":
    happy()
    unlocked_throttled()
    cold_ramp()
    print("wrote happy.jsonl, unlocked_throttled.jsonl, cold_ramp.jsonl")
