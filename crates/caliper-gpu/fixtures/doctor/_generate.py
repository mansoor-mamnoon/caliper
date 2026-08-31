#!/usr/bin/env python3
"""Regenerate the doctor replay fixtures. Deterministic; stdlib only.

Run from anywhere:  python3 crates/caliper-gpu/fixtures/doctor/_generate.py

Each fixture is a `doctor` probe in call order:
    device_info.snapshot -> gpu_clock.lock -> gpu_clock.unlock (iff Locked)
    -> gpu_clock.throttle_reasons
"""

from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).parent
HEADER = "# caliper-fixture v=0.0.1 arch=sm_89 (synthetic, see _generate.py)"

FULL_MACHINE = {
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

LOCK_ARGS = {"sm_mhz": None, "mem_mhz": None}


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


def machine(**overrides: object) -> dict[str, object]:
    m = dict(FULL_MACHINE)
    m.update(overrides)
    return m


def fit() -> None:
    write(
        "fit.jsonl",
        [
            ok("device_info", "snapshot", None, machine()),
            ok("gpu_clock", "lock", LOCK_ARGS, "Locked"),
            ok("gpu_clock", "unlock", None, None),
            ok("gpu_clock", "throttle_reasons", None, []),
        ],
    )


def throttling() -> None:
    write(
        "throttling.jsonl",
        [
            ok("device_info", "snapshot", None, machine()),
            ok("gpu_clock", "lock", LOCK_ARGS, "Locked"),
            ok("gpu_clock", "unlock", None, None),
            ok("gpu_clock", "throttle_reasons", None, ["SW_THERMAL_SLOWDOWN"]),
        ],
    )


def constrained() -> None:
    write(
        "constrained.jsonl",
        [
            ok("device_info", "snapshot", None, machine(ecc=True, persistence_mode=False)),
            ok("gpu_clock", "lock", LOCK_ARGS, "Denied"),
            ok("gpu_clock", "throttle_reasons", None, []),
        ],
    )


def no_device() -> None:
    write(
        "no_device.jsonl",
        [err("device_info", "snapshot", None, "NoDevice")],
    )


if __name__ == "__main__":
    for f in (fit, throttling, constrained, no_device):
        f()
    print("wrote fit, throttling, constrained, no_device")
