"""The ``Result`` record: caliper's single output schema.

Every command that measures something emits a :class:`Result` (or a table of
them). The schema is versioned via :data:`SCHEMA_VERSION`; :func:`Result.to_dict`
and :func:`Result.from_dict` are the reference (de)serialisation and must
round-trip exactly.

At this stage every field is optional and defaults to ``None`` (or an empty
container). Modules fill in the sections they own as the measurement pipeline is
built. Freezing the record for the stable public API is deferred until the
builder paths settle.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from caliper.__about__ import __version__

SCHEMA_VERSION = "1"

JsonDict = dict[str, Any]


@dataclass
class KernelLabel:
    """Identity of the thing being measured."""

    name: str | None = None
    impl: str | None = None  # "triton" | "cuda" | "cublas" | "torch" | ...
    source_hash: str | None = None
    autotune_config: JsonDict = field(default_factory=dict)
    dtype: str | None = None
    shape: JsonDict = field(default_factory=dict)
    layout: str | None = None


@dataclass
class Timing:
    """Wall-clock and GPU-event timing, as a distribution."""

    p10_us: float | None = None
    p50_us: float | None = None
    p90_us: float | None = None
    mad_us: float | None = None
    wall_p50_us: float | None = None
    launch_overhead_us: float | None = None
    n_samples: int | None = None
    n_warmup_to_steady: int | None = None
    invalidated_samples: int | None = None
    cross_pass_cov: float | None = None


@dataclass
class Roofline:
    """Achieved throughput relative to the hardware roofline."""

    achieved_tflops: float | None = None
    roofline_pct: float | None = None
    achieved_gbps: float | None = None
    arithmetic_intensity: float | None = None
    ridge_point: float | None = None
    bound: str | None = None  # "compute" | "memory" | "latency" | "unknown"
    baseline_pct: float | None = None
    baseline: str | None = None  # "cublas" | "cudnn" | "torch" | ...


@dataclass
class Ptxas:
    """Static resource usage reported by the compiler."""

    regs_per_thread: int | None = None
    smem_static_bytes: int | None = None
    smem_dynamic_bytes: int | None = None
    spill_loads_bytes: int | None = None
    spill_stores_bytes: int | None = None
    local_bytes: int | None = None
    stack_bytes: int | None = None


@dataclass
class Occupancy:
    """Theoretical and achieved occupancy."""

    theoretical: float | None = None
    achieved: float | None = None
    active_warps_per_sm: int | None = None
    waves: float | None = None


@dataclass
class Clocks:
    """GPU clock state during the measurement."""

    sm_mhz: int | None = None
    mem_mhz: int | None = None
    locked: bool | None = None
    lock_method: str | None = None  # "nvml" | None


@dataclass
class Toolkit:
    """Versions of the compilation / kernel toolchain."""

    triton: str | None = None
    torch: str | None = None
    ptxas: str | None = None
    nvcc: str | None = None


@dataclass
class Machine:
    """Everything about the host and device needed to interpret a row."""

    gpu_name: str | None = None
    sm_arch: str | None = None  # "sm_89"
    vram_mib: int | None = None
    sm_count: int | None = None
    l2_bytes: int | None = None
    bar1_mib: int | None = None
    driver: str | None = None
    cuda_runtime: str | None = None
    cuda_driver: str | None = None
    nvml_version: str | None = None
    ecc: bool | None = None
    mig: str | None = None  # "disabled" | "<geometry>"
    persistence_mode: bool | None = None
    pcie_gen: int | None = None
    pcie_width: int | None = None
    toolkit: Toolkit = field(default_factory=Toolkit)


_SECTION_TYPES: dict[str, type] = {
    "kernel": KernelLabel,
    "timing": Timing,
    "roofline": Roofline,
    "ptxas": Ptxas,
    "occupancy": Occupancy,
    "clocks": Clocks,
    "machine": Machine,
}


@dataclass
class Result:
    """One measurement, with all the context needed to interpret it."""

    schema_version: str = SCHEMA_VERSION
    caliper_version: str = __version__
    measured_at: str | None = None  # ISO-8601 UTC
    host_id_class: str | None = None  # salted, non-identifying
    kernel: KernelLabel = field(default_factory=KernelLabel)
    timing: Timing = field(default_factory=Timing)
    roofline: Roofline = field(default_factory=Roofline)
    ptxas: Ptxas = field(default_factory=Ptxas)
    occupancy: Occupancy = field(default_factory=Occupancy)
    clocks: Clocks = field(default_factory=Clocks)
    machine: Machine = field(default_factory=Machine)
    throttle_reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        """Serialise to a plain, JSON-compatible dict with every key present."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> Result:
        """Rebuild a :class:`Result` from :func:`to_dict` output.

        Unknown keys are ignored; missing keys fall back to defaults, so output
        from an older schema still loads.
        """
        kwargs: dict[str, Any] = {}

        for name in ("schema_version", "caliper_version", "measured_at", "host_id_class"):
            if name in data and data[name] is not None:
                kwargs[name] = data[name]

        for name, section_type in _SECTION_TYPES.items():
            section = data.get(name)
            if isinstance(section, dict):
                kwargs[name] = _build_section(section_type, section)

        for name in ("throttle_reasons", "flags"):
            value = data.get(name)
            if isinstance(value, list):
                kwargs[name] = list(value)

        return cls(**kwargs)


def _build_section(section_type: type, data: JsonDict) -> Any:
    """Construct a dataclass section, recursing into nested dataclass fields."""
    field_types = {f.name: f.type for f in dataclasses.fields(section_type)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in field_types:
            continue
        if key == "toolkit" and isinstance(value, dict):
            kwargs[key] = _build_section(Toolkit, value)
        else:
            kwargs[key] = value
    return section_type(**kwargs)
