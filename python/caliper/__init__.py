"""caliper -- correct-by-default GPU kernel benchmarking.

The measurement logic lives in a Rust core (``caliper._core``); this package is
the Python-facing surface: the public API, the command-line tool, and the
Triton-compatible ``do_bench`` shim. It is in early development and the API will
change until the first tagged release.
"""

from __future__ import annotations

from caliper import _core
from caliper._record import Result
from caliper.api import bench, do_bench

__version__: str = _core.__version__


def schema_version() -> str:
    """Return the result-schema version this build understands."""
    return _core.schema_version()


__all__ = ["Result", "__version__", "bench", "do_bench", "schema_version"]
