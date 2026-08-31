"""caliper -- correct-by-default GPU kernel benchmarking.

This package is in early development. The public API is defined in the project
plan and is being implemented incrementally; import paths under
``caliper._internal`` are private and may change without notice.
"""

from __future__ import annotations

from caliper.__about__ import __version__
from caliper._internal.schema import KernelLabel, Result

__all__ = ["KernelLabel", "Result", "__version__"]
