"""L0: the named shape libraries, through the bindings."""

from __future__ import annotations

import json
from typing import Any

import pytest

from caliper import _core

pytestmark = pytest.mark.l0

NAMES = ["square-pow2", "prime-odd", "llm-7b", "llm-70b"]


def _resolve(name: str) -> list[dict[str, Any]]:
    raw = _core.resolve_shape_library(name)
    assert raw is not None
    result: list[dict[str, Any]] = json.loads(raw)
    return result


def test_names_are_stable() -> None:
    assert sorted(_core.shape_library_names()) == sorted(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_every_library_resolves_to_gemm_shapes(name: str) -> None:
    shapes = _resolve(name)
    assert shapes
    for s in shapes:
        assert s["kind"] == "gemm"
        assert {"m", "n", "k"} <= set(s)
        assert all(isinstance(s[d], int) and s[d] > 0 for d in "mnk")


def test_unknown_library_is_none() -> None:
    assert _core.resolve_shape_library("no-such-lib") is None


def test_square_pow2_is_square_and_power_of_two() -> None:
    shapes = _resolve("square-pow2")
    assert len(shapes) == 5
    for s in shapes:
        assert s["m"] == s["n"] == s["k"]
        assert s["m"] & (s["m"] - 1) == 0  # power of two


def test_llm_7b_matches_the_llama_config() -> None:
    shapes = _resolve("llm-7b")
    assert len(shapes) == 6  # 3 gemms x 2 seq lengths
    assert {"m": 2048, "n": 11008, "k": 4096, "kind": "gemm"} in shapes
