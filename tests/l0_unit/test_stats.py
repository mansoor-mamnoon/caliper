"""L0: the Rust statistics and warm-up code, exercised through the bindings.

Exhaustive coverage lives in ``cargo test``; these check the binding surface and
a couple of anchor values.
"""

from __future__ import annotations

import math

import pytest

from caliper import _core

pytestmark = pytest.mark.l0


def test_summarize_matches_hand_computed_values() -> None:
    s = _core.summarize([float(i) for i in range(1, 11)])
    assert s["n"] == 10
    assert s["min"] == 1.0
    assert s["max"] == 10.0
    assert math.isclose(s["p10"], 1.9)
    assert math.isclose(s["p50"], 5.5)
    assert math.isclose(s["p90"], 9.1)
    assert math.isclose(s["mean"], 5.5)
    assert math.isclose(s["mad"], 2.5)
    assert math.isclose(s["cov"], math.sqrt(82.5 / 9.0) / 5.5)


def test_summarize_percentiles_are_ordered() -> None:
    s = _core.summarize([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0])
    assert s["min"] <= s["p10"] <= s["p50"] <= s["p90"] <= s["max"]


def test_summarize_constant_has_no_spread_and_omits_nothing() -> None:
    s = _core.summarize([7.0] * 16)
    assert s["p50"] == 7.0
    assert s["mad"] == 0.0
    assert s["cov"] == 0.0


@pytest.mark.parametrize("bad", [[], [1.0, float("nan")], [float("inf"), 2.0]])
def test_summarize_rejects_empty_or_non_finite(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        _core.summarize(bad)


def test_cross_pass_cov_needs_two_finite_passes() -> None:
    assert _core.cross_pass_cov([]) is None
    assert _core.cross_pass_cov([5.0]) is None
    assert _core.cross_pass_cov([5.0, float("nan")]) is None
    assert _core.cross_pass_cov([10.0, 10.0, 10.0]) == 0.0


def test_steady_state_index_on_flat_series() -> None:
    start, converged = _core.steady_state_index([50.0] * 100)
    assert (start, converged) == (0, True)


def test_steady_state_index_beats_a_fixed_warmup_of_25() -> None:
    # A cold ramp that is still well above steady at sample 25.
    steady, hot, tau = 50.0, 120.0, 40.0
    times = [steady + (hot - steady) * math.exp(-i / tau) for i in range(200)]
    times += [steady] * 40
    assert times[25] > steady * 1.10

    start, converged = _core.steady_state_index(times, window=20, tol=0.02, min_warm=30)
    assert converged
    assert start > 25
