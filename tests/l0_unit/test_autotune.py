"""L0: the autotune-config cache key and store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from caliper import _core
from caliper._autotune import AutotuneCache

pytestmark = pytest.mark.l0


def _machine(**over: Any) -> dict[str, Any]:
    m = {
        "sm_arch": "sm_89",
        "driver": "550.90.07",
        "cuda_runtime": "12.4",
        "toolkit": {"ptxas": "12.4.131", "triton": "3.2.0", "torch": "2.6.0"},
    }
    m.update(over)
    return m


def test_key_is_config_order_independent() -> None:
    a = _core.autotune_key(
        _json(_machine()), "src:abc", '{"BLOCK_M": 128, "num_warps": 8}'
    )
    b = _core.autotune_key(
        _json(_machine()), "src:abc", '{"num_warps": 8, "BLOCK_M": 128}'
    )
    assert a == b
    assert _core.autotune_config_canonical('{"b":1,"a":2}') == '{"a":2,"b":1}'


def test_key_changes_when_env_or_source_or_config_changes() -> None:
    base = AutotuneCache.key(_machine(), "src:abc", {"BLOCK_M": 128})
    assert base != AutotuneCache.key(_machine(), "src:abc", {"BLOCK_M": 256})
    assert base != AutotuneCache.key(_machine(), "src:def", {"BLOCK_M": 128})
    assert base != AutotuneCache.key(_machine(driver="560.0"), "src:abc", {"BLOCK_M": 128})


def test_cache_hit_miss_and_persistence(tmp_path: Path) -> None:
    m, ksh = _machine(), "src:abc"
    cache = AutotuneCache(tmp_path / "at.json")

    assert cache.get(m, ksh, {"BLOCK_M": 128}) is None  # miss
    cache.put(m, ksh, {"BLOCK_M": 128}, {"p50_us": 200.0})
    assert cache.get(m, ksh, {"BLOCK_M": 128}) == {"p50_us": 200.0}  # hit
    assert (cache.hits, cache.misses) == (1, 1)
    assert len(cache) == 1

    # a fresh cache over the same file sees the entry (persisted)
    reopened = AutotuneCache(tmp_path / "at.json")
    assert reopened.get(m, ksh, {"BLOCK_M": 128}) == {"p50_us": 200.0}


def test_adding_a_config_re_times_only_the_new_one(tmp_path: Path) -> None:
    m, ksh = _machine(), "src:abc"
    cache = AutotuneCache(tmp_path / "at.json")
    configs = [{"BLOCK_M": 64}, {"BLOCK_M": 128}, {"BLOCK_M": 256}]

    # first sweep: everything is a miss and gets timed
    timed = 0
    for c in configs:
        if cache.get(m, ksh, c) is None:
            cache.put(m, ksh, c, {"p50_us": 100.0 + c["BLOCK_M"]})
            timed += 1
    assert timed == 3

    # second sweep with one extra config: only the new one is timed
    cache2 = AutotuneCache(tmp_path / "at.json")
    timed = 0
    for c in [*configs, {"BLOCK_M": 512}]:
        if cache2.get(m, ksh, c) is None:
            cache2.put(m, ksh, c, {"p50_us": 999.0})
            timed += 1
    assert timed == 1
    assert cache2.hits == 3


def test_autotune_key_rejects_bad_json() -> None:
    with pytest.raises(ValueError):
        _core.autotune_key(_json(_machine()), "src:abc", "{not json")
    with pytest.raises(ValueError):
        _core.autotune_key("not a machine", "src:abc", "{}")


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj)
