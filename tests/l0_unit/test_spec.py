"""L0: the sweep-spec parser and expander (Appendix D), through the bindings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from caliper import _spec

pytestmark = pytest.mark.l0

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "crates" / "caliper-core" / "tests" / "spec"


def test_appendix_d_yaml_matches_the_golden_cell_list() -> None:
    cells = _spec.load_cells(GOLDEN_DIR / "appendix_d.yaml")
    golden = json.loads((GOLDEN_DIR / "appendix_d.cells.json").read_text())
    assert cells == golden
    assert len(cells) == 3 * 2 * 6  # dtypes x layouts x llm-7b shapes


def test_yaml_text_and_dict_are_accepted_too() -> None:
    yaml_text = "target: corpus:gemm\ndtypes: [bf16]\nshapes: square-pow2\n"
    from_text = _spec.load_cells(yaml_text)
    from_dict = _spec.load_cells(
        {"target": "corpus:gemm", "dtypes": ["bf16"], "shapes": "square-pow2"}
    )
    assert from_text == from_dict
    assert len(from_text) == 5  # one dtype, default layout, 5 square-pow2 shapes


def test_inline_attention_shapes_expand() -> None:
    cells = _spec.load_cells(
        {
            "target": "attn.py::flash",
            "dtypes": ["bf16"],
            "shapes": [{"kind": "attn", "b": 1, "h": 32, "s": 2048, "d": 128}],
        }
    )
    assert len(cells) == 1
    assert cells[0]["shape"] == {"kind": "attn", "b": 1, "h": 32, "s": 2048, "d": 128}


def test_inline_shapes_are_deduped() -> None:
    cells = _spec.load_cells(
        {
            "target": "k",
            "dtypes": ["bf16"],
            "shapes": [
                {"kind": "gemm", "m": 64, "n": 64, "k": 64},
                {"kind": "gemm", "m": 64, "n": 64, "k": 64},
                {"kind": "gemm", "m": 128, "n": 128, "k": 128},
            ],
        }
    )
    assert len(cells) == 2


@pytest.mark.parametrize(
    ("spec", "needle"),
    [
        ({"dtypes": ["bf16"], "shapes": "square-pow2"}, "parse"),  # missing target
        ({"target": "k", "dtypes": ["bf16"], "shapes": "square-pow2", "wat": 1}, "parse"),
        ({"target": "k", "dtypes": ["bf17"], "shapes": "square-pow2"}, "dtype"),
        (
            {"target": "k", "dtypes": ["bf16"], "layouts": ["diag"], "shapes": "square-pow2"},
            "layout",
        ),
        ({"target": "k", "dtypes": ["bf16"], "shapes": "llm-9000b"}, "shape library"),
        ({"target": "k", "dtypes": [], "shapes": "square-pow2"}, "dtypes"),
        ({"target": "k", "dtypes": ["bf16"], "layouts": [], "shapes": "square-pow2"}, "layouts"),
        ({"target": "k", "dtypes": ["bf16"], "shapes": []}, "zero cells"),
        (
            {
                "target": "k",
                "dtypes": ["bf16"],
                "shapes": "square-pow2",
                "bench": {"cuda_graph": "x"},
            },
            "cuda_graph",
        ),
        (
            {
                "target": "k",
                "dtypes": ["bf16"],
                "shapes": "square-pow2",
                "bench": {"min_samples": 0},
            },
            "min_samples",
        ),
    ],
)
def test_a_bad_spec_is_a_typed_value_error(spec: dict[str, Any], needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        _spec.load_cells(spec)


def test_resume_drops_finished_cells() -> None:
    cells = _spec.load_cells({"target": "k", "dtypes": ["bf16", "fp16"], "shapes": "square-pow2"})
    keys = _spec.cell_keys(cells)
    left = _spec.pending_cells(cells, keys[:3])
    assert len(left) == len(cells) - 3
    left_again = _spec.pending_cells(cells, keys[:3] + keys[:3])  # idempotent
    assert len(left_again) == len(cells) - 3
