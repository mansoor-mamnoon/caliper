"""Regenerate the submit-bundle fixtures from ``base.json``.

    python tests/testdata/build_bundles.py

Each bundle is a directory with ``manifest.json`` + ``rows.jsonl`` +
``fingerprint.json`` (jsonl, not parquet, so the fixtures stay diffable). One
clean bundle plus one per FR-16 / playbook-#14 rejection class, and a standalone
``over_peak_row.json`` that a bare ``caliper validate`` must reject.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from caliper import _core

_HERE = Path(__file__).parent


def _rows() -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = json.loads((_HERE / "base.json").read_text())
    return parsed


def _write_bundle(
    name: str, rows: list[dict[str, Any]], calibration: tuple[float, float] | None
) -> None:
    d = _HERE / name
    d.mkdir(exist_ok=True)
    machine = rows[0].get("machine") or {}
    from caliper._grid import _toolchain_hash

    cal_json = (
        json.dumps({"measured_p50_us": calibration[0], "expected_p50_us": calibration[1]})
        if calibration
        else "null"
    )
    manifest = json.loads(
        _core.submit_manifest(
            json.dumps(rows),
            _toolchain_hash(machine),
            "0.3.0",
            "2026-09-04T00:00:00Z",
            cal_json,
        )
    )
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "fingerprint.json").write_text(json.dumps(machine, indent=2) + "\n")
    print(f"wrote {name}/ ({len(rows)} rows)")


def main() -> None:
    base = _rows()

    _write_bundle("bundle_ok", base, calibration=(101.0, 100.0))

    missing = json.loads(json.dumps(base))
    missing[0]["kernel"].pop("name")
    _write_bundle("bundle_missing_field", missing, calibration=(101.0, 100.0))

    repeat = []
    for p50 in (243.0, 300.0, 210.0, 278.0):  # ~15% CoV -> past the 5% unlocked tolerance
        r = json.loads(json.dumps(base[0]))
        scale = p50 / base[0]["timing"]["p50_us"]
        r["timing"] = {k: (v * scale if k.endswith("_us") else v) for k, v in r["timing"].items()}
        r["flags"] = ["clocks-unlocked"]
        repeat.append(r)
    _write_bundle("bundle_nonreproducing", repeat, calibration=None)

    _write_bundle("bundle_slow_calibration", base, calibration=(118.0, 100.0))

    over_peak = json.loads(json.dumps(base[0]))
    over_peak.setdefault("roofline", {})["roofline_pct"] = 1.7  # past the schema's 1.5 clamp
    (_HERE / "over_peak_row.json").write_text(json.dumps([over_peak], indent=2) + "\n")
    print("wrote over_peak_row.json")


if __name__ == "__main__":
    main()
