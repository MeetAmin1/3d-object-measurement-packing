#!/usr/bin/env python3
"""Independent validation of generated Part 1 and Part 2 result files."""

from pathlib import Path
import json
import math
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part1_obb_measurement import load_mesh  # noqa: E402
from part2_gravity_packing import Item, Placement, validate_solution  # noqa: E402


def validate_obb() -> None:
    obb_path = ROOT / "outputs" / "obb_results.json"
    rows = json.loads(obb_path.read_text(encoding="utf-8"))
    assert len(rows) == 3

    for row in rows:
        path = ROOT / "data" / row["filename"]
        mesh = load_mesh(path)
        transform = np.asarray(row["world_to_obb"], dtype=float)
        extents = np.asarray(row["local_extents"], dtype=float)
        transformed = trimesh.transform_points(np.asarray(mesh.vertices), transform)

        minimum = transformed.min(axis=0)
        maximum = transformed.max(axis=0)
        measured_extents = maximum - minimum
        assert np.allclose(measured_extents, extents, rtol=1e-7, atol=1e-7)
        assert math.isclose(math.prod(extents), row["volume"], rel_tol=1e-9)
        assert all(value > 0 for value in row["dimensions_lwh"])


def validate_packing() -> dict:
    packing_path = ROOT / "outputs" / "packing_solution.json"
    item_path = ROOT / "data" / "Item List.json"
    item_rows = json.loads(item_path.read_text(encoding="utf-8"))
    items = [Item(int(row["id"]), tuple(row["dims"]), row["type"]) for row in item_rows]
    payload = json.loads(packing_path.read_text(encoding="utf-8"))
    placements = [Placement(**row) for row in payload["placements"]]
    return validate_solution(items, placements, tuple(payload["master_box"]))


def main() -> None:
    validate_obb()
    metrics = validate_packing()
    assert metrics["valid"]
    assert metrics["overlap_free"]
    assert metrics["fully_supported"]
    assert metrics["height_is_optimal"]
    print("PASS: All full-mesh vertices lie inside their reported OBB")
    print("PASS: OBB extents and volumes are internally consistent")
    print("PASS: Part 2 contains all 20 items")
    print("PASS: All items are in bounds, overlap-free, and fully supported")
    print(f"PASS: Maximum height {metrics['maximum_height']} equals lower bound {metrics['height_lower_bound']}")


if __name__ == "__main__":
    main()
