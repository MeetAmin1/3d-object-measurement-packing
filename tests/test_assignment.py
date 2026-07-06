#!/usr/bin/env python3
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from part2_gravity_packing import (  # noqa: E402
    Item,
    Placement,
    boxes_overlap,
    is_fully_supported,
    load_items,
    solve,
    validate_solution,
)


class PackingTests(unittest.TestCase):
    def test_touching_faces_are_not_overlap(self):
        first = Placement(1, "a", 0, 0, 0, 10, 10, 10)
        second = Placement(2, "b", 10, 0, 0, 10, 10, 10)
        self.assertFalse(boxes_overlap(first, second))

    def test_union_support(self):
        left = Placement(1, "base", 0, 0, 0, 5, 10, 10)
        right = Placement(2, "base", 5, 0, 0, 5, 10, 10)
        top = Placement(3, "top", 0, 0, 10, 10, 10, 5)
        self.assertTrue(is_fully_supported(top, [left, right]))

    def test_supplied_solution(self):
        items = load_items(ROOT / "data" / "Item List.json")
        placements = solve(items)
        metrics = validate_solution(items, placements)
        self.assertEqual(len(placements), 20)
        self.assertTrue(metrics["overlap_free"])
        self.assertTrue(metrics["fully_supported"])
        self.assertEqual(metrics["maximum_height"], 30)
        self.assertTrue(metrics["height_is_optimal"])


if __name__ == "__main__":
    unittest.main()
