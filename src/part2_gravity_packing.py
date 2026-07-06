#!/usr/bin/env python3
"""Part 2: Gravity-aware 3D packing for the supplied Item List.json.

The solver uses an integer grid derived from the dimensions, checks all 90-degree
orientations, rejects overlap, requires full bottom support, and minimizes
occupied height before compactness. It validates and visualizes the solution.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass
from functools import reduce
from pathlib import Path
from typing import Sequence


MASTER_BOX = (100, 100, 100)


@dataclass(frozen=True)
class Item:
    id: int
    dims: tuple[int, int, int]
    type: str

    @property
    def volume(self) -> int:
        return math.prod(self.dims)


@dataclass(frozen=True)
class Placement:
    id: int
    type: str
    x: int
    y: int
    z: int
    dx: int
    dy: int
    dz: int

    @property
    def volume(self) -> int:
        return self.dx * self.dy * self.dz

    @property
    def top(self) -> int:
        return self.z + self.dz


class PackingError(RuntimeError):
    pass


def load_items(path: Path) -> list[Item]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Item file must contain a non-empty JSON list")

    items: list[Item] = []
    ids: set[int] = set()
    for row in raw:
        item_id = int(row["id"])
        dims = tuple(int(v) for v in row["dims"])
        if len(dims) != 3 or any(v <= 0 for v in dims):
            raise ValueError(f"Invalid dimensions for item {item_id}: {dims}")
        if item_id in ids:
            raise ValueError(f"Duplicate item id: {item_id}")
        ids.add(item_id)
        items.append(Item(item_id, dims, str(row.get("type", "item"))))
    return items


def unique_orientations(dims: Sequence[int]) -> list[tuple[int, int, int]]:
    return sorted(
        set(itertools.permutations(tuple(int(v) for v in dims), 3)),
        key=lambda d: (d[2], -(d[0] * d[1]), -max(d[0], d[1]), d),
    )


def boxes_overlap(a: Placement, b: Placement) -> bool:
    return (
        a.x < b.x + b.dx and a.x + a.dx > b.x
        and a.y < b.y + b.dy and a.y + a.dy > b.y
        and a.z < b.z + b.dz and a.z + a.dz > b.z
    )


def rectangle_union_area(rectangles: list[tuple[int, int, int, int]]) -> int:
    """Exact union area of axis-aligned rectangles using x-slab integration."""
    if not rectangles:
        return 0
    xs = sorted({x for rectangle in rectangles for x in (rectangle[0], rectangle[2])})
    area = 0
    for x0, x1 in zip(xs, xs[1:]):
        intervals = sorted(
            (ry0, ry1)
            for rx0, ry0, rx1, ry1 in rectangles
            if rx0 < x1 and rx1 > x0
        )
        if not intervals:
            continue
        start, end = intervals[0]
        covered = 0
        for next_start, next_end in intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                covered += end - start
                start, end = next_start, next_end
        covered += end - start
        area += (x1 - x0) * covered
    return area


def support_area(candidate: Placement, placed: Sequence[Placement]) -> int:
    if candidate.z == 0:
        return candidate.dx * candidate.dy

    x0, y0 = candidate.x, candidate.y
    x1, y1 = candidate.x + candidate.dx, candidate.y + candidate.dy
    rectangles: list[tuple[int, int, int, int]] = []
    for other in placed:
        if other.top != candidate.z:
            continue
        ix0 = max(x0, other.x)
        iy0 = max(y0, other.y)
        ix1 = min(x1, other.x + other.dx)
        iy1 = min(y1, other.y + other.dy)
        if ix0 < ix1 and iy0 < iy1:
            rectangles.append((ix0, iy0, ix1, iy1))
    return rectangle_union_area(rectangles)


def is_fully_supported(candidate: Placement, placed: Sequence[Placement]) -> bool:
    return support_area(candidate, placed) == candidate.dx * candidate.dy


def grid_step(items: Sequence[Item], master_box: tuple[int, int, int]) -> int:
    values = [*master_box, *(value for item in items for value in item.dims)]
    return reduce(math.gcd, values)


def lower_bound_height(items: Sequence[Item], box_x: int, box_y: int, step: int) -> int:
    volume_bound = math.ceil(sum(item.volume for item in items) / (box_x * box_y))
    orientation_bound = max(min(item.dims) for item in items)
    raw = max(volume_bound, orientation_bound)
    return int(math.ceil(raw / step) * step)


def flattest_orientation(item: Item) -> tuple[int, int, int]:
    return unique_orientations(item.dims)[0]


def priority_key(item: Item, target_height: int, floor_area: int) -> tuple:
    """Prioritize support platforms and floor-constrained pieces."""
    dx, dy, dz = flattest_orientation(item)
    base_area = dx * dy
    wide_platform = dz <= target_height // 3 and base_area >= floor_area // 5
    full_height_piece = dz == target_height
    structural_low_profile = dz <= target_height // 3 and base_area >= floor_area // 25

    if wide_platform:
        group = 0
    elif full_height_piece:
        group = 1
    elif structural_low_profile:
        group = 2
    else:
        group = 3
    return group, -base_area, -item.volume, item.id


def candidate_score(candidate: Placement, placed: Sequence[Placement]) -> tuple:
    all_boxes = [*placed, candidate]
    max_x = max(box.x + box.dx for box in all_boxes)
    max_y = max(box.y + box.dy for box in all_boxes)
    max_z = max(box.z + box.dz for box in all_boxes)
    footprint = max_x * max_y
    envelope_volume = footprint * max_z
    return (
        max_z,
        envelope_volume,
        footprint,
        -candidate.z,
        candidate.y,
        candidate.x,
        candidate.dx,
        candidate.dy,
        candidate.dz,
    )


def try_pack_at_height(
    items: Sequence[Item],
    master_box: tuple[int, int, int],
    target_height: int,
    step: int,
) -> list[Placement] | None:
    box_x, box_y, _ = master_box
    ordered = sorted(items, key=lambda item: priority_key(item, target_height, box_x * box_y))
    placed: list[Placement] = []

    for item in ordered:
        best: tuple[tuple, Placement] | None = None
        for dx, dy, dz in unique_orientations(item.dims):
            if dx > box_x or dy > box_y or dz > target_height:
                continue
            for z in range(0, target_height - dz + 1, step):
                for y in range(0, box_y - dy + 1, step):
                    for x in range(0, box_x - dx + 1, step):
                        candidate = Placement(item.id, item.type, x, y, z, dx, dy, dz)
                        if any(boxes_overlap(candidate, other) for other in placed):
                            continue
                        if not is_fully_supported(candidate, placed):
                            continue
                        score = candidate_score(candidate, placed)
                        if best is None or score < best[0]:
                            best = score, candidate
        if best is None:
            return None
        placed.append(best[1])
    return placed


def solve(items: Sequence[Item], master_box: tuple[int, int, int] = MASTER_BOX) -> list[Placement]:
    step = grid_step(items, master_box)
    start_height = lower_bound_height(items, master_box[0], master_box[1], step)
    for target_height in range(start_height, master_box[2] + 1, step):
        placements = try_pack_at_height(items, master_box, target_height, step)
        if placements is not None:
            validate_solution(items, placements, master_box)
            return placements
    raise PackingError("No valid packing found inside the master box")


def validate_solution(
    items: Sequence[Item],
    placements: Sequence[Placement],
    master_box: tuple[int, int, int] = MASTER_BOX,
) -> dict:
    by_id = {item.id: item for item in items}
    if len(placements) != len(items) or {p.id for p in placements} != set(by_id):
        raise PackingError("Solution does not contain every input item exactly once")

    for placement in placements:
        if min(placement.x, placement.y, placement.z) < 0:
            raise PackingError(f"Item {placement.id} has a negative coordinate")
        if placement.x + placement.dx > master_box[0]:
            raise PackingError(f"Item {placement.id} exceeds master-box X")
        if placement.y + placement.dy > master_box[1]:
            raise PackingError(f"Item {placement.id} exceeds master-box Y")
        if placement.z + placement.dz > master_box[2]:
            raise PackingError(f"Item {placement.id} exceeds master-box Z")
        if sorted((placement.dx, placement.dy, placement.dz)) != sorted(by_id[placement.id].dims):
            raise PackingError(f"Item {placement.id} dimensions are not a valid rotation")

    for index, first in enumerate(placements):
        for second in placements[index + 1:]:
            if boxes_overlap(first, second):
                raise PackingError(f"Items {first.id} and {second.id} overlap")

    settled: list[Placement] = []
    for placement in placements:
        if not is_fully_supported(placement, settled):
            ratio = support_area(placement, settled) / (placement.dx * placement.dy)
            raise PackingError(f"Item {placement.id} is only {ratio:.1%} supported")
        settled.append(placement)

    max_x = max(p.x + p.dx for p in placements)
    max_y = max(p.y + p.dy for p in placements)
    max_z = max(p.z + p.dz for p in placements)
    item_volume = sum(p.volume for p in placements)
    envelope_volume = max_x * max_y * max_z
    master_volume = math.prod(master_box)
    step = grid_step(items, master_box)
    height_lb = lower_bound_height(items, master_box[0], master_box[1], step)
    return {
        "valid": True,
        "item_count": len(placements),
        "grid_step": step,
        "occupied_bounds": [max_x, max_y, max_z],
        "item_volume": item_volume,
        "master_box_volume": master_volume,
        "master_volume_utilization": item_volume / master_volume,
        "occupied_envelope_volume": envelope_volume,
        "occupied_envelope_utilization": item_volume / envelope_volume,
        "maximum_height": max_z,
        "height_lower_bound": height_lb,
        "height_is_optimal": max_z == height_lb,
        "all_items_in_bounds": True,
        "overlap_free": True,
        "fully_supported": True,
    }


def save_solution(placements: Sequence[Placement], metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "master_box": list(MASTER_BOX),
        "placements": [asdict(p) for p in placements],
        "metrics": metrics,
    }
    (output_dir / "packing_solution.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (output_dir / "packing_solution.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "type", "x", "y", "z", "dx", "dy", "dz", "top_z"])
        for p in placements:
            writer.writerow([p.id, p.type, p.x, p.y, p.z, p.dx, p.dy, p.dz, p.top])


def print_solution(placements: Sequence[Placement], metrics: dict) -> None:
    print("\nGRAVITY-AWARE 3D PACKING SOLUTION")
    print("-" * 96)
    print(f"{'ID':>3} {'Type':<16} {'Position (x,y,z)':<19} {'Size (dx,dy,dz)':<19} {'Top':>5}")
    print("-" * 96)
    for p in placements:
        print(
            f"{p.id:>3} {p.type:<16} ({p.x:>3},{p.y:>3},{p.z:>3}){'':<6} "
            f"({p.dx:>3},{p.dy:>3},{p.dz:>3}){'':<6} {p.top:>5}"
        )
    print("-" * 96)
    print(f"Overlap-free: {metrics['overlap_free']} | Fully supported: {metrics['fully_supported']}")
    print(f"Occupied bounds: {metrics['occupied_bounds']} | Maximum height: {metrics['maximum_height']}")
    print(f"Height lower bound: {metrics['height_lower_bound']} | Height-optimal: {metrics['height_is_optimal']}")
    print(f"Master-box utilization: {metrics['master_volume_utilization']:.3%}")
    print(f"Occupied-envelope utilization: {metrics['occupied_envelope_utilization']:.3%}")


def cuboid_faces(p: Placement, z_override: float | None = None):
    z0 = float(p.z if z_override is None else z_override)
    x0, x1 = float(p.x), float(p.x + p.dx)
    y0, y1 = float(p.y), float(p.y + p.dy)
    z1 = z0 + p.dz
    vertices = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    index_faces = (
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    )
    return [[vertices[i] for i in face] for face in index_faces]


def wire_box_edges(size: tuple[int, int, int]):
    dx, dy, dz = size
    corners = [
        (0, 0, 0), (dx, 0, 0), (dx, dy, 0), (0, dy, 0),
        (0, 0, dz), (dx, 0, dz), (dx, dy, dz), (0, dy, dz),
    ]
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    return corners, edges


def draw_scene(
    ax,
    settled: Sequence[Placement],
    master_box: tuple[int, int, int],
    colors: dict[str, tuple],
    current: Placement | None = None,
    current_z: float | None = None,
    show_labels: bool = True,
) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    corners, edges = wire_box_edges(master_box)
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), linewidth=1.0, alpha=0.35, color="dimgray")

    def add_box(box: Placement, z_override: float | None = None, alpha: float = 0.78) -> None:
        faces = cuboid_faces(box, z_override=z_override)
        collection = Poly3DCollection(
            faces,
            facecolors=[colors[box.type]],
            edgecolors="black",
            linewidths=0.65,
            alpha=alpha,
        )
        ax.add_collection3d(collection)
        if show_labels:
            z_base = box.z if z_override is None else z_override
            ax.text(
                box.x + box.dx / 2,
                box.y + box.dy / 2,
                z_base + box.dz + 1.0,
                str(box.id),
                ha="center",
                va="bottom",
                fontsize=7,
                weight="bold",
            )

    for box in settled:
        add_box(box)
    if current is not None:
        add_box(current, z_override=current_z, alpha=0.88)

    ax.set_xlim(0, master_box[0])
    ax.set_ylim(0, master_box[1])
    ax.set_zlim(0, master_box[2])
    ax.set_box_aspect(master_box)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def render_static(
    placements: Sequence[Placement],
    metrics: dict,
    output_path: Path,
    master_box: tuple[int, int, int] = MASTER_BOX,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    types = sorted({p.type for p in placements})
    cmap = plt.get_cmap("tab20")
    colors = {name: cmap(index % 20) for index, name in enumerate(types)}

    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    draw_scene(ax, placements, master_box, colors)
    ax.view_init(elev=25, azim=38)
    ax.set_title(
        "Gravity-aware 3D packing\n"
        f"20 items | occupied bounds {metrics['occupied_bounds']} | "
        f"height-optimal = {metrics['height_is_optimal']}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_video(
    placements: Sequence[Placement],
    metrics: dict,
    output_path: Path,
    master_box: tuple[int, int, int] = MASTER_BOX,
    fps: int = 12,
    drop_frames: int = 7,
    final_hold_seconds: float = 3.0,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    types = sorted({p.type for p in placements})
    cmap = plt.get_cmap("tab20")
    colors = {name: cmap(index % 20) for index, name in enumerate(types)}

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    writer = FFMpegWriter(fps=fps, bitrate=3500, metadata={"title": "3D Gravity Packing Demo"})
    settled: list[Placement] = []

    with writer.saving(fig, str(output_path), dpi=100):
        for index, placement in enumerate(placements, start=1):
            start_z = min(master_box[2] - placement.dz, max(placement.z + 35, 55))
            for frame in range(drop_frames):
                t = (frame + 1) / drop_frames
                eased = 1.0 - (1.0 - t) ** 2
                animated_z = start_z + (placement.z - start_z) * eased

                fig.clear()
                ax = fig.add_subplot(111, projection="3d")
                draw_scene(ax, settled, master_box, colors, current=placement, current_z=animated_z)
                ax.view_init(elev=24, azim=35 + index * 1.3)
                ax.set_title("Part 2 - Gravity-aware 3D packing", fontsize=16, pad=18)
                fig.text(
                    0.025,
                    0.94,
                    f"Placing item {placement.id} ({placement.type})\n"
                    f"Position: ({placement.x}, {placement.y}, {placement.z})\n"
                    f"Orientation: {placement.dx} x {placement.dy} x {placement.dz}",
                    family="monospace",
                    fontsize=10.5,
                    va="top",
                )
                fig.text(
                    0.74,
                    0.94,
                    f"Placed: {index - 1}/20\nNo overlap: checked\nSupport: checked",
                    fontsize=10.5,
                    va="top",
                    bbox={"boxstyle": "round,pad=0.5", "alpha": 0.85},
                )
                fig.text(
                    0.5,
                    0.025,
                    "The item drops vertically and settles only at a fully supported position",
                    ha="center",
                    fontsize=10,
                )
                writer.grab_frame()

            settled.append(placement)
            # Brief settled frame for readability.
            fig.clear()
            ax = fig.add_subplot(111, projection="3d")
            draw_scene(ax, settled, master_box, colors)
            ax.view_init(elev=24, azim=35 + index * 1.3)
            ax.set_title("Part 2 - Gravity-aware 3D packing", fontsize=16, pad=18)
            fig.text(0.025, 0.94, f"[OK] Item {placement.id} settled\n[OK] {index}/20 items placed", family="monospace", fontsize=11, va="top")
            writer.grab_frame()

        final_frames = max(1, int(round(final_hold_seconds * fps)))
        for frame in range(final_frames):
            fig.clear()
            ax = fig.add_subplot(111, projection="3d")
            draw_scene(ax, placements, master_box, colors)
            ax.view_init(elev=24 + 3 * math.sin(frame / 8), azim=62 + 60 * frame / max(1, final_frames))
            ax.set_title("Packing complete - all constraints validated", fontsize=16, pad=18)
            fig.text(
                0.025,
                0.94,
                f"[OK] 20/20 items inside 100 x 100 x 100\n"
                f"[OK] No overlaps\n[OK] Every item fully supported",
                family="monospace",
                fontsize=10.5,
                va="top",
            )
            fig.text(
                0.72,
                0.94,
                f"Occupied bounds: {metrics['occupied_bounds']}\n"
                f"Maximum height: {metrics['maximum_height']}\n"
                f"Height-optimal: {metrics['height_is_optimal']}\n"
                f"Envelope utilization: {metrics['occupied_envelope_utilization']:.1%}",
                fontsize=10.5,
                va="top",
                bbox={"boxstyle": "round,pad=0.5", "alpha": 0.85},
            )
            writer.grab_frame()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=Path("data/Item List.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--save-video", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--drop-frames", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Reading item list: {args.items}")
    items = load_items(args.items)
    print(f"Loaded {len(items)} items")
    placements = solve(items, MASTER_BOX)
    metrics = validate_solution(items, placements, MASTER_BOX)
    print_solution(placements, metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_solution(placements, metrics, args.output_dir)
    render_static(placements, metrics, args.output_dir / "part2_packing_visualization.png")

    if args.save_video is not None:
        args.save_video.parent.mkdir(parents=True, exist_ok=True)
        print(f"Rendering MP4: {args.save_video}")
        render_video(
            placements,
            metrics,
            args.save_video,
            MASTER_BOX,
            fps=args.fps,
            drop_frames=args.drop_frames,
        )
        print(f"Saved: {args.save_video}")


if __name__ == "__main__":
    main()
