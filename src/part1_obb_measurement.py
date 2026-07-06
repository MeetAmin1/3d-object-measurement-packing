#!/usr/bin/env python3
"""Part 1: Measure raw 3D meshes with a minimum-volume oriented bounding box.

The script:
1. Reads CUBE.obj, CYLINDER.obj, and TEAPOT.obj.
2. Computes a high-precision oriented bounding box (OBB) from each mesh's convex hull.
3. Prints and saves dimensions and volume.
4. Produces a static visualization and, optionally, an MP4 demonstration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh


DEFAULT_FILES = ("CUBE.obj", "CYLINDER.obj", "TEAPOT.obj")


@dataclass(frozen=True)
class OBBMeasurement:
    filename: str
    vertex_count: int
    face_count: int
    dimensions_lwh: tuple[float, float, float]
    volume: float
    local_extents: tuple[float, float, float]
    world_to_obb: list[list[float]]
    obb_corners_world: list[list[float]]


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load an OBJ safely and return one geometry with scene transforms applied."""
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"No geometry found in {path}")
        mesh = loaded.to_geometry()
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported geometry type in {path}: {type(loaded).__name__}")

    if len(mesh.vertices) < 4:
        raise ValueError(f"{path} does not contain enough 3D points for an OBB")
    return mesh


def compute_obb(mesh: trimesh.Trimesh, filename: str) -> OBBMeasurement:
    """Compute an oriented box using convex-hull candidate orientations.

    angle_digits=4 avoids the coarse angular merging used by Trimesh's default
    while keeping runtime modest for these meshes.
    """
    world_to_obb, extents = trimesh.bounds.oriented_bounds(
        mesh, angle_digits=4, ordered=True
    )
    extents = np.asarray(extents, dtype=float)

    # Local box corners are transformed back into the original mesh coordinates.
    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    local_corners = signs * (extents / 2.0)
    obb_to_world = np.linalg.inv(world_to_obb)
    world_corners = trimesh.transform_points(local_corners, obb_to_world)

    # L/W/H are reported from largest to smallest. The OBB remains fully 3D;
    # these labels are simply a conventional presentation of its three extents.
    length, width, height = sorted((float(v) for v in extents), reverse=True)
    return OBBMeasurement(
        filename=filename,
        vertex_count=int(len(mesh.vertices)),
        face_count=int(len(mesh.faces)),
        dimensions_lwh=(length, width, height),
        volume=float(np.prod(extents)),
        local_extents=tuple(float(v) for v in extents),
        world_to_obb=np.asarray(world_to_obb, dtype=float).tolist(),
        obb_corners_world=np.asarray(world_corners, dtype=float).tolist(),
    )


def box_edges() -> tuple[tuple[int, int], ...]:
    return (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )


def _sample_vertices(vertices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(vertices) <= maximum:
        return vertices
    rng = np.random.default_rng(seed)
    return vertices[rng.choice(len(vertices), size=maximum, replace=False)]


def _set_equal_3d_limits(ax, points: np.ndarray, pad: float = 0.10) -> None:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) * (0.5 + pad), 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def render_static(
    meshes: list[trimesh.Trimesh],
    measurements: list[OBBMeasurement],
    output_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 5), dpi=160)
    for index, (mesh, result) in enumerate(zip(meshes, measurements), start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        points = _sample_vertices(np.asarray(mesh.vertices), maximum=6000, seed=index)
        corners = np.asarray(result.obb_corners_world)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.35, alpha=0.48)
        for a, b in box_edges():
            ax.plot(*zip(corners[a], corners[b]), linewidth=2.0, color="crimson")
        _set_equal_3d_limits(ax, np.vstack((points, corners)))
        l, w, h = result.dimensions_lwh
        ax.set_title(
            f"{result.filename}\nOBB: {l:.4f} × {w:.4f} × {h:.4f}\n"
            f"Volume: {result.volume:.4f}",
            fontsize=9,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=24, azim=35)
    fig.suptitle("Oriented Bounding Boxes (full mesh used for measurement)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_video(
    meshes: list[trimesh.Trimesh],
    measurements: list[OBBMeasurement],
    output_path: Path,
    fps: int = 12,
    seconds_per_object: float = 4.0,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    frames_per_object = max(24, int(round(fps * seconds_per_object)))
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    writer = FFMpegWriter(fps=fps, bitrate=3200, metadata={"title": "3D OBB Measurement Demo"})

    sampled = [
        _sample_vertices(np.asarray(mesh.vertices), maximum=5000, seed=100 + i)
        for i, mesh in enumerate(meshes)
    ]

    with writer.saving(fig, str(output_path), dpi=100):
        for index, (mesh, result, points) in enumerate(zip(meshes, measurements, sampled)):
            corners = np.asarray(result.obb_corners_world)
            all_points = np.vstack((points, corners))
            for frame in range(frames_per_object):
                fig.clear()
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.7, alpha=0.55)
                for a, b in box_edges():
                    ax.plot(*zip(corners[a], corners[b]), linewidth=2.6, color="crimson")
                _set_equal_3d_limits(ax, all_points)
                ax.view_init(elev=22 + 6 * math.sin(frame / 11), azim=20 + 360 * frame / frames_per_object)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")

                l, w, h = result.dimensions_lwh
                ax.set_title(f"Part 1 - Reading and measuring {result.filename}", fontsize=16, pad=18)
                fig.text(
                    0.025,
                    0.94,
                    f"[OK] Read {result.filename}\n"
                    f"[OK] Vertices: {result.vertex_count:,} | Faces: {result.face_count:,}\n"
                    f"[OK] Minimum-volume OBB calculated",
                    fontsize=10.5,
                    family="monospace",
                    va="top",
                )
                fig.text(
                    0.72,
                    0.94,
                    f"Dimensions (L x W x H)\n{l:.4f} x {w:.4f} x {h:.4f}\n"
                    f"OBB volume: {result.volume:.4f}",
                    fontsize=11,
                    va="top",
                    bbox={"boxstyle": "round,pad=0.5", "alpha": 0.85},
                )
                fig.text(
                    0.5,
                    0.025,
                    f"Object {index + 1} of {len(meshes)} | The wireframe is the oriented bounding box",
                    ha="center",
                    fontsize=10,
                )
                writer.grab_frame()
    plt.close(fig)


def save_results(measurements: Iterable[OBBMeasurement], output_dir: Path) -> None:
    results = list(measurements)
    with (output_dir / "obb_results.json").open("w", encoding="utf-8") as file:
        json.dump([asdict(r) for r in results], file, indent=2)

    with (output_dir / "obb_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "length", "width", "height", "obb_volume", "vertices", "faces"])
        for r in results:
            writer.writerow([r.filename, *r.dimensions_lwh, r.volume, r.vertex_count, r.face_count])


def print_results(measurements: Iterable[OBBMeasurement]) -> None:
    print("\nORIENTED BOUNDING BOX RESULTS")
    print("-" * 96)
    print(f"{'File':<16} {'Length':>12} {'Width':>12} {'Height':>12} {'Volume':>15} {'Vertices':>12}")
    print("-" * 96)
    for result in measurements:
        l, w, h = result.dimensions_lwh
        print(
            f"{result.filename:<16} {l:>12.6f} {w:>12.6f} {h:>12.6f} "
            f"{result.volume:>15.6f} {result.vertex_count:>12,}"
        )
    print("-" * 96)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--files", nargs="+", default=list(DEFAULT_FILES))
    parser.add_argument("--save-video", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seconds-per-object", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meshes: list[trimesh.Trimesh] = []
    measurements: list[OBBMeasurement] = []
    for name in args.files:
        path = args.input_dir / name
        print(f"Reading: {path}")
        mesh = load_mesh(path)
        print(f"  loaded {len(mesh.vertices):,} vertices and {len(mesh.faces):,} faces")
        result = compute_obb(mesh, filename=path.name)
        l, w, h = result.dimensions_lwh
        print(f"  OBB dimensions = {l:.6f} x {w:.6f} x {h:.6f}; volume = {result.volume:.6f}")
        meshes.append(mesh)
        measurements.append(result)

    print_results(measurements)
    save_results(measurements, args.output_dir)
    render_static(meshes, measurements, args.output_dir / "part1_obb_visualization.png")

    if args.save_video is not None:
        args.save_video.parent.mkdir(parents=True, exist_ok=True)
        print(f"Rendering MP4: {args.save_video}")
        render_video(
            meshes,
            measurements,
            args.save_video,
            fps=args.fps,
            seconds_per_object=args.seconds_per_object,
        )
        print(f"Saved: {args.save_video}")


if __name__ == "__main__":
    main()
