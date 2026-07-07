# 3D Measurement and Gravity-Aware Packing Assignment

This submission directly implements both parts of the supplied assignment.

## Contents

- `src/part1_obb_measurement.py` - reads all three OBJ files, computes high-precision oriented bounding boxes, reports dimensions/volume, and creates the Part 1 visualization/video.
- `src/part2_gravity_packing.py` - ingests the supplied JSON, computes coordinates and rotations, enforces non-overlap and gravity support, and creates the one-by-one packing visualization/video.
- `src/validate_submission.py` - independently rechecks output consistency and every Part 2 constraint.
- `outputs/` - CSV/JSON results, static visualizations, and submission-ready MP4 videos.
- `data/` - the exact files supplied with the assignment.

## Run

Python 3.10+ and FFmpeg are recommended.

```bash
python -m pip install -r requirements.txt
bash run_all.sh
```

On Windows Command Prompt:

```bat
run_all.bat
```

Individual commands:

```bash
python src/part1_obb_measurement.py --input-dir data --output-dir outputs \
  --save-video outputs/part1_obb_demo.mp4

python src/part2_gravity_packing.py --items "data/Item List.json" --output-dir outputs \
  --save-video outputs/part2_packing_demo.mp4
```

## Part 1 method

The script loads the complete mesh and computes an oriented bounding box from convex-hull candidate orientations using `trimesh.bounds.oriented_bounds(..., angle_digits=4)`. The full geometry is used for measurement. Only the displayed points are sampled for smooth animation.

Dimensions are reported as `Length x Width x Height` from largest to smallest OBB extent. The JSON also stores the local OBB extents, transform matrix, and eight world-space box corners.

## Part 2 method and guarantees

The solver automatically derives a 5-unit grid from the greatest common divisor of all dimensions. It:

1. Tries every distinct 90-degree orientation.
2. Searches height starting at a mathematical lower bound.
3. Rejects candidates outside the 100 x 100 x 100 master box.
4. Rejects every pairwise overlap.
5. Computes exact union coverage under each base and requires 100% support from the floor or previously placed items.
6. Minimizes maximum occupied height first, then occupied envelope volume and footprint.

For the supplied list, the solution uses an occupied envelope of `100 x 80 x 30`. Its maximum height is 30, equal to the lower bound caused by the 30 x 30 x 30 crates, so the height is optimal. All 20 items are inside the master box, overlap-free, and fully supported.
