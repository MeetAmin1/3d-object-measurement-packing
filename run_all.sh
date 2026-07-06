#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python src/part1_obb_measurement.py \
  --input-dir data \
  --output-dir outputs \
  --save-video outputs/part1_obb_demo.mp4
python src/part2_gravity_packing.py \
  --items "data/Item List.json" \
  --output-dir outputs \
  --save-video outputs/part2_packing_demo.mp4
python src/validate_submission.py
