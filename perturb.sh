#!/bin/bash
set -euo pipefail

SCRIPT=/home2/zhangjiawei/respace/scripts/perturb_scene_for_grpo.py
SRC_ROOT=/home2/zhangjiawei/respace/benchmark/scenes_filter
DST_ROOT=/home2/zhangjiawei/respace/training_data/scenes_filter_scale_only

for ROOM in bedroom livingroom diningroom
do
  python "$SCRIPT" \
    --in_dir "${SRC_ROOT}/${ROOM}" \
    --out_dir "${DST_ROOT}/${ROOM}" \
    --disable_pos \
    --disable_rot \
    --enable_scale \
    --sx 0.22 --sy 0.10 --sz 0.22 \
    --min_sx 0.7 --min_sy 0.9 --min_sz 0.7 \
    --max_sx 1.4 --max_sy 1.1 --max_sz 1.4 \
    --n_perturb_objects 2 \
    --skip_hanging_for_scale \
    --write_meta \
    --seed 123
done