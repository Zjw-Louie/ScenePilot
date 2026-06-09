#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------
# Project import setup
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.respace import ReSpace
from src.viz import render_annotated_top_view


DEFAULT_ROOTS = [
    "/home2/zhangjiawei/respace/results/ablation_rag_group_123",
    "/home2/zhangjiawei/respace/results/ablation_rag_group_456",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def find_scene_jsons(root: Path) -> List[Path]:
    """
    Find all final_scene_from_group_repair.json under a root.
    """
    if not root.exists():
        return []
    return sorted(root.rglob("scene_respace_renderable.json"))


def has_existing_render(final_dir: Path) -> bool:
    """
    Check whether final_dir already contains rendered images.
    """
    if not final_dir.exists():
        return False
    exts = {".png", ".jpg", ".jpeg"}
    for p in final_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            return True
    return False


def load_scene(scene_json_path: Path) -> dict:
    return json.loads(scene_json_path.read_text(encoding="utf-8"))


def render_one_scene(
    respace: ReSpace,
    scene_json_path: Path,
    force: bool = False,
    resolution: Tuple[int, int] = (1024, 1024),
) -> bool:
    """
    Render one scene from final_scene_from_group_repair.json
    into <scene_dir>/final/.
    """
    scene_dir = scene_json_path.parent
    final_dir = scene_dir / "final"

    if (not force) and has_existing_render(final_dir):
        log(f"[SKIP] already rendered: {scene_dir}")
        return True

    final_dir.mkdir(parents=True, exist_ok=True)

    try:
        scene = load_scene(scene_json_path)
    except Exception as e:
        log(f"[FAIL] load scene json failed: {scene_json_path} | {e}")
        return False

    ok = True

    # Save a copy of the scene json into final/scene.json
    try:
        (final_dir / "scene.json").write_text(
            json.dumps(scene, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as e:
        ok = False
        log(f"[WARN] failed to write scene copy: {final_dir / 'scene.json'} | {e}")

    # RGB / perspective render
    try:
        respace.render_scene_frame(
            scene,
            filename="final",
            pth_viz_output=final_dir,
        )
        log(f"[OK] RGB render saved under: {final_dir}")
    except Exception as e:
        ok = False
        log(f"[FAIL] render_scene_frame failed: {scene_json_path}")
        log(f"       error: {e}")
        traceback.print_exc()

    # annotated top render
    try:
        top_path = render_annotated_top_view(
            scene,
            "final",
            final_dir,
            resolution=resolution,
            show_assets=True,
            font_size=14,
        )
        log(f"[OK] annotated top render saved: {top_path}")
    except Exception as e:
        ok = False
        log(f"[FAIL] render_annotated_top_view failed: {scene_json_path}")
        log(f"       error: {e}")
        traceback.print_exc()

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill missing renders for rag_group result folders."
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=DEFAULT_ROOTS,
        help="Root directories to scan.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render even if final/ already contains images.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Annotated top render width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Annotated top render height.",
    )
    args = parser.parse_args()

    roots = [Path(x).expanduser() for x in args.roots]
    resolution = (args.width, args.height)

    log("============================================================")
    log("Backfill missing renders for rag_group results")
    log("============================================================")
    for r in roots:
        log(f"[ROOT] {r}")

    # Load ReSpace once
    log("[INFO] loading ReSpace once...")
    respace = ReSpace()

    all_scene_jsons: List[Path] = []
    for root in roots:
        found = find_scene_jsons(root)
        log(f"[SCAN] {root} -> found {len(found)} scene json(s)")
        all_scene_jsons.extend(found)

    all_scene_jsons = sorted(set(all_scene_jsons))
    total = len(all_scene_jsons)
    ok_cnt = 0
    fail_cnt = 0
    skip_cnt = 0

    log(f"[INFO] total scene jsons to consider: {total}")

    for idx, scene_json_path in enumerate(all_scene_jsons, start=1):
        scene_dir = scene_json_path.parent
        final_dir = scene_dir / "final"

        log("------------------------------------------------------------")
        log(f"[{idx}/{total}] scene_dir = {scene_dir}")
        log(f"[{idx}/{total}] scene_json = {scene_json_path}")

        if (not args.force) and has_existing_render(final_dir):
            log(f"[SKIP] final render already exists: {final_dir}")
            skip_cnt += 1
            continue

        success = render_one_scene(
            respace=respace,
            scene_json_path=scene_json_path,
            force=args.force,
            resolution=resolution,
        )
        if success:
            ok_cnt += 1
        else:
            fail_cnt += 1

    log("")
    log("============================================================")
    log("[DONE]")
    log(f"total = {total}")
    log(f"ok    = {ok_cnt}")
    log(f"fail  = {fail_cnt}")
    log(f"skip  = {skip_cnt}")
    log("============================================================")


if __name__ == "__main__":
    main()
    
# python render_miss.py \
#   --roots /home2/zhangjiawei/respace/results/reason3d \
#   --force \
#   --width 1024 \
#   --height 1024