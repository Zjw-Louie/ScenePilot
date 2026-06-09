#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert v13 text-only JSONL SFT data into Scheme-A multimodal SFT data.

Input line format:
  {"messages": [...], "metadata": {"trajectory_dir": ".../scenes_growth/<traj_id>", "step_from": "step_00.json", ...}}

Output sample format:
  {
    "messages": [...],
    "images": ["<traj_id>/diag/step_00.jpg", "<traj_id>/top/step_00.jpg", "<traj_id>/annotated_top/step_00.jpg"],
    "metadata": {...}
  }

Use with training:
  --image_folder /home2/zhangjiawei/respace/training_data/growth2_frames
  --data_path /home2/zhangjiawei/respace/training_data/v13_sft_multimodal.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return rows


def get_traj_id(meta: Dict[str, Any]) -> str:
    traj_dir = str(meta.get("trajectory_dir", "")).rstrip("/")
    if not traj_dir:
        raise ValueError("metadata.trajectory_dir is missing")
    return Path(traj_dir).name


def step_json_to_jpg(step_from: str) -> str:
    # step_00.json -> step_00.jpg
    p = Path(step_from)
    if p.suffix.lower() != ".json":
        # fallback: keep stem if no suffix
        return p.stem + ".jpg"
    return p.with_suffix(".jpg").name


def build_image_relpaths(traj_id: str, step_jpg: str, views: List[str]) -> List[str]:
    # Relative to --image-root, e.g. <traj_id>/diag/step_00.jpg
    return [str(Path(traj_id) / view / step_jpg) for view in views]


def check_images_exist(image_root: Path, relpaths: List[str]) -> List[str]:
    missing = []
    for rel in relpaths:
        if not (image_root / rel).exists():
            missing.append(str(image_root / rel))
    return missing


def convert(
    input_path: Path,
    output_path: Path,
    image_root: Path,
    views: List[str],
    output_format: str,
    skip_missing: bool,
    rewrite_prompt: bool,
) -> None:
    samples = load_jsonl(input_path)
    out: List[Dict[str, Any]] = []
    skipped_missing = 0

    for i, sample in enumerate(samples):
        meta = sample.get("metadata") or {}
        traj_id = get_traj_id(meta)
        step_from = str(meta.get("step_from", ""))
        if not step_from:
            raise ValueError(f"Sample {i} missing metadata.step_from")
        step_jpg = step_json_to_jpg(step_from)
        images = build_image_relpaths(traj_id, step_jpg, views)

        missing = check_images_exist(image_root, images)
        if missing:
            if skip_missing:
                skipped_missing += 1
                continue
            raise FileNotFoundError(
                "Missing image files for sample "
                f"{i}, traj={traj_id}, step={step_from}:\n" + "\n".join(missing)
            )

        new_sample = dict(sample)
        new_sample["images"] = images

        # Optional: make user text explicitly refer to the 3 input images.
        # This is useful if the model otherwise cannot know image order/meaning.
        if rewrite_prompt:
            msgs = new_sample.get("messages", [])
            for msg in msgs:
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    prefix = (
                        "You are given three current-scene renderings in this fixed order:\n"
                        "Image 1: diagonal perspective render.\n"
                        "Image 2: top-down render.\n"
                        "Image 3: annotated top-down render with physical coordinates / visual marks.\n"
                        "Use only the current step images and current scene JSON. Do not assume access to the next-step image.\n\n"
                    )
                    if "Image 1: diagonal perspective render" not in msg["content"]:
                        msg["content"] = prefix + msg["content"]
                    break

        out.append(new_sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    elif output_format == "jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")

    print(f"input samples: {len(samples)}")
    print(f"output samples: {len(out)}")
    print(f"skipped_missing: {skipped_missing}")
    print(f"saved to: {output_path}")
    print(f"image_root for training --image_folder: {image_root}")
    if out:
        print("first sample images:")
        for img in out[0]["images"]:
            print("  ", img)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="Input v13 text-only JSONL file")
    ap.add_argument("--output", required=True, type=Path, help="Output multimodal JSON or JSONL file")
    ap.add_argument(
        "--image-root",
        required=True,
        type=Path,
        help="Root folder passed to training as --image_folder, e.g. /home2/.../training_data/growth2_frames",
    )
    ap.add_argument(
        "--views",
        default="diag,top,annotated_top",
        help="Comma-separated view subfolders under each trajectory id",
    )
    ap.add_argument("--output-format", choices=["json", "jsonl"], default="json")
    ap.add_argument("--skip-missing", action="store_true", help="Skip samples with missing images instead of failing")
    ap.add_argument("--rewrite-prompt", action="store_true", help="Prepend image-order description to user message")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    convert(
        input_path=args.input,
        output_path=args.output,
        image_root=args.image_root,
        views=views,
        output_format=args.output_format,
        skip_missing=args.skip_missing,
        rewrite_prompt=args.rewrite_prompt,
    )


if __name__ == "__main__":
    main()
