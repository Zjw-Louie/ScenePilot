#!/usr/bin/env python3
"""
Batch evaluation for generated 3D scenes.

What this script computes:
1) Physical metrics: OOB / MBL / PBL
2) Semantic matching: txt_pms_score / txt_pms_sampled_score
3) Image statistics from existing render folders: FID / CLIP-FID / KID / Diversity
4) Optional VLM judge scores from diag/top renders via Yunwu OpenAI-compatible API

Supported input layouts
=======================
Layout A (flat scene jsons + flat render root)
------------------------------------------------
scenes_root/
  *.json
renders_root/
  <scene_id>/diag/
  <scene_id>/top/

Example:
  /.../batch_outputs_baseline_123/updated_scenes/*.json
  /.../batch_outputs_baseline_123/renders/<scene_id>/{diag,top}

Layout B (per-scene folders with final/)
-----------------------------------------
root/
  <scene_id>/final/scene.json
  <scene_id>/final/diag/
  <scene_id>/final/top/

Example:
  /.../batch_outputs_rag_planning_only_123/<scene_id>/final/scene.json
  /.../batch_outputs_rag_planning_only_123/<scene_id>/final/{diag,top}

This script keeps the physical and semantic metric definitions aligned with the user's eval.py,
while changing the I/O layer so it can directly evaluate an already-generated batch.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import pickle
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import trimesh
from dotenv import load_dotenv
from shapely.geometry import Polygon
from tqdm import tqdm
from trimesh.transformations import quaternion_matrix

# Reuse your repo utilities for mesh lookup, floor polygon creation, and image metrics.
from src.utils import (
    compute_diversity_score,
    compute_fid_scores,
    create_floor_plan_polygon,
    get_pth_mesh,
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# -----------------------------
# Optional GPT/VLM scene judge
# -----------------------------

DEFAULT_GPT_JUDGE_PROMPT = """You are an expert reviewer for indoor 3D scene layout and furniture arrangement.

You will be given rendered images of the SAME room, typically including:
- a top view, which is most reliable for judging global spatial arrangement, circulation, alignment, and object overlap;
- a diagonal view, which is most reliable for judging realism, object scale, accessibility, and whether furniture placement looks usable in perspective.

You may also be given optional textual context, such as the room type or a natural-language design request. If textual context is provided, use it only as auxiliary information. The primary judgment must come from the rendered images.

Your task is to evaluate the scene on exactly three criteria:

1) lc: Layout correctness
2) spa: Semantic plausibility
3) fc: Functional completeness

General evaluation principles:
- Be generous .
- Judge only what is visible in the provided images and optional textual context.
- Do not assume hidden furniture, invisible functional areas, or unshown geometry.
- If the two views disagree, use both views together and penalize uncertainty or visible inconsistency.
- Penalize physical or spatial problems such as:
  - obvious object-object collisions or severe overlap,
  - furniture intersecting walls or being partly outside the room,
  - blocked circulation or inaccessible furniture,
  - awkward spacing, unnatural placement, or unusable arrangements,
  - severe misalignment between related furniture pieces,
  - implausible scale or proportion,
  - missing core furniture required for the room’s apparent function.
- Prefer concise but informative reasons.
- Use the full score range from 7 to 10 when appropriate.
- Scores do not need to be integers; use floats.

Detailed scoring criteria:

A) lc: Layout correctness
Definition:
Evaluate the geometric and spatial quality of the arrangement itself, regardless of style preference.

Focus on:
- whether objects are placed in reasonable positions inside the room boundary;
- whether there are collisions, overlaps, wall intersections, or out-of-bound placements;
- whether there is clear and usable free space for movement;
- whether furniture alignment, orientation, spacing, and grouping are spatially coherent;
- whether the overall composition looks organized rather than chaotic or arbitrarily scattered.

High lc score (8-10):
- Objects are well placed, mostly collision-free, inside the room, and support clear circulation.
- Relative positions and orientations are coherent.
- Major furniture is arranged in a spatially sensible and usable way.

Medium lc score (6-7):
- Layout is partially reasonable but has noticeable spacing issues, weak alignment, mild obstruction, or questionable placement.

Low lc score (3-5):
- Serious collisions, blocked walkways, unusable access, severe boundary violations, or obviously broken placement.

B) spa: Semantic plausibility
Definition:
Evaluate whether the scene makes semantic sense as a believable room layout for its apparent room type and intended use.

Focus on:
- whether object relationships are semantically appropriate;
- whether furniture types and pairings make sense together;
- whether items are positioned in functionally meaningful relations (e.g., seating around a table, bedside furniture near a bed, desk and chair pairing, TV facing seating, storage placed sensibly);
- whether scales, orientations, and usage relationships look realistic;
- whether the room appears like a plausible human-designed interior rather than a random collection of objects.

High spa score (9-10):
- Furniture relationships are natural and believable.
- The room reads clearly as an intended functional space.
- Object placement supports typical human use patterns.

Medium spa score (7-8):
- Scene is somewhat believable but contains odd pairings, weak semantic grouping, or several unnatural relations.

Low spa score (5-6):
- Scene appears semantically confused, implausible, or obviously unrealistic for the room type.

C) fc: Functional completeness
Definition:
Evaluate whether the room contains enough of the key furniture and arrangement structure needed to support its intended function.

Focus on:
- whether essential furniture for the apparent room type is present;
- whether the scene supports the main activity of the room;
- whether the functional zones look complete rather than partial or under-specified;
- whether the room feels missing major items that would normally be necessary.

Examples:
- A bedroom should usually include a bed and enough supporting furniture to make the room feel usable.
- A living room should usually include core seating and a coherent social or media focus.
- A dining room should usually include a dining table and appropriate seating.
- A study/workspace should usually include a desk or work surface and seating.
These are examples only; judge based on what is visible and any provided textual context.

High fc score (8-10):
- The room includes the major furniture needed for its purpose and feels functionally usable and reasonably complete.

Medium fc score (5-7):
- Some core functionality is present, but important supporting furniture or functional structure is missing.

Low fc score (3-4):
- The room is missing essential furniture and does not adequately support its intended use.

Scoring instructions:
- Each criterion score must be a float in [0, 10].
- 0 means extremely poor.
- 10 means excellent.
- Be conservative: visible flaws should meaningfully reduce the score.
- Do not inflate scores just because the render looks visually clean.
- If a scene is physically broken or clearly unusable, lc should be low even if the furniture categories seem correct.
- If the room contains plausible objects but lacks key function, fc should be low.
- If the arrangement is collision-free but semantically awkward, spa should be low.

Reason instructions:
- For each criterion, provide a short reason grounded in visible evidence.
- Mention the most important positive or negative factors only.
- Avoid long explanations, speculation, or restating the rubric.

Output instructions:
Return ONLY one JSON object with exactly this schema:
{
  "lc": {"score": 0.0, "reason": ""},
  "spa": {"score": 0.0, "reason": ""},
  "fc": {"score": 0.0, "reason": ""},
  "overall": 0.0
}

Additional output constraints:
- Do not output markdown.
- Do not output code fences.
- Do not output any text before or after the JSON object.
- The "overall" field must be the arithmetic average of lc.score, spa.score, and fc.score.
- Ensure the JSON is valid and directly parseable.
"""


def _clamp_score(x: object, low: float = 0.0, high: float = 10.0) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return max(low, min(high, v))


def _find_first_json_dict(text: str) -> Dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response.")

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"Could not locate JSON object in model response: {text[:500]}")

    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")
    return obj


def image_path_to_data_url(img_path: Path) -> str:
    suffix = img_path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_scene_prompt(scene: Dict) -> Optional[str]:
    for key in ["prompt", "scene_prompt", "text", "instruction", "user_prompt"]:
        value = scene.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    objs = scene.get("objects") or []
    candidate_prompts = []
    for obj in objs:
        value = obj.get("prompt")
        if isinstance(value, str) and value.strip():
            candidate_prompts.append(value.strip())
    if candidate_prompts:
        return candidate_prompts[-1]
    return None


def _dedup_keep_order(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _looks_like_scene_dict(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    # Keep this permissive enough for converted Reason3D/ReSpace scenes,
    # while excluding batch-level metadata jsons/lists.
    sceneish_keys = {
        "objects", "bounds_bottom", "bounds_top", "room_type",
        "scene_id", "prompt", "scene_prompt", "user_prompt"
    }
    return any(k in obj for k in sceneish_keys)


def _is_valid_scene_json_file(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return _looks_like_scene_dict(obj)
    except Exception:
        return False


def _dir_has_scene_jsons(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False

    # Prefer actual per-scene outputs over loose batch-level json files.
    if any(p.is_file() for p in root.glob("*/final/scene.json")):
        return True
    if any(p.is_file() for p in root.glob("*/final/final/scene.json")):
        return True
    if any(
        p.is_file() and p.name == "scene.json" and p.parent.name == "final"
        for p in root.rglob("scene.json")
    ):
        return True

    # Flat-json layout is only valid if at least one json actually looks like a scene dict.
    for p in root.glob("*.json"):
        if p.is_file() and _is_valid_scene_json_file(p):
            return True
    return False


def _dir_has_render_layout(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False

    # Layout A: root/<scene_id>/diag or root/<scene_id>/top
    # Layout B: root/<scene_id>/final/diag or root/<scene_id>/final/top
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if (d / "diag").is_dir() or (d / "top").is_dir():
            return True
        if (d / "final" / "diag").is_dir() or (d / "final" / "top").is_dir():
            return True

    # Recursive fallback
    if any(p.is_dir() and p.name in {"diag", "top"} for p in root.rglob("*")):
        return True

    return False


def resolve_scenes_root(user_path: Path) -> Path:
    """
    Allow user to pass:
    1) updated_scenes root
    2) nested-final root
    3) whole batch root
    """
    candidates = [
        user_path,
        user_path / "updated_scenes",
        user_path / "scenes",
    ]

    for c in candidates:
        if _dir_has_scene_jsons(c):
            return c.resolve()

    if user_path.exists() and user_path.is_dir():
        for p in [user_path] + [x for x in user_path.rglob("*") if x.is_dir()]:
            if _dir_has_scene_jsons(p):
                return p.resolve()

    raise FileNotFoundError(
        f"Cannot resolve scenes_root from input: {user_path}\n"
        "Expected one of:\n"
        "  - updated_scenes/*.json\n"
        "  - <scene_id>/final/scene.json"
    )


def resolve_renders_root(user_path: Path) -> Path:
    """
    Allow user to pass:
    1) renders root
    2) nested-final root
    3) whole batch root
    """
    candidates = [
        user_path,
        user_path / "renders",
    ]

    for c in candidates:
        if _dir_has_render_layout(c):
            return c.resolve()

    if user_path.exists() and user_path.is_dir():
        for p in [user_path] + [x for x in user_path.rglob("*") if x.is_dir()]:
            if _dir_has_render_layout(p):
                return p.resolve()

    raise FileNotFoundError(
        f"Cannot resolve renders_root from input: {user_path}\n"
        "Expected one of:\n"
        "  - renders/<scene_id>/{diag,top}\n"
        "  - <scene_id>/final/{diag,top}"
    )


def collect_scene_render_images(
    renders_root: Path,
    scene_id: str,
    max_images_per_view: int = 2,
) -> Dict[str, List[Path]]:
    """
    Compatible with:
    1) renders/<scene_id>/diag
    2) renders/<scene_id>/final/diag
    3) whole batch root input, auto-resolved
    4) recursive fallback search by scene_id
    """
    renders_root = resolve_renders_root(renders_root)

    def _collect(view: str) -> List[Path]:
        candidate_dirs = [
            renders_root / scene_id / view,
            renders_root / scene_id / "final" / view,
            renders_root / scene_id / "final" / "final" / view,
        ]

        # Recursive fallback
        for p in renders_root.rglob(view):
            if not p.is_dir() or p.name != view:
                continue
            try:
                sid = infer_scene_id_from_view_dir(p)
            except Exception:
                sid = None
            if sid == scene_id:
                candidate_dirs.append(p)

        candidate_dirs = _dedup_keep_order([p for p in candidate_dirs if p.is_dir()])

        imgs: List[Path] = []
        seen = set()
        for cdir in candidate_dirs:
            for p in sorted(cdir.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    key = str(p.resolve())
                    if key not in seen:
                        seen.add(key)
                        imgs.append(p)

        return imgs[:max_images_per_view]

    return {
        "diag": _collect("diag"),
        "top": _collect("top"),
    }


def call_yunwu_multiview_scene_judge(
    diag_images: Sequence[Path],
    top_images: Sequence[Path],
    room_type: Optional[str],
    scene_prompt: Optional[str],
    scene_id: Optional[str],
    api_key: str,
    base_url: str = "https://yunwu.ai/v1",
    model: str = "gpt-5.4",
    timeout: int = 180,
    max_retries: int = 2,
) -> Dict:
    if not api_key:
        raise ValueError("Missing Yunwu API key. Pass --yunwu-api-key or set YUNWU_API_KEY.")
    if len(diag_images) == 0 and len(top_images) == 0:
        raise ValueError("No render images found for GPT judge.")

    context_lines = []
    if scene_id:
        context_lines.append(f"Scene id: {scene_id}")
    if room_type:
        context_lines.append(f"Room type: {room_type}")
    if scene_prompt:
        context_lines.append(f"Text prompt: {scene_prompt}")

    content = [{
        "type": "text",
        "text": DEFAULT_GPT_JUDGE_PROMPT + "\n\nContext:\n" + ("\n".join(context_lines) if context_lines else "(none)")
    }]

    if top_images:
        content.append({"type": "text", "text": f"Top-view render(s): {len(top_images)} image(s)."})
        for img in top_images:
            content.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(img)}})

    if diag_images:
        content.append({"type": "text", "text": f"Diagonal-view render(s): {len(diag_images)} image(s)."})
        for img in diag_images:
            content.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(img)}})

    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
    }

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:1000]}")
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            obj = _find_first_json_dict(text)

            lc_score = _clamp_score((obj.get("lc") or {}).get("score")) if isinstance(obj.get("lc"), dict) else _clamp_score(obj.get("lc"))
            spa_score = _clamp_score((obj.get("spa") or {}).get("score")) if isinstance(obj.get("spa"), dict) else _clamp_score(obj.get("spa"))
            fc_score = _clamp_score((obj.get("fc") or {}).get("score")) if isinstance(obj.get("fc"), dict) else _clamp_score(obj.get("fc"))

            if lc_score is None or spa_score is None or fc_score is None:
                raise ValueError(f"Model JSON missing lc/spa/fc score fields: {obj}")

            overall = _clamp_score(obj.get("overall"))
            if overall is None:
                overall = float((lc_score + spa_score + fc_score) / 3.0)

            return {
                "lc": {
                    "score": float(lc_score),
                    "reason": str((obj.get("lc") or {}).get("reason", "")) if isinstance(obj.get("lc"), dict) else "",
                },
                "spa": {
                    "score": float(spa_score),
                    "reason": str((obj.get("spa") or {}).get("reason", "")) if isinstance(obj.get("spa"), dict) else "",
                },
                "fc": {
                    "score": float(fc_score),
                    "reason": str((obj.get("fc") or {}).get("reason", "")) if isinstance(obj.get("fc"), dict) else "",
                },
                "overall": float(overall),
                "model": model,
                "n_diag_images": len(diag_images),
                "n_top_images": len(top_images),
                "raw_response": text,
            }
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"Yunwu GPT judge failed after {max_retries + 1} attempts: {e}") from e

    raise RuntimeError(f"Yunwu GPT judge failed: {last_err}")


def score_scene_renders_with_yunwu(
    scene_id: str,
    scene: Dict,
    renders_root: Path,
    api_key: str,
    base_url: str = "https://yunwu.ai/v1",
    model: str = "gpt-5.4",
    timeout: int = 180,
    max_images_per_view: int = 2,
) -> Dict:
    render_images = collect_scene_render_images(
        renders_root=renders_root,
        scene_id=scene_id,
        max_images_per_view=max_images_per_view,
    )
    return call_yunwu_multiview_scene_judge(
        diag_images=render_images["diag"],
        top_images=render_images["top"],
        room_type=scene.get("room_type"),
        scene_prompt=extract_scene_prompt(scene),
        scene_id=scene_id,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
    )


# -----------------------------
# Geometry / physical metrics
# -----------------------------

def get_y_angle_from_xyzw_quaternion(quaternion_xyzw: Sequence[float]) -> Tuple[float, float]:
    x, y, z, w = quaternion_xyzw
    angle_yaw_radians = np.arctan2(2 * (w * y + x * z), 1 - 2 * (y**2 + z**2))
    angle_yaw_degrees = float(np.round(np.degrees(angle_yaw_radians), 1))
    return angle_yaw_degrees, float(angle_yaw_radians)


def get_xz_bbox_from_obj(obj: Dict) -> Tuple[Polygon, float, float, float]:
    bbox_position = obj.get("pos")
    bbox_size = obj.get("size")
    if bbox_position is None or bbox_size is None:
        raise ValueError(f"Object missing pos/size: {obj}")

    rotation_xyzw = np.array(obj.get("rot", [0, 0, 0, 1]), dtype=float)
    _, asset_rot_angle_radians = get_y_angle_from_xyzw_quaternion(rotation_xyzw)

    half_size_x = bbox_size[0] / 2
    half_size_z = bbox_size[2] / 2
    corners_2d_floor = np.array([
        [half_size_x, half_size_z],
        [-half_size_x, half_size_z],
        [-half_size_x, -half_size_z],
        [half_size_x, -half_size_z],
    ])

    cos_theta = np.cos(asset_rot_angle_radians)
    sin_theta = np.sin(asset_rot_angle_radians)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta],
    ])

    rotated_corners_2d_floor = np.dot(corners_2d_floor, rotation_matrix.T)
    translated_corners_2d_floor = rotated_corners_2d_floor + np.array([bbox_position[0], bbox_position[2]])
    polygon_coords_2d_floor = [(corner[0], corner[1]) for corner in translated_corners_2d_floor]
    bbox_2d_obj = Polygon(polygon_coords_2d_floor)

    obj_height = float(bbox_size[1])
    obj_y_start = float(bbox_position[1])
    obj_y_end = float(bbox_position[1] + obj_height)

    return bbox_2d_obj, obj_height, obj_y_start, obj_y_end


def create_room_mesh(bounds_bottom: List[List[float]], bounds_top: List[List[float]], floor_plan_polygon: Polygon) -> trimesh.Trimesh:
    num_verts = len(bounds_bottom)
    all_vertices = np.array(bounds_bottom + bounds_top, dtype=float)

    _, floor_faces = trimesh.creation.triangulate_polygon(floor_plan_polygon, engine="triangle")
    idxs = []
    for i, row in enumerate(floor_faces):
        if np.any(row == num_verts):
            idxs.append(i)
    if len(idxs) > 0:
        floor_faces = np.delete(floor_faces, idxs, axis=0)

    ceiling_faces = floor_faces + num_verts

    side_faces = []
    for i in range(num_verts):
        next_i = (i + 1) % num_verts
        side_faces.append([i, next_i, i + num_verts])
        side_faces.append([next_i, next_i + num_verts, i + num_verts])
    side_faces = np.array(side_faces)

    all_faces = np.concatenate((floor_faces, ceiling_faces, side_faces), axis=0)
    room_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
    trimesh.repair.fix_normals(room_mesh)
    return room_mesh


def get_intersection_area(obj_x: Polygon, obj_y: Polygon, epsilon: float = 1e-7) -> float:
    intersection = obj_x.intersection(obj_y)
    if intersection.is_empty:
        return 0.0
    area = float(intersection.area)
    if area < epsilon:
        return 0.0
    return area


def compute_oob(obj: Dict, floor_plan_polygon: Polygon, bounds_bottom, bounds_top, epsilon: float = 1e-7) -> float:
    bbox_obj, obj_height, obj_y_start, obj_y_end = get_xz_bbox_from_obj(obj)
    intersection_area = get_intersection_area(floor_plan_polygon, bbox_obj)

    room_bottom = bounds_bottom[0][1]
    room_top = bounds_top[0][1]

    if (obj_y_start < room_bottom and obj_y_end < room_bottom) or (obj_y_start > room_top and obj_y_end > room_top):
        obj_intersection_height = 0.0
    else:
        obj_intersection_height = abs(np.clip(obj_y_end, room_bottom, room_top) - np.clip(obj_y_start, room_bottom, room_top))

    bbox_vol_total = bbox_obj.area * obj_height
    bbox_vol_inside = intersection_area * obj_intersection_height
    oob = float(bbox_vol_total - bbox_vol_inside)

    if oob < epsilon:
        return 0.0
    return oob


def compute_bbl(obj_x: Dict, obj_y: Dict, epsilon: float = 1e-7) -> float:
    bbox_obj_x, _, y_start_x, y_end_x = get_xz_bbox_from_obj(obj_x)
    bbox_obj_y, _, y_start_y, y_end_y = get_xz_bbox_from_obj(obj_y)

    intersection_area = get_intersection_area(bbox_obj_x, bbox_obj_y)
    if intersection_area == 0.0:
        return 0.0

    y_start_intersection = max(y_start_x, y_start_y)
    y_end_intersection = min(y_end_x, y_end_y)
    overlap_height = max(0.0, y_end_intersection - y_start_intersection)
    bbl = float(intersection_area * overlap_height)

    if bbl < epsilon:
        return 0.0
    return bbl


def voxelize_mesh_and_get_matrix(
    asset_mesh: trimesh.Trimesh,
    voxel_size: float,
    debug_name: str = "asset",
):
    mesh = asset_mesh.copy()

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        pass

    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass

    try:
        asset_voxels = mesh.voxelized(pitch=voxel_size)
    except Exception as e:
        raise RuntimeError(f"voxelization failed for {debug_name}: {e}") from e

    try:
        return asset_voxels.fill().matrix
    except Exception as e:
        print(
            f"[WARN] voxel fill failed for {debug_name}: {e}; "
            "fallback to surface voxelization only"
        )
        return asset_voxels.matrix


def voxelize_raw_asset(pth_voxelized_mesh: Path, obj: Dict, voxel_size: float, rotation_matrix=None):
    asset_jid = obj.get("sampled_asset_jid") if obj.get("sampled_asset_jid") is not None else obj.get("jid")
    pth_mesh = get_pth_mesh(asset_jid)
    asset_scene = trimesh.load(pth_mesh)

    if isinstance(asset_scene, trimesh.Scene):
        asset_mesh = asset_scene.to_geometry()
    else:
        asset_mesh = asset_scene

    if rotation_matrix is not None:
        asset_mesh.apply_transform(rotation_matrix)
        asset_voxel_matrix = voxelize_mesh_and_get_matrix(
            asset_mesh,
            voxel_size,
            debug_name=f"asset_jid={asset_jid}"
        )
        pth_voxelized_mesh.parent.mkdir(parents=True, exist_ok=True)
        with open(pth_voxelized_mesh, "wb") as fp:
            pickle.dump(asset_voxel_matrix, fp)
    else:
        asset_voxel_matrix = voxelize_mesh_and_get_matrix(
            asset_mesh,
            voxel_size,
            debug_name=f"asset_jid={asset_jid}"
        )

    return asset_voxel_matrix


def prepare_asset(obj: Dict, voxel_size: float):
    rotation_xyzw = np.array(obj.get("rot", [0, 0, 0, 1]), dtype=float)
    asset_rot_y_euler_angle, _ = get_y_angle_from_xyzw_quaternion(rotation_xyzw)
    asset_jid = obj.get("sampled_asset_jid") if obj.get("sampled_asset_jid") is not None else obj.get("jid")

    pth_assets_root = os.getenv("PTH_3DFUTURE_ASSETS")
    if not pth_assets_root:
        raise RuntimeError("PTH_3DFUTURE_ASSETS is not set. It is required for mesh-based OOB/MBL/PBL.")

    pth_voxelized_mesh = Path(pth_assets_root) / asset_jid / f"rot-{asset_rot_y_euler_angle}-scale-{voxel_size}.pkl"

    if pth_voxelized_mesh.is_file():
        with open(pth_voxelized_mesh, "rb") as fp:
            asset_voxel_matrix = pickle.load(fp)
    else:
        quat_wxyz = [rotation_xyzw[3], rotation_xyzw[0], rotation_xyzw[1], rotation_xyzw[2]]
        rotation_matrix = quaternion_matrix(quat_wxyz)
        asset_voxel_matrix = voxelize_raw_asset(pth_voxelized_mesh, obj, voxel_size, rotation_matrix)

    asset_pos = np.array(obj.get("pos"), dtype=float)
    asset_pos_voxels = np.floor(asset_pos / voxel_size)
    asset_start_voxels = np.array([
        asset_voxel_matrix.shape[0] // 2,
        0,
        asset_voxel_matrix.shape[2] // 2,
    ])
    asset_shift_from_origin = asset_pos_voxels - asset_start_voxels
    return asset_voxel_matrix, asset_shift_from_origin


def occupancy_overlap(voxel_matrix_a, voxel_matrix_b, offset_b):
    overlap_matrix = copy.deepcopy(voxel_matrix_a).astype(int)
    for i in range(voxel_matrix_b.shape[0]):
        for j in range(voxel_matrix_b.shape[1]):
            for k in range(voxel_matrix_b.shape[2]):
                if voxel_matrix_b[i, j, k]:
                    shifted_pos = (i + offset_b[0], j + offset_b[1], k + offset_b[2])
                    if (
                        0 <= shifted_pos[0] < overlap_matrix.shape[0]
                        and 0 <= shifted_pos[1] < overlap_matrix.shape[1]
                        and 0 <= shifted_pos[2] < overlap_matrix.shape[2]
                    ):
                        overlap_matrix[shifted_pos[0], shifted_pos[1], shifted_pos[2]] += 1
    return overlap_matrix == 2


def compute_mesh_oob(obj: Dict, voxel_size: float, room_origin_shift, room_voxel_matrix, voxel_volume: float) -> float:
    asset_voxel_matrix, asset_shift_from_origin = prepare_asset(obj, voxel_size)
    asset_offset = np.floor(room_origin_shift + asset_shift_from_origin).astype(int)

    inside_voxels = occupancy_overlap(room_voxel_matrix, asset_voxel_matrix, asset_offset)
    num_asset_voxels = int(np.sum(asset_voxel_matrix))
    num_inside_voxels = int(np.sum(inside_voxels))
    return float((num_asset_voxels - num_inside_voxels) * voxel_volume)


def compute_mesh_bbl(obj_x: Dict, obj_y: Dict, voxel_size: float, voxel_volume: float) -> float:
    asset_voxel_matrix_x, asset_shift_from_origin_x = prepare_asset(obj_x, voxel_size)
    asset_voxel_matrix_y, asset_shift_from_origin_y = prepare_asset(obj_y, voxel_size)
    inside_voxels = occupancy_overlap(
        asset_voxel_matrix_x,
        asset_voxel_matrix_y,
        np.floor(asset_shift_from_origin_y - asset_shift_from_origin_x).astype(int),
    )
    num_inside_voxels = int(np.sum(inside_voxels))
    return float(num_inside_voxels * voxel_volume)


# -----------------------------
# Semantic metrics
# -----------------------------

def compute_pms_score(prompt: Optional[str], new_obj_desc: Optional[str]) -> float:
    if prompt is None:
        return float("inf")
    if new_obj_desc is None:
        return 0.0

    prompt_words = prompt.split(" ")
    if len(prompt_words) == 0:
        return 0.0

    correct_words = 0
    new_obj_desc_lower = new_obj_desc.lower()
    for word in prompt_words:
        if word in new_obj_desc_lower:
            correct_words += 1
    return float(correct_words / len(prompt_words))


# -----------------------------
# Scene-level evaluation
# -----------------------------

def eval_bounds(scene):
    try:
        bounds_bottom = scene.get("bounds_bottom", None)
        bounds_top = scene.get("bounds_top", None)

        if bounds_bottom is None:
            raise ValueError("bounds_bottom is None")
        if bounds_top is None:
            raise ValueError("bounds_top is None")

        floor_plan_polygon = create_floor_plan_polygon(bounds_bottom)

        if floor_plan_polygon is None or floor_plan_polygon.is_empty:
            return False

        if not floor_plan_polygon.is_valid:
            floor_plan_polygon = floor_plan_polygon.buffer(0)

        if floor_plan_polygon.is_empty:
            return False

        arr_bottom = np.asarray(bounds_bottom, dtype=object)
        arr_top = np.asarray(bounds_top, dtype=object)
        if arr_bottom.ndim != 2 or arr_bottom.shape[0] < 3 or arr_bottom.shape[1] < 3:
            raise ValueError(f"bounds_bottom has invalid shape: {arr_bottom.shape}")
        if arr_top.ndim != 2 or arr_top.shape[0] < 3 or arr_top.shape[1] < 3:
            raise ValueError(f"bounds_top has invalid shape: {arr_top.shape}")

        return True

    except Exception as e:
        scene_id = scene.get("scene_id", "unknown")
        print(f"[WARN] invalid scene bounds for scene_id={scene_id}: {e}")
        return False


def eval_scene(
    scene: Dict,
    voxel_size: float = 0.05,
    total_loss_threshold: float = 0.1,
    do_pms_full_scene: bool = False,
) -> Dict:
    bounds_top = scene.get("bounds_top")
    bounds_bottom = scene.get("bounds_bottom")
    objs = scene.get("objects") or []
    scene_id = scene.get("scene_id", "unknown")

    bounds_valid = eval_bounds(scene)

    metrics = {
        "bounds_valid": bool(bounds_valid),
        "total_oob_loss": float("nan"),
        "total_mbl_loss": float("nan"),
        "total_pbl_loss": float("nan"),
        "is_valid_scene_pbl": False,
        "obj_with_highest_pbl_loss": {"idx": None, "pbl": float("nan")},
        "txt_pms_score": 0.0,
        "txt_pms_sampled_score": 0.0,
    }

    if bounds_valid:
        floor_plan_polygon = create_floor_plan_polygon(bounds_bottom)
        voxel_volume = voxel_size**3

        room_mesh = create_room_mesh(bounds_bottom, bounds_top, floor_plan_polygon)
        room_voxels = room_mesh.voxelized(pitch=voxel_size).fill()
        room_voxel_matrix = room_voxels.matrix
        room_size_voxels = np.ceil(abs(room_mesh.bounds[0] - room_mesh.bounds[1]) / voxel_size)
        room_origin_shift = np.array([room_size_voxels[0] / 2.0, 0, room_size_voxels[2] / 2.0])

        mesh_oobs: List[float] = []
        mesh_bbls: List[float] = []
        idx_highest_pbl_loss = None
        highest_pbl_loss = float("-inf")

        for i, obj_x in enumerate(objs):
            obj_pbl = 0.0

            oob = compute_oob(obj_x, floor_plan_polygon, bounds_bottom, bounds_top)
            if oob > 0.0:
                try:
                    mesh_oob = compute_mesh_oob(obj_x, voxel_size, room_origin_shift, room_voxel_matrix, voxel_volume)
                except Exception as e:
                    raise RuntimeError(
                        f"mesh OOB failed for scene_id={scene_id}, obj={obj_x.get('desc', 'unknown')}: {e}"
                    ) from e
                obj_pbl += mesh_oob
            else:
                mesh_oob = 0.0
            mesh_oobs.append(mesh_oob)

            for obj_y in objs[i + 1:]:
                bbl = compute_bbl(obj_x, obj_y)
                if bbl > 0.0:
                    try:
                        mesh_bbl = compute_mesh_bbl(obj_x, obj_y, voxel_size, voxel_volume)
                    except Exception as e:
                        raise RuntimeError(
                            f"mesh MBL failed for scene_id={scene_id}, obj_x={obj_x.get('desc', 'unknown')}, obj_y={obj_y.get('desc', 'unknown')}: {e}"
                        ) from e
                    obj_pbl += mesh_bbl
                else:
                    mesh_bbl = 0.0
                mesh_bbls.append(mesh_bbl)

            if obj_pbl > highest_pbl_loss:
                idx_highest_pbl_loss = i
                highest_pbl_loss = obj_pbl

        metrics["total_oob_loss"] = float(np.sum(mesh_oobs)) if len(mesh_oobs) > 0 else 0.0
        metrics["total_mbl_loss"] = float(np.sum(mesh_bbls)) if len(mesh_bbls) > 0 else 0.0
        metrics["total_pbl_loss"] = metrics["total_oob_loss"] + metrics["total_mbl_loss"]
        metrics["is_valid_scene_pbl"] = bool(metrics["total_pbl_loss"] <= total_loss_threshold)
        metrics["obj_with_highest_pbl_loss"] = {
            "idx": idx_highest_pbl_loss,
            "pbl": highest_pbl_loss if highest_pbl_loss != float("-inf") else 0.0,
        }
    else:
        print(f"[WARN] Invalid scene bounds, skip bound-based metrics for scene_id={scene_id}")

    if len(objs) > 0:
        all_txt_pms_scores = []
        all_txt_pms_sampled_scores = []
        objs_pms = objs if do_pms_full_scene else [objs[-1]]
        for obj in objs_pms:
            if obj.get("prompt") is not None:
                new_obj_desc = obj.get("desc")
                txt_pms_score = compute_pms_score(obj.get("prompt"), new_obj_desc)
                all_txt_pms_scores.append(txt_pms_score)

                txt_pms_score_sampled = compute_pms_score(obj.get("prompt"), obj.get("sampled_asset_desc"))
                all_txt_pms_sampled_scores.append(txt_pms_score_sampled)

        if len(all_txt_pms_scores) > 0:
            metrics["txt_pms_score"] = float(np.mean(all_txt_pms_scores))
        if len(all_txt_pms_sampled_scores) > 0:
            metrics["txt_pms_sampled_score"] = float(np.mean(all_txt_pms_sampled_scores))

    return metrics


# -----------------------------
# Dataset discovery
# -----------------------------

def dedup_paths(paths: Iterable[Path]) -> List[Path]:
    uniq: Dict[str, Path] = {}
    for p in paths:
        uniq[str(p.resolve())] = p
    return [uniq[k] for k in sorted(uniq.keys())]


def infer_scene_id_from_scene_path(scene_path: Path) -> str:
    # Layout C: .../<scene_id>/final/final/scene.json
    if (
        scene_path.name == "scene.json"
        and scene_path.parent.name == "final"
        and scene_path.parent.parent.name == "final"
    ):
        return scene_path.parent.parent.parent.name

    # Layout B: .../<scene_id>/final/scene.json
    if scene_path.name == "scene.json" and scene_path.parent.name == "final":
        return scene_path.parent.parent.name

    # Layout A: .../<scene_id>.json
    stem = scene_path.stem
    for suffix in ["_updated", "_final", "_scene"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def infer_layout_from_scene_path(scene_path: Path) -> str:
    if scene_path.name == "scene.json" and scene_path.parent.name == "final" and scene_path.parent.parent.name == "final":
        return "nested_final_final"
    if scene_path.name == "scene.json" and scene_path.parent.name == "final":
        return "nested_final"
    return "flat_json"


def discover_scene_jsons(scenes_root: Path) -> List[Path]:
    """
    Priority order:
    1) single-scene exact path
    2) strict double-final layout: root/<scene_id>/final/final/scene.json
    3) strict one-level nested-final layout: root/<scene_id>/final/scene.json
    4) flat json layout
    5) recursive fallback

    This avoids accidentally counting batch-level metadata jsons at the root when
    a per-scene nested layout is actually present.
    """
    scenes_root = resolve_scenes_root(scenes_root)

    # Case 0a: user passes a single scene directory: <scene_id>/final/scene.json
    single_scene_final = scenes_root / "final" / "scene.json"
    if single_scene_final.is_file():
        return dedup_paths([single_scene_final])

    # Case 0b: user passes a single scene directory: <scene_id>/final/final/scene.json
    single_scene_final_final = scenes_root / "final" / "final" / "scene.json"
    if single_scene_final_final.is_file():
        return dedup_paths([single_scene_final_final])

    # Case 0c: user passes the final directory itself: .../final/scene.json
    if scenes_root.name == "final":
        scene_json = scenes_root / "scene.json"
        if scene_json.is_file():
            return dedup_paths([scene_json])
        scene_json_nested = scenes_root / "final" / "scene.json"
        if scene_json_nested.is_file():
            return dedup_paths([scene_json_nested])

    # Case 1: strict double-final layout only
    double_level_finals = [p for p in scenes_root.glob("*/final/final/scene.json") if p.is_file()]
    if double_level_finals:
        return dedup_paths(double_level_finals)

    # Case 2: strict one-level nested-final layout only
    one_level_finals = [p for p in scenes_root.glob("*/final/scene.json") if p.is_file()]
    if one_level_finals:
        return dedup_paths(one_level_finals)

    # Case 3: flat layout, e.g. updated_scenes/*.json
    flat_jsons = [
        p for p in scenes_root.glob("*.json")
        if p.is_file() and p.name != "scene.json"
    ]
    if flat_jsons:
        return dedup_paths(flat_jsons)

    # Case 4: recursive fallback
    recursive_finals = [
        p for p in scenes_root.rglob("scene.json")
        if p.is_file() and p.parent.name == "final"
    ]
    return dedup_paths(recursive_finals)


def infer_scene_id_from_view_dir(view_dir: Path) -> str:
    # Layout C: .../<scene_id>/final/final/diag or .../<scene_id>/final/final/top
    if view_dir.parent.name == "final" and view_dir.parent.parent.name == "final":
        return view_dir.parent.parent.parent.name
    # Layout B: .../<scene_id>/final/diag or .../<scene_id>/final/top
    if view_dir.parent.name == "final":
        return view_dir.parent.parent.name
    # Layout A: .../<scene_id>/diag or .../<scene_id>/top
    return view_dir.parent.name


def discover_render_view_dirs(renders_root: Path, view: str) -> List[Path]:
    renders_root = resolve_renders_root(renders_root)
    candidates: List[Path] = []

    direct = renders_root / view
    if direct.is_dir():
        candidates.append(direct)

    candidates.extend([p for p in renders_root.glob(f"*/{view}") if p.is_dir()])
    candidates.extend([p for p in renders_root.glob(f"*/final/{view}") if p.is_dir()])
    candidates.extend([p for p in renders_root.glob(f"*/final/final/{view}") if p.is_dir()])

    # Recursive fallback
    candidates.extend([p for p in renders_root.rglob(view) if p.is_dir() and p.name == view])

    return dedup_paths(candidates)


# -----------------------------
# Render directory processing
# -----------------------------

def safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except Exception:
        shutil.copy2(src, dst)


def flatten_render_view(
    renders_root: Path,
    view: str,
    out_dir: Path,
    scene_id_filter: Optional[set[str]] = None,
) -> int:
    renders_root = resolve_renders_root(renders_root)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    view_dirs = discover_render_view_dirs(renders_root, view)
    for view_dir in sorted(view_dirs):
        scene_id = infer_scene_id_from_view_dir(view_dir)
        if scene_id_filter is not None and scene_id not in scene_id_filter:
            continue

        image_files = sorted([
            p for p in view_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ])
        for i, img_path in enumerate(image_files):
            dst = out_dir / f"{scene_id}__{i:03d}{img_path.suffix.lower()}"
            safe_link_or_copy(img_path, dst)
            count += 1
    return count


# -----------------------------
# Aggregation helpers
# -----------------------------

def summarize_numeric(df: pd.DataFrame, col: str) -> Dict[str, float]:
    if col not in df.columns or len(df) == 0:
        return {}
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(values) == 0:
        return {}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def compute_scene_table(
    scenes_root: Path,
    renders_root: Optional[Path],
    voxel_size: float,
    total_loss_threshold: float,
    do_pms_full_scene: bool,
    enable_gpt_judge: bool = False,
    yunwu_api_key: Optional[str] = None,
    yunwu_base_url: str = "https://yunwu.ai/v1",
    yunwu_model: str = "gpt-5.4",
    gpt_judge_timeout: int = 180,
    gpt_judge_max_images_per_view: int = 2,
) -> pd.DataFrame:
    rows = []

    scenes_root = resolve_scenes_root(scenes_root)
    resolved_renders_root = resolve_renders_root(renders_root) if renders_root is not None else None

    print(f"[INFO] resolved scenes_root  = {scenes_root}")
    print(f"[INFO] resolved renders_root = {resolved_renders_root}")

    scene_files = discover_scene_jsons(scenes_root)
    if len(scene_files) == 0:
        raise FileNotFoundError(
            "No scene jsons found after resolving scenes_root.\n"
            f"Resolved scenes_root: {scenes_root}"
        )

    for scene_path in tqdm(scene_files, desc="Evaluating scenes"):
        scene_id = infer_scene_id_from_scene_path(scene_path)

        try:
            with open(scene_path, "r", encoding="utf-8") as f:
                scene = json.load(f)

            if not isinstance(scene, dict):
                print(f"[WARN] Skip non-dict JSON file: {scene_path}")
                continue

            if not _looks_like_scene_dict(scene):
                print(f"[WARN] Skip non-scene JSON file: {scene_path}")
                continue

            scene["scene_id"] = scene.get("scene_id") or scene_id
            scene["__scene_path"] = str(scene_path)

            metrics = eval_scene(
                scene,
                voxel_size=voxel_size,
                total_loss_threshold=total_loss_threshold,
                do_pms_full_scene=do_pms_full_scene,
            )

            row = {
                "scene_file": scene_path.name,
                "scene_path": str(scene_path),
                "scene_id": scene_id,
                "input_layout": infer_layout_from_scene_path(scene_path),
                "room_type": scene.get("room_type", None),
                "n_objects": len(scene.get("objects", []) or []),
                "bounds_valid": metrics.get("bounds_valid", False),
                "total_oob_loss": metrics.get("total_oob_loss", np.nan),
                "total_mbl_loss": metrics.get("total_mbl_loss", np.nan),
                "total_pbl_loss": metrics.get("total_pbl_loss", np.nan),
                "is_valid_scene_pbl": metrics.get("is_valid_scene_pbl", False),
                "txt_pms_score": metrics.get("txt_pms_score", np.nan),
                "txt_pms_sampled_score": metrics.get("txt_pms_sampled_score", np.nan),
                "highest_pbl_obj_idx": (metrics.get("obj_with_highest_pbl_loss") or {}).get("idx"),
                "highest_pbl_obj_loss": (metrics.get("obj_with_highest_pbl_loss") or {}).get("pbl", np.nan),
            }

            if enable_gpt_judge:
                try:
                    if resolved_renders_root is None:
                        raise ValueError("renders_root is required when enable_gpt_judge=True")

                    render_images = collect_scene_render_images(
                        renders_root=resolved_renders_root,
                        scene_id=scene_id,
                        max_images_per_view=gpt_judge_max_images_per_view,
                    )

                    row["gpt_judge_n_diag_images"] = len(render_images["diag"])
                    row["gpt_judge_n_top_images"] = len(render_images["top"])

                    if len(render_images["diag"]) == 0 and len(render_images["top"]) == 0:
                        raise ValueError(
                            f"No render images found for GPT judge. "
                            f"scene_id={scene_id}, renders_root={resolved_renders_root}"
                        )

                    judge = call_yunwu_multiview_scene_judge(
                        diag_images=render_images["diag"],
                        top_images=render_images["top"],
                        room_type=scene.get("room_type"),
                        scene_prompt=extract_scene_prompt(scene),
                        scene_id=scene_id,
                        api_key=yunwu_api_key or os.getenv("YUNWU_API_KEY", ""),
                        base_url=yunwu_base_url,
                        model=yunwu_model,
                        timeout=gpt_judge_timeout,
                    )
                    row.update({
                        "gpt_lc_score": judge["lc"]["score"],
                        "gpt_spa_score": judge["spa"]["score"],
                        "gpt_fc_score": judge["fc"]["score"],
                        "gpt_overall_score": judge["overall"],
                        "gpt_lc_reason": judge["lc"]["reason"],
                        "gpt_spa_reason": judge["spa"]["reason"],
                        "gpt_fc_reason": judge["fc"]["reason"],
                        "gpt_judge_model": judge.get("model"),
                        "gpt_judge_error": "",
                    })
                except Exception as e:
                    row.update({
                        "gpt_lc_score": np.nan,
                        "gpt_spa_score": np.nan,
                        "gpt_fc_score": np.nan,
                        "gpt_overall_score": np.nan,
                        "gpt_lc_reason": "",
                        "gpt_spa_reason": "",
                        "gpt_fc_reason": "",
                        "gpt_judge_model": yunwu_model,
                        "gpt_judge_n_diag_images": row.get("gpt_judge_n_diag_images", 0),
                        "gpt_judge_n_top_images": row.get("gpt_judge_n_top_images", 0),
                        "gpt_judge_error": str(e),
                    })

            rows.append(row)

        except Exception as e:
            print(f"[WARN] Skip bad scene: scene_id={scene_id}, scene_path={scene_path}, error={e}")
            continue

    if len(rows) == 0:
        raise RuntimeError("No valid scenes were evaluated. All scenes were skipped.")

    return pd.DataFrame(rows)


def add_image_metrics(
    summary: Dict,
    flat_diag_dir: Path,
    flat_top_dir: Path,
    ref_diag_dir: Optional[Path],
    ref_top_dir: Optional[Path],
    fid_name_prefix: str,
    dataset_res: int,
    device_for_diversity: str,
) -> Dict:
    image_metrics = {}

    if ref_diag_dir is not None and ref_diag_dir.is_dir() and flat_diag_dir.is_dir():
        try:
            compute_fid_scores(
                "diag",
                fid_score_name=f"{fid_name_prefix}-diag",
                pth_src=str(ref_diag_dir),
                pth_gen=str(flat_diag_dir),
                aggregated_metrics=image_metrics,
                do_renderings=False,
                dataset_res=dataset_res,
            )
        except Exception as e:
            image_metrics["diag_metrics_error"] = str(e)

    if ref_top_dir is not None and ref_top_dir.is_dir() and flat_top_dir.is_dir():
        try:
            compute_fid_scores(
                "top",
                fid_score_name=f"{fid_name_prefix}-top",
                pth_src=str(ref_top_dir),
                pth_gen=str(flat_top_dir),
                aggregated_metrics=image_metrics,
                do_renderings=False,
                dataset_res=dataset_res,
            )
        except Exception as e:
            image_metrics["top_metrics_error"] = str(e)

    if flat_diag_dir.is_dir():
        try:
            compute_diversity_score(
                "diag",
                pth_gen=str(flat_diag_dir),
                do_renderings=False,
                dvc=device_for_diversity,
                aggregated_metrics=image_metrics,
            )
        except Exception as e:
            image_metrics["diversity_diag_error"] = str(e)

    if flat_top_dir.is_dir():
        try:
            compute_diversity_score(
                "top",
                pth_gen=str(flat_top_dir),
                do_renderings=False,
                dvc=device_for_diversity,
                aggregated_metrics=image_metrics,
            )
        except Exception as e:
            image_metrics["diversity_top_error"] = str(e)

    summary["image_metrics"] = image_metrics
    return summary


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate generated scene jsons and existing renders.")
    parser.add_argument(
        "--scenes-root",
        type=str,
        required=True,
        help=(
            "Can be:\n"
            "1) updated_scenes root\n"
            "2) nested-final root\n"
            "3) whole batch root"
        ),
    )
    parser.add_argument(
        "--renders-root",
        type=str,
        required=True,
        help=(
            "Can be:\n"
            "1) renders root\n"
            "2) nested-final root\n"
            "3) whole batch root"
        ),
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save flattened render caches and metric files.")
    parser.add_argument("--env-file", type=str, default=None, help="Optional .env file to load before evaluation.")

    parser.add_argument("--ref-diag-dir", type=str, default=None, help="Reference diag render directory for FID / CLIP-FID / KID.")
    parser.add_argument("--ref-top-dir", type=str, default=None, help="Reference top render directory for FID / CLIP-FID / KID.")
    parser.add_argument("--fid-name-prefix", type=str, default="batch-scene-eval")
    parser.add_argument("--dataset-res", type=int, default=1024)
    parser.add_argument("--device-for-diversity", type=str, default="cuda")

    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--total-loss-threshold", type=float, default=0.1)
    parser.add_argument("--do-pms-full-scene", action="store_true", default=False)
    parser.add_argument("--skip-image-metrics", action="store_true", default=False)

    parser.add_argument("--enable-gpt-judge", action="store_true", default=False, help="Use Yunwu OpenAI-compatible API to score each scene from diag/top render images.")
    parser.add_argument("--yunwu-api-key", type=str, default=None, help="Yunwu API key. If omitted, the script uses env var YUNWU_API_KEY.")
    parser.add_argument("--yunwu-base-url", type=str, default="https://yunwu.ai/v1")
    parser.add_argument("--yunwu-model", type=str, default="gpt-5.4", help="Judge model name on Yunwu.")
    parser.add_argument("--gpt-judge-timeout", type=int, default=180)
    parser.add_argument("--gpt-judge-max-images-per-view", type=int, default=2)

    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file)

    scenes_root_input = Path(args.scenes_root)
    renders_root_input = Path(args.renders_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_scenes_root = resolve_scenes_root(scenes_root_input)
    resolved_renders_root = resolve_renders_root(renders_root_input)

    print(f"[INFO] input scenes_root    = {scenes_root_input}")
    print(f"[INFO] input renders_root   = {renders_root_input}")
    print(f"[INFO] resolved scenes_root = {resolved_scenes_root}")
    print(f"[INFO] resolved renders_root= {resolved_renders_root}")
    print(f"[INFO] output_dir          = {output_dir}")

    scene_df = compute_scene_table(
        scenes_root=resolved_scenes_root,
        renders_root=resolved_renders_root,
        voxel_size=args.voxel_size,
        total_loss_threshold=args.total_loss_threshold,
        do_pms_full_scene=args.do_pms_full_scene,
        enable_gpt_judge=args.enable_gpt_judge,
        yunwu_api_key=args.yunwu_api_key,
        yunwu_base_url=args.yunwu_base_url,
        yunwu_model=args.yunwu_model,
        gpt_judge_timeout=args.gpt_judge_timeout,
        gpt_judge_max_images_per_view=args.gpt_judge_max_images_per_view,
    )

    scene_csv = output_dir / "scene_metrics.csv"
    scene_json = output_dir / "scene_metrics.json"
    scene_df.to_csv(scene_csv, index=False)
    scene_df.to_json(scene_json, orient="records", indent=2, force_ascii=False)

    summary = {
        "n_scenes": int(len(scene_df)),
        "room_type_counts": scene_df["room_type"].value_counts(dropna=False).to_dict(),
        "input_layout_counts": scene_df["input_layout"].value_counts(dropna=False).to_dict(),
        "physical_metrics": {
            "total_oob_loss": summarize_numeric(scene_df, "total_oob_loss"),
            "total_mbl_loss": summarize_numeric(scene_df, "total_mbl_loss"),
            "total_pbl_loss": summarize_numeric(scene_df, "total_pbl_loss"),
            "valid_scene_ratio_pbl": float(scene_df["is_valid_scene_pbl"].mean()) if len(scene_df) > 0 else 0.0,
        },
        "semantic_metrics": {
            "txt_pms_score": summarize_numeric(scene_df, "txt_pms_score"),
            "txt_pms_sampled_score": summarize_numeric(scene_df, "txt_pms_sampled_score"),
        },
    }

    if "gpt_overall_score" in scene_df.columns:
        summary["gpt_judge_metrics"] = {
            "model": args.yunwu_model,
            "n_success": int(pd.to_numeric(scene_df["gpt_overall_score"], errors="coerce").notna().sum()),
            "n_failed": int((scene_df.get("gpt_judge_error", pd.Series(dtype=object)).fillna("") != "").sum())
            if "gpt_judge_error" in scene_df.columns else 0,
            "lc": summarize_numeric(scene_df, "gpt_lc_score"),
            "spa": summarize_numeric(scene_df, "gpt_spa_score"),
            "fc": summarize_numeric(scene_df, "gpt_fc_score"),
            "overall": summarize_numeric(scene_df, "gpt_overall_score"),
        }

    if not args.skip_image_metrics:
        flat_diag_dir = output_dir / "flat_renders" / "diag"
        flat_top_dir = output_dir / "flat_renders" / "top"
        scene_id_filter = set(scene_df["scene_id"].astype(str).tolist())

        n_diag = flatten_render_view(resolved_renders_root, "diag", flat_diag_dir, scene_id_filter=scene_id_filter)
        n_top = flatten_render_view(resolved_renders_root, "top", flat_top_dir, scene_id_filter=scene_id_filter)
        summary["flat_render_counts"] = {"diag": int(n_diag), "top": int(n_top)}

        summary = add_image_metrics(
            summary=summary,
            flat_diag_dir=flat_diag_dir,
            flat_top_dir=flat_top_dir,
            ref_diag_dir=Path(args.ref_diag_dir) if args.ref_diag_dir else None,
            ref_top_dir=Path(args.ref_top_dir) if args.ref_top_dir else None,
            fid_name_prefix=args.fid_name_prefix,
            dataset_res=args.dataset_res,
            device_for_diversity=args.device_for_diversity,
        )

    summary_path = output_dir / "summary_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Per-scene metrics CSV : {scene_csv}")
    print(f"Per-scene metrics JSON: {scene_json}")
    print(f"Summary JSON          : {summary_path}")


if __name__ == "__main__":
    main()


# Example:
# export PTH_3DFUTURE_ASSETS=/home2/zhangjiawei/respace/dataset/3D-FUTURE-model/
# export YUNWU_API_KEY="sk-3PxZML90syfHtBF9PP6gFdG0GGwyUV97hJZ6iIKTwApAvwib"
# #
# baseline:
# python /home2/zhangjiawei/respace/eval_batch_scene_metrices.py \
#   --scenes-root /home2/zhangjiawei/respace/results/batch_outputs_baseline_123 \
#   --renders-root /home2/zhangjiawei/respace/results/batch_outputs_baseline_123 \
#   --output-dir /home2/zhangjiawei/respace/results/batch_outputs_baseline_123/eval_metrics \
#   --enable-gpt-judge \
#   --yunwu-model gpt-5.4
#
# nested-final:
# python /home2/zhangjiawei/respace/eval_batch_scene_metrices.py \
#   --scenes-root /home2/zhangjiawei/respace/results/ablation_rag_group_vlm_qwen3-sft+grpo \
#   --renders-root /home2/zhangjiawei/respace/results/ablation_rag_group_vlm_qwen3-sft+grpo \
#   --output-dir /home2/zhangjiawei/respace/results/ablation_rag_group_vlm_qwen3-sft+grpo/eval_metrics \
#   --enable-gpt-judge \
#   --yunwu-model gpt-5.4