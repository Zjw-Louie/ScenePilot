from __future__ import annotations

import copy
import json
import math
import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

from src.eval import compute_oob, create_floor_plan_polygon, eval_scene, get_xz_bbox_from_obj


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _norm_text(text: Any) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _singularize_object_prompt(text: str) -> str:
    s = _norm_text(text)
    s = re.sub(r"^(a|an|the)\s+", "", s)
    s = re.sub(r"\b(set of|pair of|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _group_sort_key(group_item: Dict[str, Any]) -> Tuple[int, int]:
    priority = int(group_item.get("priority", 999))
    n_obj = len(group_item.get("objects", []))
    return (priority, -n_obj)


def _group_name_slug(name: str) -> str:
    s = _norm_text(name)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "group"


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _strip_planning_rag_block_for_terminal(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return s
    marker = "Now perform scene planning for the following request:"
    if marker in s:
        tail = s.split(marker, 1)[1].strip()
        return tail or s
    rag_header = "RETRIEVED GROUP PRIORS FOR PLANNING:"
    if s.startswith(rag_header):
        lines = s.splitlines()
        kept = []
        in_rag_block = True
        for line in lines:
            if in_rag_block:
                if not line.strip():
                    continue
                if line.lstrip().lower().startswith((
                    "add ", "create ", "generate ", "design ", "place ", "make ", "build ",
                    "a ", "an ", "the ", "this ", "that ", "room ", "scene ",
                )):
                    in_rag_block = False
                    kept.append(line)
            else:
                kept.append(line)
        compact = "\n".join(kept).strip()
        if compact:
            return compact
    return s


def _terminal_prompt_preview(add_prompt: str, max_len: int = 600) -> str:
    s = _strip_planning_rag_block_for_terminal(add_prompt)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _compute_group_progress_metrics(scene: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight metrics for group-stage logging.
    Only uses eval_scene so we avoid importing the heavy v13 optimizer module here.
    """
    try:
        metrics = eval_scene(scene, is_debug=False)
        return {
            "num_objects": len(scene.get("objects", [])),
            "pbl": round(_safe_float(metrics.get("total_pbl_loss")), 6),
            "oob": round(_safe_float(metrics.get("total_oob_loss")), 6),
            "mbl": round(_safe_float(metrics.get("total_mbl_loss")), 6),
        }
    except Exception as exc:
        return {
            "num_objects": len(scene.get("objects", [])),
            "metric_error": str(exc),
        }


def _format_group_progress_metrics(metrics: Dict[str, Any]) -> str:
    if not isinstance(metrics, dict):
        return "metrics=<invalid>"
    if "metric_error" in metrics:
        return (
            f"objects={metrics.get('num_objects', '?')} "
            f"metric_error={metrics.get('metric_error')}"
        )
    return (
        f"objects={metrics.get('num_objects', '?')} "
        f"pbl={metrics.get('pbl', 0.0):.6f} "
        f"oob={metrics.get('oob', 0.0):.6f} "
        f"mbl={metrics.get('mbl', 0.0):.6f}"
    )


def _metrics_compare_ready(metrics: Dict[str, Any]) -> bool:
    return isinstance(metrics, dict) and "metric_error" not in metrics


def _is_strictly_better_group_metrics(
    before: Dict[str, Any],
    after: Dict[str, Any],
    eps: float = 1e-9,
) -> bool:
    """
    Accept after only if:
    - pbl/oob/mbl are all no worse than before
    - and at least one of them is strictly better
    """
    if not (_metrics_compare_ready(before) and _metrics_compare_ready(after)):
        return False

    before_pbl = _safe_float(before.get("pbl"))
    before_oob = _safe_float(before.get("oob"))
    before_mbl = _safe_float(before.get("mbl"))

    after_pbl = _safe_float(after.get("pbl"))
    after_oob = _safe_float(after.get("oob"))
    after_mbl = _safe_float(after.get("mbl"))

    no_worse = (
        after_pbl <= before_pbl + eps
        and after_oob <= before_oob + eps
        and after_mbl <= before_mbl + eps
    )
    strictly_better = (
        after_pbl < before_pbl - eps
        or after_oob < before_oob - eps
        or after_mbl < before_mbl - eps
    )
    return no_worse and strictly_better


def _build_group_plan_system_prompt(max_groups: int = 4) -> str:
    return f"""you are a world-class interior scene planner.

# input
- <prompt>: user request
- <scenegraph>: current scene json (may be empty except room bounds)

# task
decompose the requested scene into FUNCTIONAL GROUPS for group-wise scene generation.

# important
- a group should correspond to a functional zone, not a single random category bucket.
- each group must have one anchor_object, usually the dominant object of that zone.
- objects must remain atomic: one string = one physical object.
- if two identical objects are needed, repeat the same string twice.
- assign a priority. lower priority number means earlier generation.
- keep the number of groups small. Prefer 2-4 groups, and do not exceed {max_groups} groups unless absolutely unavoidable.
- assign a zone_hint from:
  ["against_wall", "near_wall", "center", "corner", "near_window", "open_area"]

# output
output ONLY valid json:
{{
  "groups": [
    {{
      "group_name": "sleeping",
      "anchor_object": "queen bed",
      "objects": ["queen bed", "nightstand", "nightstand", "table lamp", "rug"],
      "zone_hint": "against_wall",
      "priority": 1
    }}
  ]
}}

# rules
- no markdown
- no explanations
- no coordinates
- no object ids
- objects must be short singular noun phrases
"""


def _merge_overflow_groups(groups: List[Dict[str, Any]], max_groups: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if max_groups <= 0:
        return [], {
            "merged": False,
            "max_groups": max_groups,
            "num_groups_before_cap": len(groups),
            "num_groups_after_cap": 0,
            "overflow_group_names": [],
        }
    if len(groups) <= max_groups:
        return groups, {
            "merged": False,
            "max_groups": max_groups,
            "num_groups_before_cap": len(groups),
            "num_groups_after_cap": len(groups),
            "overflow_group_names": [],
        }
    kept = copy.deepcopy(groups[:max_groups])
    overflow = groups[max_groups - 1:]
    base = kept[-1]
    merged_names = [str(g.get("group_name", "")) for g in overflow[1:]]
    for extra in overflow[1:]:
        for obj in extra.get("objects", []):
            if isinstance(obj, str) and obj.strip():
                base["objects"].append(_singularize_object_prompt(obj))
    anchor = _singularize_object_prompt(base.get("anchor_object", ""))
    base["objects"] = [anchor] + [o for o in base["objects"] if _singularize_object_prompt(o) != anchor]
    if merged_names:
        base["group_name"] = f"{base['group_name']}_merged"
        base["merged_from"] = merged_names
    return kept, {
        "merged": True,
        "max_groups": max_groups,
        "num_groups_before_cap": len(groups),
        "num_groups_after_cap": len(kept),
        "overflow_group_names": merged_names,
        "merge_target_group_name": base.get("group_name", ""),
    }


def _normalize_group_plan(payload: Optional[Dict[str, Any]], max_groups: int = 4) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"groups": []}
    groups = payload.get("groups")
    if not isinstance(groups, list):
        return {"groups": []}
    normalized: List[Dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_name = _norm_text(group.get("group_name", ""))
        anchor_object = _singularize_object_prompt(group.get("anchor_object", ""))
        objects_raw = group.get("objects", [])
        if not group_name or not anchor_object or not isinstance(objects_raw, list):
            continue
        objects: List[str] = []
        for obj in objects_raw:
            obj_norm = _singularize_object_prompt(obj)
            if obj_norm:
                objects.append(obj_norm)
        if not objects:
            continue
        if anchor_object not in objects:
            objects.insert(0, anchor_object)
        zone_hint = _norm_text(group.get("zone_hint", "against_wall"))
        if zone_hint not in {"against_wall", "near_wall", "center", "corner", "near_window", "open_area"}:
            zone_hint = "against_wall"
        try:
            priority = int(group.get("priority", len(normalized) + 1))
        except Exception:
            priority = len(normalized) + 1
        normalized.append(
            {
                "group_name": group_name,
                "anchor_object": anchor_object,
                "objects": objects,
                "zone_hint": zone_hint,
                "priority": priority,
            }
        )
    normalized.sort(key=_group_sort_key)
    merged_groups, merge_meta = _merge_overflow_groups(normalized, max_groups=max_groups)
    return {
        "groups": merged_groups,
        "max_groups": int(max_groups),
        "num_groups_before_cap": len(normalized),
        "num_groups_after_cap": len(merged_groups),
        "overflow_merge_meta": merge_meta,
    }


# ------------------------------
# simple deterministic group-local optimizer
# rules used:
# 1) anchor snap / yaw align
# 3) collision push
# 4) out-of-bounds inward projection
# ------------------------------

def _normalize_angle(deg: float) -> float:
    return deg % 360.0


def _quaternion_from_yaw(yaw_deg: float) -> List[float]:
    yaw_rad = math.radians(yaw_deg)
    return [0.0, math.sin(yaw_rad / 2.0), 0.0, math.cos(yaw_rad / 2.0)]


def _yaw_from_quaternion(q: List[float]) -> float:
    if not isinstance(q, (list, tuple)) or len(q) != 4:
        return 0.0
    x, y, z, w = q
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (y * y + x * x)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def _get_floor_polygon(scene: Dict[str, Any]) -> Optional[ShapelyPolygon]:
    bounds_bottom = scene.get("bounds_bottom")
    if not isinstance(bounds_bottom, list) or len(bounds_bottom) < 3:
        return None
    try:
        return create_floor_plan_polygon(bounds_bottom)
    except Exception:
        return None


def _room_extents_xz(scene: Dict[str, Any]) -> Tuple[float, float, float, float]:
    pts = [(float(p[0]), float(p[2])) for p in scene.get("bounds_bottom", []) if isinstance(p, list) and len(p) >= 3]
    if not pts:
        return -1.0, 1.0, -1.0, 1.0
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    return min(xs), max(xs), min(zs), max(zs)


def _room_center_xz(scene: Dict[str, Any]) -> Tuple[float, float]:
    poly = _get_floor_polygon(scene)
    if poly is not None and not poly.is_empty:
        c = poly.centroid
        return float(c.x), float(c.y)
    minx, maxx, minz, maxz = _room_extents_xz(scene)
    return 0.5 * (minx + maxx), 0.5 * (minz + maxz)


def _object_diag_size_xz(obj: Dict[str, Any]) -> float:
    size = obj.get("size") or obj.get("sampled_asset_size") or [0.6, 0.6, 0.6]
    if not isinstance(size, (list, tuple)) or len(size) < 3:
        return 1.0
    sx = float(size[0])
    sz = float(size[2])
    return max(0.2, math.hypot(sx, sz))


def _forward_vec_from_yaw(yaw_deg: float) -> Tuple[float, float]:
    rad = math.radians(yaw_deg)
    return math.sin(rad), math.cos(rad)


def _nearest_wall_info(scene: Dict[str, Any], pos: List[float]) -> Optional[Dict[str, float]]:
    pts = [(float(p[0]), float(p[2])) for p in scene.get("bounds_bottom", []) if isinstance(p, list) and len(p) >= 3]
    if len(pts) < 3:
        return None
    center_x, center_z = _room_center_xz(scene)
    px, pz = float(pos[0]), float(pos[2])
    best = None
    best_dist = float("inf")
    for i in range(len(pts)):
        ax, az = pts[i]
        bx, bz = pts[(i + 1) % len(pts)]
        abx, abz = bx - ax, bz - az
        apx, apz = px - ax, pz - az
        denom = abx * abx + abz * abz
        if denom < 1e-12:
            continue
        t = max(0.0, min(1.0, (apx * abx + apz * abz) / denom))
        proj_x = ax + t * abx
        proj_z = az + t * abz
        dist = math.hypot(px - proj_x, pz - proj_z)
        if dist >= best_dist:
            continue
        nx, nz = -abz, abx
        nl = math.hypot(nx, nz)
        if nl < 1e-9:
            continue
        nx, nz = nx / nl, nz / nl
        if nx * (center_x - proj_x) + nz * (center_z - proj_z) < 0:
            nx, nz = -nx, -nz
        wall_yaw = _normalize_angle(math.degrees(math.atan2(nx, nz)))
        best_dist = dist
        best = {
            "proj_x": proj_x,
            "proj_z": proj_z,
            "normal_x": nx,
            "normal_z": nz,
            "wall_yaw": wall_yaw,
            "dist": dist,
        }
    return best


def _nearest_parallel_wall_yaw(scene: Dict[str, Any], pos: List[float], current_yaw: Optional[float] = None) -> Optional[float]:
    info = _nearest_wall_info(scene, pos)
    if info is None:
        return None
    wall_yaw = info["wall_yaw"]
    options = [_normalize_angle(wall_yaw + 90.0), _normalize_angle(wall_yaw + 270.0)]
    if current_yaw is None:
        return options[0]
    return min(
        options,
        key=lambda y: min(
            abs((_normalize_angle(y) - _normalize_angle(current_yaw)) % 360.0),
            abs((_normalize_angle(current_yaw) - _normalize_angle(y)) % 360.0),
        ),
    )


def _nearest_normal_axis_yaw(scene: Dict[str, Any], pos: List[float], current_yaw: Optional[float] = None) -> Optional[float]:
    info = _nearest_wall_info(scene, pos)
    if info is None:
        return None
    wall_yaw = info["wall_yaw"]
    options = [_normalize_angle(wall_yaw), _normalize_angle(wall_yaw + 180.0)]
    if current_yaw is None:
        return options[0]
    return min(
        options,
        key=lambda y: min(
            abs((_normalize_angle(y) - _normalize_angle(current_yaw)) % 360.0),
            abs((_normalize_angle(current_yaw) - _normalize_angle(y)) % 360.0),
        ),
    )


def _wall_affine_anchor(anchor_text: str, zone_hint: str) -> bool:
    t = _norm_text(anchor_text)
    if zone_hint in {"against_wall", "near_wall", "near_window", "corner"}:
        return True
    keywords = [
        "bed", "desk", "tv stand", "cabinet", "wardrobe", "dresser", "sideboard",
        "console", "counter", "vanity", "shelf", "bookcase", "refrigerator",
        "washing machine", "dryer", "sink",
    ]
    return any(k in t for k in keywords)


def _parallel_wall_anchor(anchor_text: str) -> bool:
    t = _norm_text(anchor_text)
    keywords = [
        "desk", "table", "tv stand", "cabinet", "wardrobe", "dresser", "sideboard",
        "console", "counter", "vanity", "shelf", "bookcase", "refrigerator",
        "washing machine", "dryer",
    ]
    return any(k in t for k in keywords)


def _normal_wall_anchor(anchor_text: str) -> bool:
    t = _norm_text(anchor_text)
    return "bed" in t


def _nearest_corner_target(scene: Dict[str, Any], pos: List[float], inset_ratio: float = 0.12) -> Tuple[float, float]:
    minx, maxx, minz, maxz = _room_extents_xz(scene)
    inset_x = max(0.08, (maxx - minx) * inset_ratio)
    inset_z = max(0.08, (maxz - minz) * inset_ratio)
    corners = [
        (minx + inset_x, minz + inset_z),
        (minx + inset_x, maxz - inset_z),
        (maxx - inset_x, minz + inset_z),
        (maxx - inset_x, maxz - inset_z),
    ]
    px, pz = float(pos[0]), float(pos[2])
    return min(corners, key=lambda c: math.hypot(px - c[0], pz - c[1]))


def _bbox_intersection_area(scene: Dict[str, Any], idx_a: int, idx_b: int) -> float:
    try:
        poly_a, _, ya0, ya1 = get_xz_bbox_from_obj(scene["objects"][idx_a])
        poly_b, _, yb0, yb1 = get_xz_bbox_from_obj(scene["objects"][idx_b])
    except Exception:
        return 0.0
    y_overlap = max(0.0, min(ya1, yb1) - max(ya0, yb0))
    if y_overlap <= 0:
        return 0.0
    inter = poly_a.intersection(poly_b)
    if inter.is_empty:
        return 0.0
    return float(inter.area)


def _collision_neighbors(scene: Dict[str, Any], idx: int) -> List[Tuple[int, float, float, float]]:
    objs = scene.get("objects", [])
    if not (0 <= idx < len(objs)):
        return []
    pa = objs[idx].get("pos", [0.0, 0.0, 0.0])
    results: List[Tuple[int, float, float, float]] = []
    for j in range(len(objs)):
        if j == idx:
            continue
        inter_area = _bbox_intersection_area(scene, idx, j)
        if inter_area <= 1e-8:
            continue
        pb = objs[j].get("pos", [0.0, 0.0, 0.0])
        dx, dz = float(pa[0]) - float(pb[0]), float(pa[2]) - float(pb[2])
        norm = math.hypot(dx, dz)
        if norm < 1e-9:
            dx, dz = 1.0, 0.0
        else:
            dx, dz = dx / norm, dz / norm
        results.append((j, dx, dz, inter_area))
    return results


def _set_object_xz(scene: Dict[str, Any], idx: int, x: float, z: float) -> None:
    pos = list(scene["objects"][idx].get("pos", [0.0, 0.0, 0.0]))
    pos[0] = float(x)
    pos[2] = float(z)
    scene["objects"][idx]["pos"] = pos


def _move_object_xz(scene: Dict[str, Any], idx: int, dx: float, dz: float) -> None:
    pos = list(scene["objects"][idx].get("pos", [0.0, 0.0, 0.0]))
    pos[0] += float(dx)
    pos[2] += float(dz)
    scene["objects"][idx]["pos"] = pos


def _set_object_yaw(scene: Dict[str, Any], idx: int, yaw_deg: float) -> None:
    scene["objects"][idx]["rot"] = _quaternion_from_yaw(_normalize_angle(float(yaw_deg)))


def _project_object_inside_room(scene: Dict[str, Any], idx: int, inward_step_scale: float = 0.18, max_iters: int = 6) -> float:
    floor_polygon = _get_floor_polygon(scene)
    if floor_polygon is None:
        return 0.0
    obj = scene["objects"][idx]
    last_oob = 0.0
    for _ in range(max_iters):
        last_oob = float(
            compute_oob(
                obj,
                floor_polygon,
                scene.get("bounds_bottom", []),
                scene.get("bounds_top", []),
                is_debug=False,
            )
        )
        if last_oob <= 1e-8:
            break
        pos = obj.get("pos", [0.0, 0.0, 0.0])
        point = ShapelyPoint(float(pos[0]), float(pos[2]))
        if floor_polygon.contains(point):
            centroid = floor_polygon.centroid
            dx, dz = centroid.x - float(pos[0]), centroid.y - float(pos[2])
        else:
            nearest = floor_polygon.exterior.interpolate(floor_polygon.exterior.project(point))
            dx, dz = nearest.x - float(pos[0]), nearest.y - float(pos[2])
        norm = math.hypot(dx, dz)
        if norm < 1e-9:
            break
        step = min(0.28, max(0.04, inward_step_scale * max(1.0, math.sqrt(last_oob + 1e-8))))
        _move_object_xz(scene, idx, dx / norm * step, dz / norm * step)
        obj = scene["objects"][idx]
    return last_oob


def _push_object_from_collisions(scene: Dict[str, Any], idx: int, movable_set: set[int], max_step: float = 0.20) -> float:
    if idx not in movable_set:
        return 0.0
    cols = _collision_neighbors(scene, idx)
    if not cols:
        return 0.0
    total_dx = 0.0
    total_dz = 0.0
    total_area = 0.0
    for _, dx, dz, area in cols:
        total_area += area
        weight = min(1.0, max(0.15, math.sqrt(area + 1e-8)))
        total_dx += dx * weight
        total_dz += dz * weight
    norm = math.hypot(total_dx, total_dz)
    if norm < 1e-9:
        return total_area
    step = min(max_step, max(0.03, 0.10 * math.sqrt(total_area + 1e-8)))
    _move_object_xz(scene, idx, total_dx / norm * step, total_dz / norm * step)
    return total_area


def _snap_anchor(scene: Dict[str, Any], anchor_idx: int, group_item: Dict[str, Any]) -> Dict[str, Any]:
    obj = scene["objects"][anchor_idx]
    anchor_text = str(group_item.get("anchor_object", ""))
    zone_hint = _norm_text(group_item.get("zone_hint", "against_wall"))
    pos = obj.get("pos", [0.0, 0.0, 0.0])
    moved = {
        "before_pos": list(pos),
        "after_pos": list(pos),
        "before_yaw": _yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0])),
        "after_yaw": None,
    }
    current_yaw = moved["before_yaw"]

    info = _nearest_wall_info(scene, pos)
    target_x = float(pos[0])
    target_z = float(pos[2])

    if zone_hint in {"center", "open_area"}:
        cx, cz = _room_center_xz(scene)
        target_x = 0.5 * target_x + 0.5 * cx
        target_z = 0.5 * target_z + 0.5 * cz
    elif zone_hint == "corner":
        tx, tz = _nearest_corner_target(scene, pos)
        target_x = tx
        target_z = tz
    elif info is not None and _wall_affine_anchor(anchor_text, zone_hint):
        margin = min(0.35, max(0.10, 0.12 * _object_diag_size_xz(obj)))
        target_x = float(info["proj_x"] + info["normal_x"] * margin)
        target_z = float(info["proj_z"] + info["normal_z"] * margin)
    elif zone_hint == "near_window" and info is not None:
        margin = 0.18
        target_x = float(info["proj_x"] + info["normal_x"] * margin)
        target_z = float(info["proj_z"] + info["normal_z"] * margin)

    _set_object_xz(scene, anchor_idx, target_x, target_z)

    yaw_target = None
    pos_after = scene["objects"][anchor_idx].get("pos", [0.0, 0.0, 0.0])
    if _normal_wall_anchor(anchor_text):
        yaw_target = _nearest_normal_axis_yaw(scene, pos_after, current_yaw)
    elif _parallel_wall_anchor(anchor_text):
        yaw_target = _nearest_parallel_wall_yaw(scene, pos_after, current_yaw)
    elif info is not None and _wall_affine_anchor(anchor_text, zone_hint):
        yaw_target = _nearest_normal_axis_yaw(scene, pos_after, current_yaw)
    if yaw_target is not None:
        _set_object_yaw(scene, anchor_idx, yaw_target)

    moved["after_pos"] = list(scene["objects"][anchor_idx].get("pos", [0.0, 0.0, 0.0]))
    moved["after_yaw"] = _yaw_from_quaternion(scene["objects"][anchor_idx].get("rot", [0.0, 0.0, 0.0, 1.0]))
    return moved


def simple_group_local_optimize(
    scene: Dict[str, Any],
    group_item: Dict[str, Any],
    optimize_only_indices: List[int],
    *,
    anchor_idx: Optional[int] = None,
    max_passes: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Lightweight deterministic group-local optimizer.
    Only applies:
      1) anchor snap / wall-center alignment
      3) collision push
      4) OOB inward projection
    Only newly added objects are moved.
    """
    sc = copy.deepcopy(scene)
    optimize_only_indices = sorted({
        int(i)
        for i in optimize_only_indices
        if isinstance(i, int) and 0 <= int(i) < len(sc.get("objects", []))
    })
    movable_set = set(optimize_only_indices)
    if not optimize_only_indices:
        return sc, {
            "mode": "simple_group_local_optimize",
            "optimized_indices": [],
            "anchor_idx": None,
            "passes": 0,
            "anchor_snap": None,
            "collision_push_counts": {},
            "final_oob": {},
        }

    if anchor_idx is None:
        anchor_idx = optimize_only_indices[0]
    if anchor_idx not in movable_set:
        anchor_idx = optimize_only_indices[0]

    anchor_snap = _snap_anchor(sc, anchor_idx, group_item)

    collision_push_counts: Dict[str, int] = {str(i): 0 for i in optimize_only_indices}
    for _ in range(max(1, int(max_passes))):
        for idx in optimize_only_indices:
            _project_object_inside_room(sc, idx)
        for idx in optimize_only_indices:
            total_area = _push_object_from_collisions(sc, idx, movable_set)
            if total_area > 1e-8:
                collision_push_counts[str(idx)] += 1
        for idx in optimize_only_indices:
            _project_object_inside_room(sc, idx)

    final_oob = {}
    floor_polygon = _get_floor_polygon(sc)
    for idx in optimize_only_indices:
        obj = sc["objects"][idx]
        oob = 0.0
        if floor_polygon is not None:
            oob = float(
                compute_oob(
                    obj,
                    floor_polygon,
                    sc.get("bounds_bottom", []),
                    sc.get("bounds_top", []),
                    is_debug=False,
                )
            )
        final_oob[str(idx)] = round(oob, 8)

    summary = {
        "mode": "simple_group_local_optimize",
        "optimized_indices": optimize_only_indices,
        "anchor_idx": anchor_idx,
        "passes": int(max_passes),
        "anchor_snap": anchor_snap,
        "collision_push_counts": collision_push_counts,
        "final_oob": final_oob,
    }
    return sc, summary



def _cleanup_cuda() -> None:
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def attach_group_repair_in_loop(ReSpaceCls, ConfigCls, optimize_fn, PromptGenCls):
    def build_group_plan(self, prompt, current_scene=None, room_type=None, max_groups: Optional[int] = None):
        if current_scene is None:
            current_scene = self._sample_random_bounds(self.dataset_train, room_type)
        if max_groups is None:
            max_groups = int(os.getenv("MAX_FUNCTIONAL_GROUPS", "4"))

        query = self._build_full_query_for_zeroshot_model(prompt, scenegraph=current_scene)
        max_query_chars = int(os.getenv("GROUP_PLAN_MAX_QUERY_CHARS", "2800"))
        if len(query) > max_query_chars:
            query = query[-max_query_chars:]

        messages = [
            {"role": "system", "content": _build_group_plan_system_prompt(max_groups=max_groups)},
            {"role": "user", "content": query},
        ]

        max_new_tokens = int(os.getenv("GROUP_PLAN_MAX_NEW_TOKENS", "384"))
        retry_max_new_tokens = int(os.getenv("GROUP_PLAN_RETRY_MAX_NEW_TOKENS", "160"))
        temperature = float(os.getenv("GROUP_PLAN_TEMPERATURE", "0.0"))
        top_p = float(os.getenv("GROUP_PLAN_TOP_P", "1.0"))
        top_k = int(os.getenv("GROUP_PLAN_TOP_K", "0"))

        raw_text = None
        try:
            import torch
            torch.use_deterministic_algorithms(False)
            _cleanup_cuda()
            with torch.inference_mode():
                if getattr(self, "vanilla_vllm_engine", None) is not None:
                    vllm_prompt = f"<s>[INST] {_build_group_plan_system_prompt(max_groups=max_groups)} [/INST]\n\n{query}</s>"
                    inputs = self.vanilla_tokenizer(vllm_prompt, return_tensors="pt")
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]
                    response = self.vanilla_vllm_engine.generate(
                        input_ids,
                        attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    if isinstance(response, list):
                        response = response[0]
                    raw_text = str(response).strip()
                else:
                    gen_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "pad_token_id": self.vanilla_pipeline.tokenizer.eos_token_id,
                        "do_sample": temperature > 0.0,
                    }
                    if temperature > 0.0:
                        gen_kwargs["temperature"] = temperature
                    outputs = self.vanilla_pipeline(messages, **gen_kwargs)
                    raw_text = outputs[0]["generated_text"][-1]["content"].strip()
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                print(f"group plan generation failed: {exc}")
                traceback.print_exc()
                return {"groups": []}
            print("[OOM] group planner failed, retry with shorter query + shorter decode")
            traceback.print_exc()
            _cleanup_cuda()
            try:
                import torch
                compact_query = query[-int(os.getenv("GROUP_PLAN_RETRY_QUERY_CHARS", "1200")):]
                retry_messages = [
                    {"role": "system", "content": _build_group_plan_system_prompt(max_groups=max_groups)},
                    {"role": "user", "content": compact_query},
                ]
                with torch.inference_mode():
                    if getattr(self, "vanilla_vllm_engine", None) is not None:
                        vllm_prompt = f"<s>[INST] {_build_group_plan_system_prompt(max_groups=max_groups)} [/INST]\n\n{compact_query}</s>"
                        inputs = self.vanilla_tokenizer(vllm_prompt, return_tensors="pt")
                        input_ids = inputs["input_ids"]
                        attention_mask = inputs["attention_mask"]
                        response = self.vanilla_vllm_engine.generate(
                            input_ids,
                            attention_mask,
                            max_new_tokens=retry_max_new_tokens,
                            temperature=0.0,
                            top_p=1.0,
                            top_k=0,
                        )
                        if isinstance(response, list):
                            response = response[0]
                        raw_text = str(response).strip()
                    else:
                        outputs = self.vanilla_pipeline(
                            retry_messages,
                            max_new_tokens=retry_max_new_tokens,
                            pad_token_id=self.vanilla_pipeline.tokenizer.eos_token_id,
                            do_sample=False,
                        )
                        raw_text = outputs[0]["generated_text"][-1]["content"].strip()
            except Exception as exc2:
                print(f"group plan retry failed: {exc2}")
                traceback.print_exc()
                return {"groups": []}
        except Exception as exc:
            print(f"group plan generation failed: {exc}")
            traceback.print_exc()
            return {"groups": []}
        finally:
            try:
                import torch
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            _cleanup_cuda()

        payload = None
        try:
            payload = self._extract_first_json_object(raw_text)
        except Exception:
            payload = _safe_json_loads(raw_text)
        return _normalize_group_plan(payload, max_groups=max_groups)

    def _build_group_add_prompt(self, global_prompt, group_item, obj_prompt, is_anchor, current_scene):
        group_name = group_item["group_name"]
        anchor = group_item["anchor_object"]
        zone_hint = group_item["zone_hint"]

        # 只保留已有物体的短名字
        def _iter_leaf_objects(entries):
            flat = []
            if not isinstance(entries, list):
                return flat
            for item in entries:
                if not isinstance(item, dict):
                    continue
                nested = item.get("objects")
                if isinstance(nested, list) and len(nested) > 0:
                    flat.extend(_iter_leaf_objects(nested))
                else:
                    flat.append(item)
            return flat

        def _extract_short_name(obj):
            # 1) 优先从 prompt / planning_prompt_raw 里抽 Add object: xxx
            for key in ["prompt", "planning_prompt_raw"]:
                text = str(obj.get(key) or "").strip()
                if not text:
                    continue
                marker = "Add object:"
                if marker in text:
                    tail = text.split(marker, 1)[1].strip()
                    for stop in [" Role:", " Group:", " Anchor:", " Zone hint:", " Existing objects:"]:
                        if stop in tail:
                            tail = tail.split(stop, 1)[0].strip()
                    tail = tail.strip(" .,:;\"'").lower()
                    if tail:
                        return tail[:40]

            # 2) 再退回 category / type
            for key in ["category", "type", "super_category"]:
                text = str(obj.get(key) or "").strip()
                if text:
                    return text.lower()[:40]

            # 3) 最后才退回 desc / sampled_asset_desc 的前几个词
            for key in ["desc", "sampled_asset_desc", "description", "style_description"]:
                text = str(obj.get(key) or "").strip()
                if text:
                    text = text.split(",")[0].strip().lower()
                    words = text.split()
                    if words:
                        return " ".join(words[:4])[:40]

            return ""

        existing = []
        for o in _iter_leaf_objects(current_scene.get("objects", [])):
            name = _extract_short_name(o)
            if name:
                existing.append(name)

        existing_text = "; ".join(existing[:8]) if existing else "(empty)"

        # 只保留 global_prompt 的最后请求部分，别整段塞进来
        gp = str(global_prompt or "").strip()
        marker = "Now perform scene planning for the following request:"
        if marker in gp:
            gp = gp.split(marker, 1)[1].strip()
        gp = gp[:220]

        schema = (
            'Return ONLY one JSON object with keys "desc", "pos", "rot", "size". '
            'Do not output markdown. Do not output explanations. '
            'The first character must be { and the last character must be }. '
        )

        if is_anchor:
            return (
                f"{schema}"
                f'User request: "{gp}". '
                f"Existing objects: {existing_text}. "
                f"Add object: {obj_prompt}. "
                f"Role: anchor of group {group_name}. "
                f"Zone hint: {zone_hint}."
            )

        return (
            f"{schema}"
            f'User request: "{gp}". '
            f"Existing objects: {existing_text}. "
            f"Add object: {obj_prompt}. "
            f"Group: {group_name}. "
            f"Anchor: {anchor}. "
            f"Zone hint: {zone_hint}."
        )

    def repair_partial_scene(
        self,
        current_scene,
        out_root,
        user_prompt,
        cfg=None,
        relation_plan=None,
        tag="partial_repair",
        group_item=None,
        optimize_only_indices=None,
        anchor_idx=None,
        use_simple_group_optimizer: bool = True,
    ):
        if current_scene is None or len(current_scene.get("objects", [])) == 0:
            return current_scene, None

        run_dir = Path(out_root) / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        run_dir.mkdir(parents=True, exist_ok=True)

        optimize_only_indices = sorted({int(i) for i in (optimize_only_indices or []) if isinstance(i, int)})

        if use_simple_group_optimizer and optimize_only_indices:
            before_metrics = _compute_group_progress_metrics(current_scene)

            candidate_scene, candidate_summary = simple_group_local_optimize(
                current_scene,
                group_item=group_item or {},
                optimize_only_indices=optimize_only_indices,
                anchor_idx=anchor_idx,
                max_passes=int(os.getenv("GROUP_SIMPLE_OPT_PASSES", "3")),
            )
            candidate_metrics = _compute_group_progress_metrics(candidate_scene)
            accepted = _is_strictly_better_group_metrics(before_metrics, candidate_metrics)

            kept_scene = candidate_scene if accepted else copy.deepcopy(current_scene)
            kept_metrics = candidate_metrics if accepted else before_metrics

            summary = {
                "mode": "simple_group_local_optimize",
                "optimized_indices": optimize_only_indices,
                "anchor_idx": anchor_idx,
                "accepted": bool(accepted),
                "accept_rule": "accept iff pbl/oob/mbl are all no worse and at least one is strictly better",
                "metrics_before": before_metrics,
                "metrics_candidate": candidate_metrics,
                "metrics_kept": kept_metrics,
                "candidate_summary": candidate_summary,
            }

            _write_json(run_dir / "summary.json", summary)
            _write_json(run_dir / "scene_before_simple_group_opt.json", current_scene)
            _write_json(run_dir / "scene_after_simple_group_opt_candidate.json", candidate_scene)
            _write_json(run_dir / "scene_kept_after_acceptance.json", kept_scene)

            final_dir = run_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            _write_json(final_dir / "scene.json", kept_scene)

            return kept_scene, summary

        use_local = os.getenv("USE_LOCAL_VLM_OPTIMIZER", "0") == "1"
        if use_local:
            generator = PromptGenCls(
                model=os.getenv("LOCAL_VLM_MODEL", os.getenv("MOVE_PROMPT_MODEL", "/home2/zhangjiawei/respace/model/qwen3-sft+grpo")),
                backend="local",
                timeout_s=float(os.getenv("MOVE_PROMPT_TIMEOUT_S", "120")),
                device=os.getenv("LOCAL_VLM_DEVICE", "cuda"),
                torch_dtype=os.getenv("LOCAL_VLM_DTYPE", "bfloat16"),
                max_new_tokens=int(os.getenv("MOVE_PROMPT_MAX_TOKENS", "1600")),
            )
        else:
            model = os.getenv("MOVE_PROMPT_MODEL", os.getenv("YUNWU_AI_MODEL", "gpt-4o"))
            generator = PromptGenCls(
                model=model,
                backend="api",
                api_base=os.getenv("YUNWU_AI_API_BASE"),
                api_key=os.getenv("YUNWU_AI_API_KEY"),
                timeout_s=float(os.getenv("MOVE_PROMPT_TIMEOUT_S", "120")),
                max_new_tokens=int(os.getenv("MOVE_PROMPT_MAX_TOKENS", "1600")),
            )
        cfg_local = copy.deepcopy(cfg) if cfg is not None else ConfigCls()
        extra_hints_text = (
            "FINAL GLOBAL REPAIR MODE:\n"
            "1) This is the final full-scene polish.\n"
            f"2) Global user request: {user_prompt}\n"
        )
        try:
            try:
                summary = optimize_fn(
                    scene=current_scene,
                    out_root=run_dir,
                    respace=self,
                    generator=generator,
                    extra_hints_text=extra_hints_text,
                    cfg=cfg_local,
                    zero_shot_relation_plan=relation_plan,
                )
            except TypeError:
                summary = optimize_fn(
                    scene=current_scene,
                    out_root=run_dir,
                    respace=self,
                    generator=generator,
                    extra_hints_text=extra_hints_text,
                    cfg=cfg_local,
                )
        finally:
            try:
                del generator
            except Exception:
                pass
            _cleanup_cuda()

        final_scene_path = run_dir / "final" / "scene.json"
        if final_scene_path.exists():
            repaired_scene = json.loads(final_scene_path.read_text(encoding="utf-8"))
            return repaired_scene, summary
        return current_scene, summary

    def place_group(
        self,
        group_item,
        current_scene,
        global_prompt,
        do_rendering_with_object_count=False,
        pth_viz_output=None,
    ):
        working_scene = copy.deepcopy(current_scene)
        records: List[Dict[str, Any]] = []
        added_indices: List[int] = []
        anchor = _singularize_object_prompt(group_item["anchor_object"])
        object_list = list(group_item["objects"])
        ordered_objects = [anchor] + [obj for obj in object_list if _singularize_object_prompt(obj) != anchor]

        print(
            f"\n[group start] name={group_item['group_name']} "
            f"priority={group_item.get('priority')} "
            f"zone_hint={group_item.get('zone_hint')} "
            f"anchor={group_item.get('anchor_object')} "
            f"objects={ordered_objects}",
            flush=True,
        )

        for k, obj_prompt in enumerate(ordered_objects):
            is_anchor = (k == 0)
            add_prompt = self._build_group_add_prompt(global_prompt, group_item, obj_prompt, is_anchor, working_scene)

            terminal_prompt = _terminal_prompt_preview(add_prompt)
            print(
                f"[group prompt][{group_item['group_name']}][obj_{k:02d}] {terminal_prompt}",
                flush=True,
            )

            prev_n = len(working_scene.get("objects", []))
            working_scene, success = self.add_object(
                add_prompt,
                working_scene,
                do_rendering_with_object_count=do_rendering_with_object_count,
                pth_viz_output=pth_viz_output,
                temp=0.7,
            )
            next_n = len(working_scene.get("objects", []))
            new_indices = list(range(prev_n, next_n)) if next_n > prev_n else []
            added_indices.extend(new_indices)

            metrics_after_add = _compute_group_progress_metrics(working_scene)

            print(
                f"[group add result][{group_item['group_name']}][obj_{k:02d}] "
                f"success={bool(success)} "
                f"added_indices={new_indices} "
                f"{_format_group_progress_metrics(metrics_after_add)}",
                flush=True,
            )

            records.append(
                {
                    "object_prompt": obj_prompt,
                    "is_anchor": is_anchor,
                    "success": bool(success),
                    "added_indices": new_indices,
                    "add_prompt": add_prompt,
                    "metrics_after_add": metrics_after_add,
                }
            )

        anchor_idx = added_indices[0] if added_indices else None
        return working_scene, records, added_indices, anchor_idx

    def handle_prompt_group_repair_in_loop(
        self,
        prompt,
        current_scene=None,
        room_type=None,
        do_rendering_with_object_count=False,
        pth_viz_output=None,
        return_aux=False,
        include_relation_plan=True,
        partial_repair_cfg=None,
        final_repair_cfg=None,
        run_final_global_repair=True,
        max_groups: Optional[int] = None,
    ):
        if current_scene is None:
            current_scene = self._sample_random_bounds(self.dataset_train, room_type)
        if self.dataset_stats_for_prompt is None:
            self.dataset_stats_for_prompt = self._prepare_dataset_stats_for_object_sampler(current_scene.get("room_type"))
        out_root = _ensure_dir(pth_viz_output or "./group_repair_in_loop_runs")
        if max_groups is None:
            max_groups = int(os.getenv("MAX_FUNCTIONAL_GROUPS", "4"))

        group_plan = self.build_group_plan(prompt, current_scene=current_scene, room_type=room_type, max_groups=max_groups)

        relation_plan = None
        if include_relation_plan:
            try:
                relation_plan = self.build_relation_plan(prompt, current_scene=current_scene, room_type=room_type)
            except Exception as exc:
                print(f"relation plan generation failed: {exc}")
                relation_plan = None

        aux = {
            "group_plan": group_plan,
            "global_relation_plan": relation_plan,
            "group_records": [],
            "partial_repair_records": [],
            "final_repair_summary": None,
            "max_groups": max_groups,
        }

        working_scene = copy.deepcopy(current_scene)
        groups = group_plan.get("groups", [])
        _write_json(out_root / "group_plan.json", group_plan)

        for gi, group_item in enumerate(groups):
            group_name = _group_name_slug(group_item["group_name"])
            group_dir = out_root / f"group_{gi:02d}_{group_name}"
            group_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"\n================ GROUP {gi:02d} / {len(groups):02d} ================",
                flush=True,
            )
            print(
                f"[group info] group_name={group_item['group_name']} "
                f"anchor={group_item['anchor_object']} "
                f"zone_hint={group_item['zone_hint']} "
                f"priority={group_item['priority']}",
                flush=True,
            )

            working_scene, placement_records, added_indices, anchor_idx = self.place_group(
                group_item=group_item,
                current_scene=working_scene,
                global_prompt=prompt,
                do_rendering_with_object_count=do_rendering_with_object_count,
                pth_viz_output=group_dir,
            )

            metrics_after_group_add = _compute_group_progress_metrics(working_scene)
            print(
                f"[group summary][after_add][group_{gi:02d}:{group_item['group_name']}] "
                f"{_format_group_progress_metrics(metrics_after_group_add)}",
                flush=True,
            )

            aux["group_records"].append(
                {
                    "group_index": gi,
                    "group_name": group_item["group_name"],
                    "placement_records": placement_records,
                    "num_objects_after_group": len(working_scene.get("objects", [])),
                    "added_indices": added_indices,
                    "anchor_idx": anchor_idx,
                    "metrics_after_group_add": metrics_after_group_add,
                }
            )

            working_scene, repair_summary = self.repair_partial_scene(
                current_scene=working_scene,
                out_root=group_dir,
                user_prompt=prompt,
                cfg=partial_repair_cfg,
                relation_plan=relation_plan,
                tag=f"repair_after_group_{gi:02d}",
                group_item=group_item,
                optimize_only_indices=added_indices,
                anchor_idx=anchor_idx,
                use_simple_group_optimizer=True,
            )

            metrics_after_group_repair = _compute_group_progress_metrics(working_scene)
            accepted = bool((repair_summary or {}).get("accepted", False))

            print(
                f"[group summary][after_repair][group_{gi:02d}:{group_item['group_name']}] "
                f"accepted={accepted} "
                f"{_format_group_progress_metrics(metrics_after_group_repair)}",
                flush=True,
            )

            aux["partial_repair_records"].append(
                {
                    "group_index": gi,
                    "group_name": group_item["group_name"],
                    "repair_summary": repair_summary,
                    "optimized_indices": added_indices,
                    "anchor_idx": anchor_idx,
                    "metrics_after_group_repair": metrics_after_group_repair,
                }
            )

        if run_final_global_repair and len(working_scene.get("objects", [])) > 0:
            final_dir = out_root / "final_global_repair"
            final_dir.mkdir(parents=True, exist_ok=True)
            working_scene, final_summary = self.repair_partial_scene(
                current_scene=working_scene,
                out_root=final_dir,
                user_prompt=prompt,
                cfg=final_repair_cfg,
                relation_plan=relation_plan,
                tag="final_global_repair",
                group_item=None,
                optimize_only_indices=None,
                anchor_idx=None,
                use_simple_group_optimizer=False,
            )
            aux["final_repair_summary"] = final_summary

        _write_json(out_root / "group_generation_aux.json", aux)
        ok = len(working_scene.get("objects", [])) > 0
        if return_aux:
            return working_scene, ok, aux
        return working_scene, ok

    def generate_full_scene_group_repair_in_loop(
        self,
        room_type=None,
        n_objects=None,
        scene_bounds_only=None,
        do_rendering_with_object_count=False,
        pth_viz_output=None,
        return_aux=False,
        include_relation_plan=True,
        partial_repair_cfg=None,
        final_repair_cfg=None,
        run_final_global_repair=True,
        max_groups: Optional[int] = None,
    ):
        self.dataset_stats_for_prompt = self._prepare_dataset_stats_for_object_sampler(room_type)
        if scene_bounds_only is None:
            scene_bounds_only = self._sample_random_bounds(self.dataset_train, room_type)
        if n_objects is None:
            n_objects = 8
        prompt = f"create a {room_type if room_type is not None else 'room'} with {n_objects} objects."
        return self.handle_prompt_group_repair_in_loop(
            prompt=prompt,
            current_scene=scene_bounds_only,
            room_type=room_type,
            do_rendering_with_object_count=do_rendering_with_object_count,
            pth_viz_output=pth_viz_output,
            return_aux=return_aux,
            include_relation_plan=include_relation_plan,
            partial_repair_cfg=partial_repair_cfg,
            final_repair_cfg=final_repair_cfg,
            run_final_global_repair=run_final_global_repair,
            max_groups=max_groups,
        )

    ReSpaceCls.build_group_plan = build_group_plan
    ReSpaceCls._build_group_add_prompt = _build_group_add_prompt
    ReSpaceCls.repair_partial_scene = repair_partial_scene
    ReSpaceCls.place_group = place_group
    ReSpaceCls.handle_prompt_group_repair_in_loop = handle_prompt_group_repair_in_loop
    ReSpaceCls.generate_full_scene_group_repair_in_loop = generate_full_scene_group_repair_in_loop
    return ReSpaceCls


__all__ = ["attach_group_repair_in_loop", "simple_group_local_optimize"]