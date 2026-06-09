#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

REL_PRIORITY = {
    "against_wall": 100,
    "parallel": 95,
    "facing": 90,
    "facing_pair": 85,
    "centered_with": 80,
    "near": 70,
    "side_of": 60,
    "in_front_of": 55,
    "distance_band": 40,
}

DEFAULT_REL_TYPES = {
    "near",
    "facing",
    "facing_pair",
    "centered_with",
    "against_wall",
    "parallel",
}

ALL_REL_TYPES = {
    "near",
    "distance_band",
    "facing",
    "facing_pair",
    "centered_with",
    "in_front_of",
    "side_of",
    "against_wall",
    "parallel",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_angle_deg(a: float) -> float:
    return a % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    d = abs(normalize_angle_deg(a) - normalize_angle_deg(b))
    return min(d, 360.0 - d)


def axis_angle_diff_deg(a: float, b: float) -> float:
    return min(angle_diff_deg(a, b), angle_diff_deg(a, normalize_angle_deg(b + 180.0)))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def vec_norm2(x: float, z: float) -> float:
    return math.hypot(x, z)


def dot2(ax: float, az: float, bx: float, bz: float) -> float:
    return ax * bx + az * bz


def signed_proj(ax: float, az: float, bx: float, bz: float, vx: float, vz: float) -> float:
    return (bx - ax) * vx + (bz - az) * vz


def forward_vec_from_yaw(yaw_deg: float) -> Tuple[float, float]:
    rad = math.radians(yaw_deg)
    return math.sin(rad), math.cos(rad)


def lateral_vec_from_yaw(yaw_deg: float) -> Tuple[float, float]:
    fx, fz = forward_vec_from_yaw(yaw_deg)
    return fz, -fx


def quat_to_yaw_deg(q: Sequence[float]) -> float:
    if len(q) != 4:
        return 0.0
    x, y, z, w = [float(v) for v in q]
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    return normalize_angle_deg(yaw)


def room_center_xz(bounds_bottom: Sequence[Sequence[float]]) -> Tuple[float, float]:
    pts = [(float(p[0]), float(p[2])) for p in bounds_bottom if len(p) >= 3]
    if not pts:
        return 0.0, 0.0
    return sum(x for x, _ in pts) / len(pts), sum(z for _, z in pts) / len(pts)


def obj_diag_size_xz(obj: Dict[str, Any]) -> float:
    size = obj.get("size", [1.0, 1.0, 1.0])
    sx = float(size[0]) if len(size) > 0 else 1.0
    sz = float(size[2]) if len(size) > 2 else 1.0
    return math.hypot(sx, sz)


def obj_size_xz(obj: Dict[str, Any]) -> Tuple[float, float]:
    size = obj.get("size", [1.0, 1.0, 1.0])
    sx = float(size[0]) if len(size) > 0 else 1.0
    sz = float(size[2]) if len(size) > 2 else 1.0
    return sx, sz


def pair_target_dist(a: Dict[str, Any], b: Dict[str, Any], alpha: float = 0.35, bias: float = 0.15) -> float:
    return alpha * (obj_diag_size_xz(a) + obj_diag_size_xz(b)) + bias


def xz_dist(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    pa = a.get("pos", [0.0, 0.0, 0.0])
    pb = b.get("pos", [0.0, 0.0, 0.0])
    return math.hypot(float(pa[0]) - float(pb[0]), float(pa[2]) - float(pb[2]))


def distance_to_nearest_wall_xz(bounds_bottom: Sequence[Sequence[float]], pos: Sequence[float]) -> float:
    pts = [(float(p[0]), float(p[2])) for p in bounds_bottom if len(p) >= 3]
    if len(pts) < 3:
        return 999.0
    best = float("inf")
    px, pz = float(pos[0]), float(pos[2])
    for i in range(len(pts)):
        ax, az = pts[i]
        bx, bz = pts[(i + 1) % len(pts)]
        abx, abz = bx - ax, bz - az
        apx, apz = px - ax, pz - az
        denom = abx * abx + abz * abz
        if denom < 1e-12:
            continue
        t = clamp((apx * abx + apz * abz) / denom, 0.0, 1.0)
        proj_x = ax + t * abx
        proj_z = az + t * abz
        best = min(best, math.hypot(px - proj_x, pz - proj_z))
    return best


def nearest_wall_normal_yaw(bounds_bottom: Sequence[Sequence[float]], pos: Sequence[float]) -> Optional[float]:
    pts = [(float(p[0]), float(p[2])) for p in bounds_bottom if len(p) >= 3]
    if len(pts) < 3:
        return None
    center_x, center_z = room_center_xz(bounds_bottom)
    px, pz = float(pos[0]), float(pos[2])
    best_dist = float("inf")
    best_normal = None
    for i in range(len(pts)):
        ax, az = pts[i]
        bx, bz = pts[(i + 1) % len(pts)]
        abx, abz = bx - ax, bz - az
        apx, apz = px - ax, pz - az
        denom = abx * abx + abz * abz
        if denom < 1e-12:
            continue
        t = clamp((apx * abx + apz * abz) / denom, 0.0, 1.0)
        proj_x = ax + t * abx
        proj_z = az + t * abz
        dist = math.hypot(px - proj_x, pz - proj_z)
        if dist >= best_dist:
            continue
        nx, nz = -abz, abx
        nl = math.hypot(nx, nz)
        if nl < 1e-12:
            continue
        nx, nz = nx / nl, nz / nl
        if nx * (center_x - proj_x) + nz * (center_z - proj_z) < 0:
            nx, nz = -nx, -nz
        best_dist = dist
        best_normal = (nx, nz)
    if best_normal is None:
        return None
    nx, nz = best_normal
    return normalize_angle_deg(math.degrees(math.atan2(nx, nz)))


def nearest_parallel_wall_yaw(bounds_bottom: Sequence[Sequence[float]], pos: Sequence[float], current_yaw: Optional[float] = None) -> Optional[float]:
    wall_yaw = nearest_wall_normal_yaw(bounds_bottom, pos)
    if wall_yaw is None:
        return None
    candidates = [normalize_angle_deg(wall_yaw + 90.0), normalize_angle_deg(wall_yaw + 270.0)]
    if current_yaw is None:
        return candidates[0]
    return min(candidates, key=lambda y: angle_diff_deg(current_yaw, y))


DEFAULT_DATASET_CAT_MAP = {
    "children cabinet": "storage",
    "nightstand": "nightstand",
    "bookcase / jewelry armoire": "storage",
    "wardrobe": "storage",
    "coffee table": "coffee_table",
    "corner/side table": "storage",
    "sideboard / side cabinet / console table": "storage",
    "wine cabinet": "storage",
    "tv stand": "storage",
    "drawer chest / corner cabinet": "storage",
    "shelf": "storage",
    "round end table": "storage",
    "king-size bed": "bed",
    "bunk bed": "bed",
    "bed frame": "bed",
    "single bed": "bed",
    "kids bed": "bed",
    "couch bed": "bed",
    "dining chair": "chair",
    "lounge chair / cafe chair / office chair": "chair",
    "dressing chair": "chair",
    "classic chinese chair": "chair",
    "barstool": "chair",
    "hanging chair": "chair",
    "folding chair": "chair",
    "dressing table": "table",
    "dining table": "table",
    "desk": "desk",
    "bar": "table",
    "three-seat / multi-seat sofa": "sofa",
    "armchair": "sofa",
    "loveseat sofa": "sofa",
    "l-shaped sofa": "sofa",
    "u-shaped sofa": "sofa",
    "lazy sofa": "sofa",
    "chaise longue sofa": "sofa",
    "footstool / sofastool / bed end stool / stool": "stool",
    "pendant lamp": "pendant_lamp",
    "ceiling lamp": "lighting",
    "floor lamp": "lighting",
    "wall lamp": "lighting",
}

DESC_KEYWORDS = [
    ("dining table", "table"),
    ("desk", "desk"),
    ("table", "table"),
    ("dining chair", "chair"),
    ("chair", "chair"),
    ("sofa", "sofa"),
    ("bed", "bed"),
    ("nightstand", "nightstand"),
    ("bookcase", "storage"),
    ("shelf", "storage"),
    ("cabinet", "storage"),
    ("wardrobe", "storage"),
    ("tv stand", "storage"),
    ("console table", "storage"),
    ("pendant lamp", "pendant_lamp"),
    ("ceiling lamp", "lighting"),
    ("floor lamp", "lighting"),
    ("wall lamp", "lighting"),
    ("lamp", "lighting"),
]


def normalize_category(raw_category: Optional[str], desc: str = "") -> str:
    cat = (raw_category or "").strip().lower()
    if cat in DEFAULT_DATASET_CAT_MAP:
        return DEFAULT_DATASET_CAT_MAP[cat]
    desc_l = desc.lower()
    for k, v in DESC_KEYWORDS:
        if k in cat or k in desc_l:
            return v
    if "light" in cat or "lamp" in cat or "light" in desc_l or "lamp" in desc_l:
        return "lighting"
    return "other"


def resolve_categories(scene: Dict[str, Any], jid2cat: Optional[Dict[str, str]] = None) -> List[str]:
    cats = []
    for obj in scene.get("objects", []):
        raw = obj.get("category")
        jid = obj.get("jid")
        desc = obj.get("desc", "")
        if raw is None and jid2cat and isinstance(jid, str) and jid in jid2cat:
            raw = jid2cat[jid]
        cats.append(normalize_category(raw, desc))
    return cats


@dataclass
class Relation:
    src_idx: int
    tgt_idx: Optional[int]
    type: str
    confidence: float
    reason: str
    extra: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        out = {
            "src_idx": self.src_idx,
            "tgt_idx": self.tgt_idx,
            "type": self.type,
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
        }
        out.update(self.extra)
        return out


def extract_near(scene: Dict[str, Any], cats: List[str]) -> List[Relation]:
    objs = scene["objects"]
    rels = []
    best_per_src = defaultdict(list)
    for i, ci in enumerate(cats):
        for j, cj in enumerate(cats):
            if i == j:
                continue
            valid = (
                (ci == "chair" and cj in {"table", "desk", "sofa"}) or
                (ci == "pendant_lamp" and cj in {"table", "desk"}) or
                (ci == "nightstand" and cj == "bed") or
                (ci == "coffee_table" and cj == "sofa")
            )
            if not valid:
                continue
            d = xz_dist(objs[i], objs[j])
            tgt = pair_target_dist(objs[i], objs[j])
            if d <= tgt + 0.25:
                conf = clamp(1.0 - max(0.0, d - tgt) / max(0.25, tgt + 0.25), 0.6, 0.99)
                best_per_src[i].append((d, Relation(i, j, "near", conf, f"{ci} is close to {cj}", {"dist": round(d,4), "target_dist": round(tgt,4)})))
    for i, items in best_per_src.items():
        items.sort(key=lambda x: x[0])
        for _, rel in items[:2]:
            rels.append(rel)
    return rels


def extract_facing(scene: Dict[str, Any], cats: List[str]) -> List[Relation]:
    objs = scene["objects"]
    rels = []
    best_per_src = defaultdict(list)
    for i, ci in enumerate(cats):
        for j, cj in enumerate(cats):
            if i == j:
                continue
            valid = (
                (ci == "chair" and cj in {"table", "desk", "sofa"}) or
                (ci == "sofa" and cj == "storage") or
                (ci == "bed" and cj == "storage")
            )
            if not valid:
                continue
            pi = objs[i].get("pos", [0,0,0])
            pj = objs[j].get("pos", [0,0,0])
            yaw = quat_to_yaw_deg(objs[i].get("rot", [0,0,0,1]))
            fx, fz = forward_vec_from_yaw(yaw)
            dx, dz = float(pj[0]) - float(pi[0]), float(pj[2]) - float(pi[2])
            dn = vec_norm2(dx, dz)
            if dn < 1e-6:
                continue
            ux, uz = dx / dn, dz / dn
            dot = dot2(fx, fz, ux, uz)
            tgt = pair_target_dist(objs[i], objs[j])
            if dot >= 0.7 and dn <= 2.5 * tgt:
                conf = clamp(0.5 * (dot + 1.0), 0.65, 0.99)
                best_per_src[i].append((-dot, Relation(i, j, "facing", conf, f"{ci} faces {cj}", {"dist": round(dn,4), "dot": round(dot,4)})))
    for i, items in best_per_src.items():
        items.sort(key=lambda x: x[0])
        if items:
            rels.append(items[0][1])
    return rels


def extract_facing_pair(scene: Dict[str, Any], cats: List[str]) -> List[Relation]:
    objs = scene["objects"]
    rels = []
    for i in range(len(objs)):
        if cats[i] != "chair":
            continue
        for j in range(i + 1, len(objs)):
            if cats[j] != "chair":
                continue
            pi = objs[i].get("pos", [0,0,0])
            pj = objs[j].get("pos", [0,0,0])
            dx, dz = float(pj[0]) - float(pi[0]), float(pj[2]) - float(pi[2])
            dn = vec_norm2(dx, dz)
            if dn < 1e-6:
                continue
            uijx, uijz = dx / dn, dz / dn
            ujix, ujiz = -uijx, -uijz
            yawi = quat_to_yaw_deg(objs[i].get("rot", [0,0,0,1]))
            yawj = quat_to_yaw_deg(objs[j].get("rot", [0,0,0,1]))
            fix, fiz = forward_vec_from_yaw(yawi)
            fjx, fjz = forward_vec_from_yaw(yawj)
            di = dot2(fix, fiz, uijx, uijz)
            dj = dot2(fjx, fjz, ujix, ujiz)
            if di >= 0.7 and dj >= 0.7:
                conf = clamp((di + dj) / 2.0, 0.65, 0.99)
                rels.append(Relation(i, j, "facing_pair", conf, "two seats face each other", {"dist": round(dn,4), "dot_i": round(di,4), "dot_j": round(dj,4)}))
    return rels


def extract_centered_with(scene: Dict[str, Any], cats: List[str]) -> List[Relation]:
    objs = scene["objects"]
    rels = []
    for i, ci in enumerate(cats):
        for j, cj in enumerate(cats):
            if i == j:
                continue
            valid = (
                (ci == "pendant_lamp" and cj in {"table", "desk"}) or
                (ci == "chair" and cj == "desk") or
                (ci == "coffee_table" and cj == "sofa")
            )
            if not valid:
                continue
            pi = objs[i].get("pos", [0,0,0])
            pj = objs[j].get("pos", [0,0,0])
            yawj = quat_to_yaw_deg(objs[j].get("rot", [0,0,0,1]))
            lx, lz = lateral_vec_from_yaw(yawj)
            lateral = abs(signed_proj(float(pj[0]), float(pj[2]), float(pi[0]), float(pi[2]), lx, lz))
            sx, sz = obj_size_xz(objs[j])
            eps = max(0.12, 0.2 * min(sx, sz))
            if lateral <= eps:
                conf = clamp(1.0 - lateral / max(eps, 1e-6), 0.7, 0.99)
                rels.append(Relation(i, j, "centered_with", conf, f"{ci} is laterally centered with {cj}", {"lateral": round(lateral,4), "eps": round(eps,4)}))
    return rels


def extract_in_front_of(scene: Dict[str, Any], cats: List[str]) -> List[Relation]:
    objs = scene["objects"]
    rels = []
    for i, ci in enumerate(cats):
        for j, cj in enumerate(cats):
            if i == j:
                continue
            valid = (
                (ci == "chair" and cj in {"table", "desk"}) or
                (ci == "coffee_table" and cj == "sofa")
            )
            if not valid:
                continue
            pi = objs[i].get("pos", [0,0,0])
            pj = objs[j].get("pos", [0,0,0])
            yawj = quat_to_yaw_deg(objs[j].get("rot", [0,0,0,1]))
            fx, fz = forward_vec_from_yaw(yawj)
            lx, lz = lateral_vec_from_yaw(yawj)
            front = signed_proj(float(pj[0]), float(pj[2]), float(pi[0]), float(pi[2]), fx, fz)
            lateral = abs(signed_proj(float(pj[0]), float(pj[2]), float(pi[0]), float(pi[2]), lx, lz))
            if front >= 0.2 and lateral <= 0.9:
                conf = clamp(0.7 + 0.2 * min(1.0, front), 0.55, 0.95)
                rels.append(Relation(i, j, "in_front_of", conf, f"{ci} is in front of {cj}", {"front_proj": round(front,4), "lateral": round(lateral,4)}))
    return rels


def extract_side_of(scene: Dict[str, Any], cats: List[str]) -> List[Relation]:
    objs = scene["objects"]
    rels = []
    for i, ci in enumerate(cats):
        for j, cj in enumerate(cats):
            if i == j:
                continue
            valid = (
                (ci == "nightstand" and cj == "bed") or
                (ci == "chair" and cj in {"table", "desk"})
            )
            if not valid:
                continue
            pi = objs[i].get("pos", [0,0,0])
            pj = objs[j].get("pos", [0,0,0])
            yawj = quat_to_yaw_deg(objs[j].get("rot", [0,0,0,1]))
            fx, fz = forward_vec_from_yaw(yawj)
            lx, lz = lateral_vec_from_yaw(yawj)
            forward = abs(signed_proj(float(pj[0]), float(pj[2]), float(pi[0]), float(pi[2]), fx, fz))
            lateral = abs(signed_proj(float(pj[0]), float(pj[2]), float(pi[0]), float(pi[2]), lx, lz))
            if lateral >= 0.2 and forward <= 0.7:
                conf = clamp(min(1.0, lateral) * 0.8 + 0.1, 0.55, 0.95)
                rels.append(Relation(i, j, "side_of", conf, f"{ci} is at the side of {cj}", {"forward": round(forward,4), "lateral": round(lateral,4)}))
    return rels


def extract_against_wall(scene: Dict[str, Any], cats: List[str]) -> List[Relation]:
    objs = scene["objects"]
    bounds_bottom = scene.get("bounds_bottom", [])
    room_cx, room_cz = room_center_xz(bounds_bottom)
    rels = []
    for i, ci in enumerate(cats):
        if ci not in {"storage", "bed", "desk"}:
            continue
        obj = objs[i]
        pos = obj.get("pos", [0,0,0])
        yaw = quat_to_yaw_deg(obj.get("rot", [0,0,0,1]))
        wall_dist = distance_to_nearest_wall_xz(bounds_bottom, pos)
        if wall_dist > 0.35:
            continue
        wall_yaw = nearest_wall_normal_yaw(bounds_bottom, pos)
        if wall_yaw is None:
            continue
        if ci == "bed":
            diff = axis_angle_diff_deg(yaw, wall_yaw)
            ok = diff <= 15.0
        elif ci == "desk":
            target_yaw = normalize_angle_deg(wall_yaw + 180.0)
            diff = angle_diff_deg(yaw, target_yaw)
            ok = diff <= 30.0
        else:
            target_yaw = normalize_angle_deg(math.degrees(math.atan2(room_cx - float(pos[0]), room_cz - float(pos[2]))))
            diff = angle_diff_deg(yaw, target_yaw)
            ok = diff <= 45.0
        if ok:
            conf = clamp(1.0 - wall_dist / 0.35, 0.65, 0.98)
            rels.append(Relation(i, None, "against_wall", conf, f"{ci} is close to and oriented against a wall", {"wall_dist": round(wall_dist,4), "yaw_diff": round(diff,4)}))
    return rels


def extract_parallel(scene: Dict[str, Any], cats: List[str]) -> List[Relation]:
    objs = scene["objects"]
    bounds_bottom = scene.get("bounds_bottom", [])
    rels = []
    for i, ci in enumerate(cats):
        if ci not in {"table", "storage", "desk", "bed"}:
            continue
        obj = objs[i]
        pos = obj.get("pos", [0,0,0])
        yaw = quat_to_yaw_deg(obj.get("rot", [0,0,0,1]))
        target = nearest_parallel_wall_yaw(bounds_bottom, pos, yaw)
        if target is None:
            continue
        if ci == "table":
            diff = axis_angle_diff_deg(yaw, target)
            ok = diff <= 10.0
        elif ci in {"storage", "desk"}:
            diff = angle_diff_deg(yaw, target)
            ok = diff <= 20.0
        else:
            diff = axis_angle_diff_deg(yaw, target)
            ok = diff <= 20.0
        if ok:
            conf = clamp(1.0 - diff / 20.0, 0.6, 0.97)
            rels.append(Relation(i, None, "parallel", conf, f"{ci} is aligned parallel to the nearest wall", {"yaw_diff": round(diff,4), "target_yaw": round(target,4)}))
    return rels


def extract_distance_band(scene: Dict[str, Any], cats: List[str]) -> List[Relation]:
    objs = scene["objects"]
    rels = []
    kept = set()
    for i, ci in enumerate(cats):
        for j, cj in enumerate(cats):
            if i == j:
                continue
            valid = (ci == "chair" and cj == "chair") or (ci == "storage" and cj in {"table", "desk"})
            if not valid:
                continue
            key = tuple(sorted((i, j)))
            if key in kept:
                continue
            d = xz_dist(objs[i], objs[j])
            c = 0.5 * (obj_diag_size_xz(objs[i]) + obj_diag_size_xz(objs[j])) + 0.4
            lo, hi = max(0.3, c - 0.5), c + 0.8
            if lo <= d <= hi:
                conf = clamp(1.0 - abs(d - c) / max(0.4, hi - lo), 0.55, 0.95)
                rels.append(Relation(i, j, "distance_band", conf, f"{ci} and {cj} are within a reasonable distance band", {"dist": round(d,4), "lo": round(lo,4), "hi": round(hi,4)}))
                kept.add(key)
    return rels


def dedup_relations(rels: Iterable[Relation]) -> List[Relation]:
    best = {}
    for r in rels:
        key = (r.src_idx, r.tgt_idx, r.type)
        old = best.get(key)
        if old is None or r.confidence > old.confidence:
            best[key] = r
    return list(best.values())


def prune_relations(rels: List[Relation], keep_types: set, max_per_src: int = 3) -> List[Relation]:
    rels = [r for r in rels if r.type in keep_types]
    rels = dedup_relations(rels)
    grouped = defaultdict(list)
    for r in rels:
        grouped[r.src_idx].append(r)
    kept = []
    for src, items in grouped.items():
        items.sort(key=lambda r: (-REL_PRIORITY.get(r.type, 0), -r.confidence))
        selected = []
        seen_targets = set()
        for r in items:
            pair_key = (r.tgt_idx, r.type)
            if pair_key in seen_targets:
                continue
            selected.append(r)
            seen_targets.add(pair_key)
            if len(selected) >= max_per_src:
                break
        has_strong = {(x.tgt_idx, x.type) for x in selected}
        compacted = []
        for r in selected:
            if r.type == "distance_band" and ((r.tgt_idx, "near") in has_strong or (r.tgt_idx, "facing") in has_strong):
                continue
            compacted.append(r)
        kept.extend(compacted)
    kept.sort(key=lambda r: (r.src_idx, -(REL_PRIORITY.get(r.type, 0)), -r.confidence))
    return kept


def extract_relations_for_scene(scene: Dict[str, Any], jid2cat: Optional[Dict[str, str]] = None, keep_all_9: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cats = resolve_categories(scene, jid2cat=jid2cat)
    rels = []
    rels.extend(extract_near(scene, cats))
    rels.extend(extract_facing(scene, cats))
    rels.extend(extract_facing_pair(scene, cats))
    rels.extend(extract_centered_with(scene, cats))
    rels.extend(extract_against_wall(scene, cats))
    rels.extend(extract_parallel(scene, cats))
    if keep_all_9:
        rels.extend(extract_distance_band(scene, cats))
        rels.extend(extract_in_front_of(scene, cats))
        rels.extend(extract_side_of(scene, cats))
    keep_types = ALL_REL_TYPES if keep_all_9 else DEFAULT_REL_TYPES
    rels = prune_relations(rels, keep_types=keep_types, max_per_src=3)
    elapsed = time.perf_counter() - t0
    return {
        "room_id": scene.get("room_id"),
        "room_type": scene.get("room_type"),
        "num_objects": len(scene.get("objects", [])),
        "categories": cats,
        "num_relations": len(rels),
        "relations": [r.to_json() for r in rels],
        "extract_time_sec": round(elapsed, 4),
    }


def iter_scene_files(inp: Path) -> List[Path]:
    if inp.is_file():
        return [inp]
    files = []
    for p in sorted(inp.rglob("*.json")):
        if p.name.endswith(".relations.json") or p.name == "relation_extraction_summary.json":
            continue
        files.append(p)
    return files


def maybe_load_jid2cat(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if path is None:
        return None
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("--jid2cat must be a JSON object mapping jid -> category string")
    out = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(description="Extract spatial relations from clean scene JSONs.")
    parser.add_argument("--input", required=True, help="Scene JSON file or directory.")
    parser.add_argument("--output_dir", required=True, help="Directory to save relation JSON outputs.")
    parser.add_argument("--jid2cat", default=None, help="Optional JSON file mapping jid -> dataset category.")
    parser.add_argument("--keep_all_9", action="store_true", help="Extract all 9 relations instead of default 6.")
    args = parser.parse_args()

    inp = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not inp.exists():
        raise FileNotFoundError(f"Input path does not exist: {inp}")

    jid2cat = maybe_load_jid2cat(Path(args.jid2cat).expanduser().resolve()) if args.jid2cat else None
    files = iter_scene_files(inp)
    if not files:
        raise RuntimeError(f"No JSON scene files found under: {inp}")

    summary = []
    total_start = time.perf_counter()
    pbar = tqdm(files, desc="Extracting relations", unit="scene")
    for fp in pbar:
        scene_start = time.perf_counter()
        try:
            scene = load_json(fp)
            result = extract_relations_for_scene(scene, jid2cat=jid2cat, keep_all_9=args.keep_all_9)
            if inp.is_file():
                out_path = out_dir / (fp.stem + ".relations.json")
            else:
                rel_path = fp.relative_to(inp)
                out_path = out_dir / rel_path.parent / (fp.stem + ".relations.json")
            dump_json(out_path, result)
            elapsed = time.perf_counter() - scene_start
            summary.append({
                "file": str(fp),
                "output": str(out_path),
                "room_id": result.get("room_id"),
                "room_type": result.get("room_type"),
                "num_objects": result.get("num_objects"),
                "num_relations": result.get("num_relations"),
                "extract_time_sec": round(elapsed, 4),
                "status": "ok",
            })
            ok_items = [x for x in summary if x["status"] == "ok"]
            avg_t = sum(x["extract_time_sec"] for x in ok_items) / max(1, len(ok_items))
            pbar.set_postfix(avg_sec=f"{avg_t:.3f}", rels=result.get("num_relations", 0))
        except Exception as e:
            elapsed = time.perf_counter() - scene_start
            summary.append({
                "file": str(fp),
                "extract_time_sec": round(elapsed, 4),
                "status": "error",
                "error": repr(e),
            })
            pbar.set_postfix(error=fp.name)

    total_elapsed = time.perf_counter() - total_start
    ok_items = [x for x in summary if x["status"] == "ok"]
    final_summary = {
        "num_scenes": len(files),
        "num_success": len(ok_items),
        "num_error": len(files) - len(ok_items),
        "keep_all_9": bool(args.keep_all_9),
        "total_time_sec": round(total_elapsed, 4),
        "avg_time_sec": round(sum(x["extract_time_sec"] for x in ok_items) / max(1, len(ok_items)), 4),
        "details": summary,
    }
    dump_json(out_dir / "relation_extraction_summary.json", final_summary)
    print(json.dumps({
        "num_scenes": final_summary["num_scenes"],
        "num_success": final_summary["num_success"],
        "num_error": final_summary["num_error"],
        "total_time_sec": final_summary["total_time_sec"],
        "avg_time_sec": final_summary["avg_time_sec"],
        "output_dir": str(out_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
