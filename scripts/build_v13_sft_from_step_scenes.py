#!/usr/bin/env python3
"""Build V13 SFT JSONL from directories of plain scene-step JSON files.

This version supports the trajectory format:

    /path/to/traj/step_00.json
    /path/to/traj/step_01.json
    ...
    /path/to/traj/step_5.json

where every step_*.json is a full scene JSON with room bounds and objects.
It can also scan a root directory containing many such trajectory subfolders.

The output target is a V13 SceneRepairPlan JSON object:
{
  "version": "scene_repair_plan_v1",
  "diagnosis": [...],
  "actions": [...],
  "global_strategy": "..."
}

Default behavior is for scene-repair / perturbation-recovery SFT:
- compare consecutive states;
- match common objects;
- compute dx, dz, dyaw from before -> after;
- output only move/rotate actions executable by init_gpt_image_describe_v13.py.

If your trajectory is true scene-growth where new objects are added at each step,
those added objects are NOT emitted by default because the current V13 repair
runtime executes existing-object pose edits, not object creation. Use
--emit-add-object-labels only if you are training a separate add-object policy.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "scene_repair_plan_v1"
STEP_RE = re.compile(r"step[_-]?(\d+)\.json$", re.IGNORECASE)


# -----------------------------
# I/O
# -----------------------------

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def step_number(path: Path) -> int:
    m = STEP_RE.search(path.name)
    if not m:
        return 10**12
    return int(m.group(1))


def is_scene_json(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("objects"), list) and (
        "bounds_bottom" in obj or "bounds_top" in obj or "room_type" in obj
    )


# -----------------------------
# Geometry helpers
# -----------------------------

def yaw_from_quaternion(rot: Sequence[float]) -> float:
    """Scene convention: quaternion [x, y, z, w], yaw around Y axis."""
    if not isinstance(rot, Sequence) or len(rot) < 4:
        return 0.0
    x, y, z, w = [float(v) for v in rot[:4]]
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp)) % 360.0


def angle_diff_signed(a: float, b: float) -> float:
    """Return b - a in [-180, 180]."""
    return ((float(b) - float(a) + 180.0) % 360.0) - 180.0


def safe_pos(obj: Dict[str, Any]) -> Optional[List[float]]:
    pos = obj.get("pos")
    if isinstance(pos, list) and len(pos) >= 3:
        try:
            return [float(pos[0]), float(pos[1]), float(pos[2])]
        except Exception:
            return None
    return None


def safe_size(obj: Dict[str, Any]) -> Optional[List[float]]:
    size = obj.get("size") or obj.get("sampled_asset_size")
    if isinstance(size, list) and len(size) >= 3:
        try:
            return [float(size[0]), float(size[1]), float(size[2])]
        except Exception:
            return None
    return None


def object_text(obj: Dict[str, Any], max_len: int = 180) -> str:
    return str(obj.get("desc") or obj.get("prompt") or obj.get("sampled_asset_desc") or "")[:max_len]


def object_jid(obj: Dict[str, Any]) -> str:
    return str(obj.get("sampled_asset_jid") or obj.get("jid") or obj.get("sampled_jid") or "")


# -----------------------------
# Scene compacting for SFT input
# -----------------------------

def build_object_index(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for i, obj in enumerate(scene.get("objects", [])):
        pos = safe_pos(obj)
        size = safe_size(obj)
        rot = obj.get("rot", [0, 0, 0, 1])
        jid = object_jid(obj)
        out.append({
            "idx": i,
            "match_key": obj.get("match_key", ""),
            "jid_prefix": jid[:8],
            "uuid": obj.get("uuid", ""),
            "desc": object_text(obj),
            "pos": [round(v, 4) for v in pos] if pos is not None else None,
            "yaw_deg": round(yaw_from_quaternion(rot), 3),
            "size": [round(v, 4) for v in size] if size is not None else None,
        })
    return out


def compact_scene_for_input(scene: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "room_type": scene.get("room_type"),
        "room_id": scene.get("room_id"),
        "bounds_bottom": scene.get("bounds_bottom"),
        "bounds_top": scene.get("bounds_top"),
        "objects": build_object_index(scene),
    }


# -----------------------------
# Object matching
# -----------------------------

UNIQUE_ID_KEYS = (
    "match_key",
    "uuid",
    "instance_id",
    "object_id",
    "id",
    # jid can repeat for duplicated chairs; only used if unique in both scenes.
    "sampled_asset_jid",
    "jid",
    "sampled_jid",
)


def _value(obj: Dict[str, Any], key: str) -> str:
    v = obj.get(key)
    return v.strip() if isinstance(v, str) and v.strip() else ""


def match_objects(before: Dict[str, Any], after: Dict[str, Any]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match objects from before to after.

    Returns:
        pairs: list of (before_idx, after_idx)
        removed_before_indices
        added_after_indices
    """
    b_objs = before.get("objects", [])
    a_objs = after.get("objects", [])
    used_b = set()
    used_a = set()
    pairs: List[Tuple[int, int]] = []

    # 1) Match by keys that are unique in both scenes.
    for key in UNIQUE_ID_KEYS:
        b_vals = [_value(o, key) for o in b_objs]
        a_vals = [_value(o, key) for o in a_objs]
        cb = Counter(v for v in b_vals if v)
        ca = Counter(v for v in a_vals if v)
        a_map = {v: i for i, v in enumerate(a_vals) if v and ca[v] == 1}
        for i, v in enumerate(b_vals):
            if i in used_b or not v or cb[v] != 1:
                continue
            j = a_map.get(v)
            if j is not None and j not in used_a:
                pairs.append((i, j))
                used_b.add(i)
                used_a.add(j)

    # 2) If object count is unchanged, object order in your saved scene JSON is usually stable.
    # This is important for duplicated assets, e.g. multiple chairs with identical jid.
    if len(b_objs) == len(a_objs):
        for i in range(len(b_objs)):
            if i not in used_b and i not in used_a:
                pairs.append((i, i))
                used_b.add(i)
                used_a.add(i)

    # 3) Greedy fallback for changed object counts: same jid/desc and nearest position.
    remaining_b = [i for i in range(len(b_objs)) if i not in used_b]
    remaining_a = [j for j in range(len(a_objs)) if j not in used_a]
    scored: List[Tuple[float, int, int]] = []
    for i in remaining_b:
        bi = b_objs[i]
        bp = safe_pos(bi)
        bjid = object_jid(bi)
        bdesc = object_text(bi, 80).lower()
        for j in remaining_a:
            aj = a_objs[j]
            ap = safe_pos(aj)
            ajid = object_jid(aj)
            adesc = object_text(aj, 80).lower()
            same_sem = False
            if bjid and ajid and bjid == ajid:
                same_sem = True
            elif bdesc and adesc and bdesc == adesc:
                same_sem = True
            if not same_sem:
                continue
            dist = 999.0
            if bp is not None and ap is not None:
                dist = math.hypot(bp[0] - ap[0], bp[2] - ap[2])
            scored.append((dist, i, j))
    scored.sort(key=lambda x: x[0])
    for _, i, j in scored:
        if i in used_b or j in used_a:
            continue
        pairs.append((i, j))
        used_b.add(i)
        used_a.add(j)

    pairs.sort(key=lambda p: p[0])
    removed = [i for i in range(len(b_objs)) if i not in used_b]
    added = [j for j in range(len(a_objs)) if j not in used_a]
    return pairs, removed, added


# -----------------------------
# Transition -> SceneRepairPlan
# -----------------------------

def transition_to_actions(
    before: Dict[str, Any],
    after: Dict[str, Any],
    *,
    min_move: float = 0.03,
    min_yaw: float = 3.0,
    max_actions: int = 8,
    emit_add_object_labels: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    pairs, removed, added = match_objects(before, after)
    b_objs = before.get("objects", [])
    a_objs = after.get("objects", [])

    for i, j in pairs:
        b = b_objs[i]
        a = a_objs[j]
        bp = safe_pos(b)
        ap = safe_pos(a)
        if bp is None or ap is None:
            continue
        dx = ap[0] - bp[0]
        dz = ap[2] - bp[2]
        byaw = yaw_from_quaternion(b.get("rot", [0, 0, 0, 1]))
        ayaw = yaw_from_quaternion(a.get("rot", [0, 0, 0, 1]))
        dyaw = angle_diff_signed(byaw, ayaw)
        moved = math.hypot(dx, dz) >= min_move
        rotated = abs(dyaw) >= min_yaw
        if not moved and not rotated:
            continue

        jid = object_jid(b)
        op = "move_delta" if moved else "rotate_delta"
        actions.append({
            "target_idx": i,
            "target_jid_prefix": jid[:8],
            "op": op,
            "dx": round(dx, 4),
            "dz": round(dz, 4),
            "dyaw": round(dyaw, 3),
            "yaw_abs": round(ayaw, 3) if rotated else None,
            "anchor_idx": None,
            "reason_type": "trajectory_recovery",
            "rationale": "Recover this object toward the next clean trajectory state.",
            "confidence": 1.0,
        })

    # Optional labels for a separate generation/add-object policy.
    # Current init_gpt_image_describe_v13.py will not execute these by default.
    if emit_add_object_labels:
        for j in added:
            a = a_objs[j]
            ap = safe_pos(a)
            size = safe_size(a)
            jid = object_jid(a)
            actions.append({
                "target_idx": None,
                "target_jid_prefix": jid[:8],
                "op": "add_object",
                "object": {
                    "desc": object_text(a, 260),
                    "size": [round(v, 4) for v in size] if size is not None else a.get("size"),
                    "pos": [round(v, 4) for v in ap] if ap is not None else a.get("pos"),
                    "rot": a.get("rot"),
                    "jid": jid,
                    "match_key": a.get("match_key", ""),
                },
                "reason_type": "scene_growth",
                "rationale": "Add the new object introduced by the next trajectory state.",
                "confidence": 1.0,
            })

    # Prefer larger corrections when too many changed objects.
    def action_mag(a: Dict[str, Any]) -> float:
        if a.get("op") == "add_object":
            return 10.0
        return math.hypot(float(a.get("dx", 0.0)), float(a.get("dz", 0.0))) + abs(float(a.get("dyaw", 0.0))) / 180.0

    actions.sort(key=action_mag, reverse=True)
    if max_actions > 0:
        actions = actions[:max_actions]

    stats = {
        "matched": len(pairs),
        "removed": len(removed),
        "added": len(added),
        "num_actions": len(actions),
        "object_count_before": len(b_objs),
        "object_count_after": len(a_objs),
    }
    return actions, stats


def make_plan(actions: List[Dict[str, Any]], stats: Dict[str, Any]) -> Dict[str, Any]:
    diagnosis = []
    for a in actions:
        diagnosis.append({
            "issue_type": "scene_growth" if a.get("op") == "add_object" else "perturbation_recovery",
            "target_idx": a.get("target_idx"),
            "related_idx": a.get("anchor_idx"),
            "evidence": "Object pose or object set differs from the next clean trajectory state.",
            "severity": 1.0,
        })
    if not diagnosis and (stats.get("added", 0) or stats.get("removed", 0)):
        diagnosis.append({
            "issue_type": "object_set_change_ignored",
            "target_idx": None,
            "related_idx": None,
            "evidence": "The next state changes the object set, but add/remove labels were not emitted for the V13 repair policy.",
            "severity": 0.5,
        })
    return {
        "version": SCHEMA_VERSION,
        "diagnosis": diagnosis,
        "actions": actions,
        "global_strategy": "Recover the current scene toward the next trajectory state using minimal atomic pose corrections.",
    }


# -----------------------------
# Input discovery
# -----------------------------

def load_plain_step_sequence(traj_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    files = sorted([p for p in traj_dir.glob("step*.json") if STEP_RE.search(p.name)], key=step_number)
    seq: List[Tuple[Path, Dict[str, Any]]] = []
    for p in files:
        try:
            obj = read_json(p)
        except Exception as exc:
            print(f"[warn] skip unreadable {p}: {exc}")
            continue
        if is_scene_json(obj):
            seq.append((p, obj))
    return seq


def discover_trajectory_dirs(root: Path) -> List[Path]:
    """Find dirs that directly contain step_*.json scene files."""
    if not root.is_dir():
        return []
    if load_plain_step_sequence(root):
        return [root]
    candidates = set()
    for p in root.rglob("step*.json"):
        if STEP_RE.search(p.name):
            candidates.add(p.parent)
    dirs = []
    for d in sorted(candidates):
        if len(load_plain_step_sequence(d)) >= 2:
            dirs.append(d)
    return dirs


def iter_plain_step_transitions(root: Path) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    for traj_dir in discover_trajectory_dirs(root):
        seq = load_plain_step_sequence(traj_dir)
        if len(seq) < 2:
            continue
        for k in range(len(seq) - 1):
            p0, s0 = seq[k]
            p1, s1 = seq[k + 1]
            meta = {
                "trajectory_dir": str(traj_dir),
                "step_from": p0.name,
                "step_to": p1.name,
                "transition_index": k,
            }
            yield s0, s1, meta


# -----------------------------
# SFT row formatting
# -----------------------------

def make_sft_row(
    before: Dict[str, Any],
    plan: Dict[str, Any],
    meta: Dict[str, Any],
    *,
    include_full_scene: bool,
    messages_format: bool,
) -> Dict[str, Any]:
    input_obj = {
        "task": "Given the current 3D indoor scene, output a SceneRepairPlan JSON with atomic actions that move or rotate existing objects toward a better next state.",
        "required_schema_version": SCHEMA_VERSION,
        "trajectory_meta": meta,
        "scene": before if include_full_scene else compact_scene_for_input(before),
    }
    output = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
    if messages_format:
        return {
            "messages": [
                {"role": "system", "content": "You are a 3D scene repair policy. Return only valid SceneRepairPlan JSON."},
                {"role": "user", "content": json.dumps(input_obj, ensure_ascii=False, separators=(",", ":"))},
                {"role": "assistant", "content": output},
            ],
            "metadata": meta,
        }
    return {"input": input_obj, "output": plan, "metadata": meta}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="one trajectory dir or a root containing many trajectory dirs")
    ap.add_argument("--output", required=True, type=Path, help="output JSONL path")
    ap.add_argument("--include-full-scene", action="store_true", help="include full scene JSON instead of compact object index")
    ap.add_argument("--io-format", choices=["messages", "input_output"], default="messages")
    ap.add_argument("--min-move", type=float, default=0.03, help="ignore dx/dz movement smaller than this meter threshold")
    ap.add_argument("--min-yaw", type=float, default=3.0, help="ignore yaw change smaller than this degree threshold")
    ap.add_argument("--max-actions", type=int, default=8, help="max actions per transition; <=0 keeps all")
    ap.add_argument("--emit-add-object-labels", action="store_true", help="emit add_object labels for object-set growth; not executable by current V13 repair runtime")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    total_transitions = 0
    skipped_no_actions = 0
    aggregate = defaultdict(int)

    for before, after, meta in iter_plain_step_transitions(args.input):
        total_transitions += 1
        actions, stats = transition_to_actions(
            before,
            after,
            min_move=args.min_move,
            min_yaw=args.min_yaw,
            max_actions=args.max_actions,
            emit_add_object_labels=args.emit_add_object_labels,
        )
        for k, v in stats.items():
            if isinstance(v, int):
                aggregate[k] += v
        meta = dict(meta)
        meta.update(stats)
        if not actions:
            skipped_no_actions += 1
            print(f"[skip] no pose actions: {meta['trajectory_dir']} {meta['step_from']} -> {meta['step_to']} stats={stats}")
            continue
        plan = make_plan(actions, stats)
        rows.append(make_sft_row(before, plan, meta, include_full_scene=args.include_full_scene, messages_format=args.io_format == "messages"))

    n = write_jsonl(args.output, rows)
    print(f"wrote {n} SFT rows to {args.output}")
    print(f"total_transitions={total_transitions} skipped_no_actions={skipped_no_actions}")
    if aggregate:
        print("aggregate_stats=" + json.dumps(dict(aggregate), ensure_ascii=False, sort_keys=True))
    if aggregate.get("added", 0) and not args.emit_add_object_labels:
        print("[note] object additions were detected but ignored. Current V13 repair runtime supports existing-object move/rotate actions. Use --emit-add-object-labels only for a separate add-object/growth policy.")


if __name__ == "__main__":
    main()
