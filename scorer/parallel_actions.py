from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

from src.eval import eval_scene_before_after_with_delta


# -----------------------------
# Discrete action space (STRICT)
# -----------------------------

MOVE_SIZES: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
YAW_SIZES: Tuple[float, ...] = (2.0, 5.0, 10.0, 15.0)

# numeric tolerance for matching floats like 0.1 vs 0.10
_DISCRETE_EPS = 1e-6


def _is_close_to_any(v: float, allowed: Tuple[float, ...], eps: float = _DISCRETE_EPS) -> bool:
    av = abs(float(v))
    for a in allowed:
        if abs(av - float(a)) <= eps:
            return True
    return False


# -----------------------------
# Schema
# -----------------------------

@dataclass(frozen=True)
class Action:
    jid: str
    op: str  # "translate" | "rotate_yaw" | "noop"
    delta_token: str  # "MOVE_(+0.10,+0.00,-0.05)" | "YAW_(+5)" | "noop"


@dataclass(frozen=True)
class StepActions:
    step: int
    actions: List[Action]
    default_op: str = "noop"


def validate_step_actions(payload: Dict[str, Any]) -> StepActions:
    """
    - 校验顶层 schema
    - 去重 jid（保留最后一次出现）
    - 不在离散空间的动作不会在这里抛错，但会在 parse_delta_token(strict) 时被丢弃为 noop
    """
    if not isinstance(payload, dict):
        raise TypeError("actions payload must be a dict")
    if "step" not in payload or not isinstance(payload["step"], int):
        raise ValueError("payload.step must be int")
    if "actions" not in payload or not isinstance(payload["actions"], list):
        raise ValueError("payload.actions must be list")
    default_op = payload.get("default_op", "noop")
    if default_op != "noop":
        raise ValueError("payload.default_op must be 'noop'")

    raw_actions: List[Dict[str, Any]] = []
    for a in payload["actions"]:
        if not isinstance(a, dict):
            raise ValueError("each action must be dict")
        raw_actions.append(a)

    # dedup by jid: keep last
    dedup_rev: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for a in reversed(raw_actions):
        jid = a.get("jid")
        if not isinstance(jid, str) or not jid:
            continue
        if jid in seen:
            continue
        seen.add(jid)
        dedup_rev.append(a)
    dedup = list(reversed(dedup_rev))

    actions: List[Action] = []
    for a in dedup:
        jid = a.get("jid")
        op = a.get("op")
        delta_token = a.get("delta_token")
        if not isinstance(jid, str) or not jid:
            raise ValueError("action.jid must be non-empty str")
        if op not in ("translate", "rotate_yaw", "noop"):
            raise ValueError(f"unsupported action.op: {op}")
        if not isinstance(delta_token, str) or not delta_token:
            raise ValueError("action.delta_token must be non-empty str")
        actions.append(Action(jid=jid, op=op, delta_token=delta_token))

    return StepActions(step=payload["step"], actions=actions, default_op=default_op)


# -----------------------------
# Discrete action parsing (STRICT)
# -----------------------------

_MOVE_RE = re.compile(
    r"^MOVE_\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)$"
)
_YAW_RE = re.compile(r"^YAW_\(\s*([+-]?\d+(?:\.\d+)?)\s*\)$")


def parse_delta_token(action: Action) -> Tuple[Tuple[float, float, float], float]:
    """
    STRICT discrete parsing.
    If invalid wrt discrete space, returns noop: ((0,0,0), 0).
    """
    if action.op == "noop" or action.delta_token == "noop":
        return (0.0, 0.0, 0.0), 0.0

    if action.op == "translate":
        m = _MOVE_RE.match(action.delta_token)
        if not m:
            return (0.0, 0.0, 0.0), 0.0

        dx, dy, dz = float(m.group(1)), float(m.group(2)), float(m.group(3))

        # Rule 1: exactly one axis non-zero (within eps)
        comps = [dx, dy, dz]
        nz = [c for c in comps if abs(c) > _DISCRETE_EPS]
        if len(nz) != 1:
            return (0.0, 0.0, 0.0), 0.0

        step = nz[0]
        # Rule 2: magnitude must be in MOVE_SIZES
        if not _is_close_to_any(step, MOVE_SIZES):
            return (0.0, 0.0, 0.0), 0.0

        # normalize to exact allowed magnitude (keep sign); choose closest allowed
        mag = abs(step)
        closest = min(MOVE_SIZES, key=lambda a: abs(a - mag))
        step_norm = math.copysign(float(closest), step)

        # snap the output: only the non-zero axis keeps snapped step
        if abs(dx) > _DISCRETE_EPS:
            return (step_norm, 0.0, 0.0), 0.0
        if abs(dy) > _DISCRETE_EPS:
            return (0.0, step_norm, 0.0), 0.0
        return (0.0, 0.0, step_norm), 0.0

    if action.op == "rotate_yaw":
        m = _YAW_RE.match(action.delta_token)
        if not m:
            return (0.0, 0.0, 0.0), 0.0

        dyaw = float(m.group(1))
        if abs(dyaw) <= _DISCRETE_EPS:
            return (0.0, 0.0, 0.0), 0.0

        if not _is_close_to_any(dyaw, YAW_SIZES):
            return (0.0, 0.0, 0.0), 0.0

        mag = abs(dyaw)
        closest = min(YAW_SIZES, key=lambda a: abs(a - mag))
        dyaw_norm = math.copysign(float(closest), dyaw)
        return (0.0, 0.0, 0.0), dyaw_norm

    return (0.0, 0.0, 0.0), 0.0


# -----------------------------
# Scene helpers
# -----------------------------

def _jid_to_obj_index(scene: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, obj in enumerate(scene.get("objects", [])):
        jid = obj.get("jid") or obj.get("sampled_jid") or obj.get("sampled_asset_jid")
        if isinstance(jid, str):
            out[jid] = i
    return out


def _get_obj_pos(obj: Dict[str, Any]) -> List[float]:
    for k in ("pos", "position", "translation", "t"):
        v = obj.get(k)
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return [float(v[0]), float(v[1]), float(v[2])]
    raise KeyError("object position not found (expected one of pos/position/translation/t)")


def _set_obj_pos(obj: Dict[str, Any], pos: List[float]) -> None:
    for k in ("pos", "position", "translation", "t"):
        if k in obj:
            obj[k] = [float(pos[0]), float(pos[1]), float(pos[2])]
            return
    obj["pos"] = [float(pos[0]), float(pos[1]), float(pos[2])]


def _wrap_yaw_deg(y: float) -> float:
    return (y + 180.0) % 360.0 - 180.0


def _quat_xyzw_to_yaw_deg(q: List[float]) -> float:
    """
    Quaternion [x,y,z,w] -> yaw deg about +Y.
    Assumes objects are upright (pitch/roll ~0). If not, yaw extraction is approximate.
    """
    if not (isinstance(q, (list, tuple)) and len(q) == 4):
        raise ValueError("quat must be length-4 [x,y,z,w]")
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])

    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n > 1e-12:
        x, y, z, w = x / n, y / n, z / n, w / n

    yaw = math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + x * x))
    return float(_wrap_yaw_deg(math.degrees(yaw)))


def _yaw_deg_to_quat_xyzw(yaw_deg: float) -> List[float]:
    yaw_rad = math.radians(float(yaw_deg))
    half = 0.5 * yaw_rad
    cy = math.cos(half)
    sy = math.sin(half)
    return [0.0, float(sy), 0.0, float(cy)]


def _get_obj_yaw(obj: Dict[str, Any]) -> float:
    for k in ("yaw", "rot_yaw", "rotation_yaw"):
        v = obj.get(k)
        if isinstance(v, (int, float)):
            return float(v)

    rot = obj.get("rot")
    if isinstance(rot, (list, tuple)) and len(rot) == 4:
        try:
            return _quat_xyzw_to_yaw_deg(list(rot))
        except Exception:
            pass

    return 0.0


def _set_obj_yaw(obj: Dict[str, Any], yaw_deg: float) -> None:
    """
    Preferred: if obj has 'rot' quaternion, write yaw back into 'rot' (pure yaw quat).
    Fallback: write obj['yaw'].
    """
    yaw_deg = float(_wrap_yaw_deg(yaw_deg))

    if "rot" in obj and isinstance(obj.get("rot"), (list, tuple)) and len(obj["rot"]) == 4:
        obj["rot"] = _yaw_deg_to_quat_xyzw(yaw_deg)
        return

    obj["yaw"] = yaw_deg


# -----------------------------
# Apply (pure)
# -----------------------------

def apply_actions_to_scene(scene: Dict[str, Any], step_actions: StepActions) -> Dict[str, Any]:
    """
    Pure apply (no eval/backoff). Returns a deep-copied new scene.

    NOTE: parse_delta_token is STRICT; invalid tokens are treated as noop.
    """
    scene_new = copy.deepcopy(scene)
    idx = _jid_to_obj_index(scene_new)
    action_by_jid: Dict[str, Action] = {a.jid: a for a in step_actions.actions}

    for jid, i in idx.items():
        a = action_by_jid.get(jid)
        if a is None:
            continue

        (dx, dy, dz), dyaw = parse_delta_token(a)
        if abs(dx) <= _DISCRETE_EPS and abs(dy) <= _DISCRETE_EPS and abs(dz) <= _DISCRETE_EPS and abs(dyaw) <= _DISCRETE_EPS:
            continue

        obj = scene_new["objects"][i]
        if a.op == "translate":
            p = _get_obj_pos(obj)
            _set_obj_pos(obj, [p[0] + dx, p[1] + dy, p[2] + dz])
        elif a.op == "rotate_yaw":
            yaw = _get_obj_yaw(obj)
            _set_obj_yaw(obj, yaw + dyaw)

    return scene_new


# -----------------------------
# Core: propose -> eval -> backoff (eval-driven)
# -----------------------------

def apply_actions_parallel_with_eval(
    scene_t: Dict[str, Any],
    step_actions: StepActions,
    *,
    max_backoff_iters: int = 50,
    only_backoff_if_worse: bool = True,
    priority_by_remaining: Optional[Dict[str, float]] = None,
    debug: bool = False,
    strict_monotonic_wrt_old: bool = False,
    monotonic_eps: float = 1e-12,
    monotonic_require_oob_mbl_non_increase: bool = False,
) -> Dict[str, Any]:
    """
    并行拟执行 -> 用 eval 指标(OOB/MBL/PBL)评估 -> 逐物体回退动作直至更好/有效。

    strict_monotonic_wrt_old=True 时：
    - 返回结果必须满足相对 scene_old 的 total_pbl_loss 不上升（<= old + eps）
    - 可选：同时要求 oob/mbl 不上升
    - 否则返回 scene_old（本 step 视为 noop），从而保证 step-to-step 不变差
    """
    scene_old = copy.deepcopy(scene_t)

    action_by_jid: Dict[str, Action] = {a.jid: a for a in step_actions.actions}
    idx = _jid_to_obj_index(scene_old)

    def prio(jid: str) -> float:
        if priority_by_remaining and jid in priority_by_remaining:
            return float(priority_by_remaining[jid])
        return 0.0

    # baseline metrics (old vs old)
    base = eval_scene_before_after_with_delta(scene_old, scene_old, is_debug=False)
    base_valid = bool(base.get("is_valid_scene_pbl"))
    base_pbl = float(base.get("total_pbl_loss", 1e18))
    base_oob = float(base.get("total_oob_loss", 1e18))
    base_mbl = float(base.get("total_mbl_loss", 1e18))

    def ok_monotonic(d: Dict[str, Any]) -> bool:
        if not strict_monotonic_wrt_old:
            return True
        cand_valid = bool(d.get("is_valid_scene_pbl"))
        cand_pbl = float(d.get("total_pbl_loss", 1e18))
        cand_oob = float(d.get("total_oob_loss", 1e18))
        cand_mbl = float(d.get("total_mbl_loss", 1e18))

        # do not allow valid->invalid regression
        if base_valid and (not cand_valid):
            return False
        # must not increase pbl loss
        if cand_pbl > base_pbl + float(monotonic_eps):
            return False
        if monotonic_require_oob_mbl_non_increase:
            if cand_oob > base_oob + float(monotonic_eps):
                return False
            if cand_mbl > base_mbl + float(monotonic_eps):
                return False
        return True

    # propose
    scene_prop = apply_actions_to_scene(scene_old, step_actions)
    delta = eval_scene_before_after_with_delta(scene_old, scene_prop, is_debug=False)

    if debug:
        print(
            "baseline:",
            {
                "valid": base_valid,
                "total_pbl_loss": base_pbl,
                "total_oob_loss": base_oob,
                "total_mbl_loss": base_mbl,
            },
        )
        print(
            "proposed:",
            {
                "valid": delta.get("is_valid_scene_pbl"),
                "total_pbl_loss": delta.get("total_pbl_loss"),
                "total_oob_loss": delta.get("total_oob_loss"),
                "total_mbl_loss": delta.get("total_mbl_loss"),
                "delta_pbl_loss": delta.get("delta_pbl_loss"),
                "delta_oob_loss": delta.get("delta_oob_loss"),
                "delta_mbl_loss": delta.get("delta_mbl_loss"),
            },
        )

    if bool(delta.get("is_valid_scene_pbl")) and (not only_backoff_if_worse or float(delta.get("delta_pbl_loss", 0.0)) <= 0.0):
        if ok_monotonic(delta):
            return scene_prop
        return scene_old

    # backoff
    active_jids: Set[str] = set(action_by_jid.keys())

    def sorted_active_low_first() -> List[str]:
        return sorted(active_jids, key=lambda j: (prio(j), j))

    scene_cur = scene_prop
    for _ in range(max_backoff_iters):
        if not active_jids:
            break

        loser = sorted_active_low_first()[0]
        active_jids.remove(loser)

        if loser not in idx:
            continue

        cand = copy.deepcopy(scene_cur)
        cand["objects"][idx[loser]] = copy.deepcopy(scene_old["objects"][idx[loser]])

        d = eval_scene_before_after_with_delta(scene_old, cand, is_debug=False)

        cur_valid = bool(delta.get("is_valid_scene_pbl"))
        cand_valid = bool(d.get("is_valid_scene_pbl"))

        cur_pbl = float(delta.get("total_pbl_loss", 1e18))
        cand_pbl = float(d.get("total_pbl_loss", 1e18))

        cur_oob = float(delta.get("total_oob_loss", 1e18))
        cand_oob = float(d.get("total_oob_loss", 1e18))

        cur_mbl = float(delta.get("total_mbl_loss", 1e18))
        cand_mbl = float(d.get("total_mbl_loss", 1e18))

        improved = False
        if cand_valid and not cur_valid:
            improved = True
        elif cand_valid and cur_valid:
            improved = cand_pbl <= cur_pbl
        elif (not cand_valid) and (not cur_valid):
            if not only_backoff_if_worse:
                improved = cand_pbl <= cur_pbl
            else:
                improved = (cand_pbl <= cur_pbl) and (cand_oob <= cur_oob) and (cand_mbl <= cur_mbl)

        if improved:
            # 关键：即使相对当前更好，也必须满足相对 old 的单调约束（否则不接受）
            if not ok_monotonic(d):
                continue

            scene_cur = cand
            delta = d

            if debug:
                print(
                    "revert", loser,
                    "=> valid", delta.get("is_valid_scene_pbl"),
                    "pbl", delta.get("total_pbl_loss"),
                    "oob", delta.get("total_oob_loss"),
                    "mbl", delta.get("total_mbl_loss"),
                )

            if bool(delta.get("is_valid_scene_pbl")) and (not only_backoff_if_worse or float(delta.get("delta_pbl_loss", 0.0)) <= 0.0):
                return scene_cur

    # final gate: ensure monotonic vs old if requested
    if ok_monotonic(delta):
        return scene_cur
    return scene_old