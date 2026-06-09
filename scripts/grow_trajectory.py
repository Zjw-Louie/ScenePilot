import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm


def _clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _quat_normalize(q: List[float]) -> List[float]:
    n = math.sqrt(sum(v * v for v in q))
    if n == 0:
        return [0.0, 0.0, 0.0, 1.0]
    return [v / n for v in q]


def _quat_to_yaw(q: List[float]) -> float:
    x, y, z, w = q
    t0 = 2.0 * (w * y + x * z)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t0, t1)


def _yaw_to_quat(yaw: float) -> List[float]:
    half = yaw / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def _pos_valid(p: Any) -> bool:
    return isinstance(p, list) and len(p) == 3 and all(isinstance(v, (int, float)) for v in p)


def _rot_valid(r: Any) -> bool:
    return isinstance(r, list) and len(r) == 4 and all(isinstance(v, (int, float)) for v in r)


def _l2(v: List[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _make_monotone_progress(K: int, rng: random.Random) -> List[float]:
    if K < 2:
        raise ValueError("n_steps 必须 >= 2")
    weights = [rng.random() ** rng.uniform(0.3, 2.2) + 1e-6 for _ in range(K)]
    s = sum(weights)
    steps = [w / s for w in weights]
    p = []
    acc = 0.0
    for i in range(K):
        acc += steps[i]
        p.append(acc)
    p[-1] = 1.0
    eps = 1e-6
    for i in range(1, K):
        if p[i] <= p[i - 1]:
            p[i] = min(1.0, p[i - 1] + eps)
    p[-1] = 1.0
    return p


def _score_0_10_from_done_ratio(done_ratio: float) -> float:
    return round(_clamp(done_ratio, 0.0, 1.0) * 10.0, 3)


def build_growth_steps_for_scene(
    gt_scene: Dict[str, Any],
    deg_scene: Dict[str, Any],
    n_steps: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    base_progress = _make_monotone_progress(n_steps, rng)

    gt_objs = [o for o in gt_scene.get("objects", []) if isinstance(o, dict)]
    deg_objs = [o for o in deg_scene.get("objects", []) if isinstance(o, dict)]

    gt_by_key = {o.get("match_key"): o for o in gt_objs if o.get("match_key")}
    deg_by_key = {o.get("match_key"): o for o in deg_objs if o.get("match_key")}

    common_keys = [k for k in deg_by_key.keys() if k in gt_by_key]

    obj_targets: Dict[str, Dict[str, Any]] = {}
    obj_alphas: Dict[str, Tuple[float, float]] = {}  # (alpha_pos, alpha_rot)

    for key in common_keys:
        og = gt_by_key[key]
        od = deg_by_key[key]
        if not (
            _pos_valid(og.get("pos"))
            and _pos_valid(od.get("pos"))
            and _rot_valid(og.get("rot"))
            and _rot_valid(od.get("rot"))
        ):
            continue

        gt_pos = [float(x) for x in og["pos"]]
        dg_pos = [float(x) for x in od["pos"]]
        dpos = [gt_pos[i] - dg_pos[i] for i in range(3)]

        gt_yaw = _quat_to_yaw([float(x) for x in og["rot"]])
        dg_yaw = _quat_to_yaw([float(x) for x in od["rot"]])
        dyaw = _wrap_pi(gt_yaw - dg_yaw)

        obj_targets[key] = {
            "deg_pos": dg_pos,
            "gt_pos": gt_pos,
            "dpos": dpos,
            "deg_yaw": dg_yaw,
            "gt_yaw": gt_yaw,
            "dyaw": dyaw,
            "pos_total_norm": _l2(dpos),
            "rot_total_abs": abs(dyaw),
        }

        # ✅ 每个物体给一个不同的进度形状（pos/rot 分开），保证多样性
        # alpha < 1: 前期更快；alpha > 1: 前期更慢
        alpha_pos = rng.uniform(0.55, 1.85)
        alpha_rot = rng.uniform(0.55, 1.85)
        # 再加一点“解耦”，避免 pos 与 rot 太容易一样
        if abs(alpha_pos - alpha_rot) < 0.15:
            alpha_rot = _clamp(alpha_rot + rng.choice([-0.25, 0.25]), 0.55, 1.85)
        obj_alphas[key] = (alpha_pos, alpha_rot)

    scenes_steps: List[Dict[str, Any]] = []
    step_metrics: List[Dict[str, Any]] = []

    # step00: 初始未优化状态 = degraded 场景
    step0_scene = json.loads(json.dumps(deg_scene))
    scenes_steps.append(step0_scene)

    # step00 的 metrics：progress=0，scores=0，remaining 为到 gt 的完整距离
    step0_per_obj: List[Dict[str, Any]] = []
    deg_by_key_for_jid = {o.get("match_key"): o for o in deg_scene.get("objects", []) if isinstance(o, dict) and o.get("match_key")}
    for key in common_keys:
        if key not in obj_targets:
            continue
        t = obj_targets[key]
        alpha_pos, alpha_rot = obj_alphas[key]
        obj = deg_by_key_for_jid.get(key, {})
        step0_per_obj.append(
            {
                "match_key": key,
                "jid": obj.get("jid"),
                "alpha_pos": round(alpha_pos, 4),
                "alpha_rot": round(alpha_rot, 4),
                "p_pos": 0.0,
                "p_rot": 0.0,
                "delta_pos_step": [0.0, 0.0, 0.0],
                "delta_pos_cum": [0.0, 0.0, 0.0],
                "delta_rot_step_rad": 0.0,
                "delta_rot_cum_rad": 0.0,
                "pos_remaining": [round(v, 6) for v in t["dpos"]],
                "pos_remaining_norm": round(t["pos_total_norm"], 6),
                "rot_remaining_rad": round(t["dyaw"], 6),
                "rot_remaining_abs": round(t["rot_total_abs"], 6),
                "pos_score_0_10": 0.0,
                "rot_score_0_10": 0.0,
                "total_score_0_10": 0.0,
            }
        )
    step_metrics.append(
        {
            "step": 0,
            "progress_p": 0.0,
            "progress_step": 0.0,
            "global_pos_score_0_10": 0.0,
            "global_rot_score_0_10": 0.0,
            "global_total_score_0_10": 0.0,
            "per_object": step0_per_obj,
        }
    )

    prev_base_p = 0.0

    for step_idx, base_p in enumerate(base_progress, start=1):
        step_scene = json.loads(json.dumps(deg_scene))
        per_obj = []

        total_pos_remaining = 0.0
        total_pos_need = 0.0
        total_rot_remaining = 0.0
        total_rot_need = 0.0

        for obj in step_scene.get("objects", []):
            if not isinstance(obj, dict):
                continue
            key = obj.get("match_key")
            if not key or key not in obj_targets:
                continue

            t = obj_targets[key]
            dpos = t["dpos"]
            dyaw = t["dyaw"]

            alpha_pos, alpha_rot = obj_alphas[key]

            # ✅ object-specific progress（保证最后一步仍为 1.0）
            p_pos = 1.0 if base_p >= 1.0 else base_p ** alpha_pos
            p_rot = 1.0 if base_p >= 1.0 else base_p ** alpha_rot
            prev_p_pos = 0.0 if prev_base_p <= 0.0 else (1.0 if prev_base_p >= 1.0 else prev_base_p ** alpha_pos)
            prev_p_rot = 0.0 if prev_base_p <= 0.0 else (1.0 if prev_base_p >= 1.0 else prev_base_p ** alpha_rot)

            # 累计与本步增量（相对 degraded）
            pos_cum = [p_pos * dpos[i] for i in range(3)]
            pos_step = [(p_pos - prev_p_pos) * dpos[i] for i in range(3)]
            rot_cum = p_rot * dyaw
            rot_step = (p_rot - prev_p_rot) * dyaw

            # 更新 scene
            new_pos = [t["deg_pos"][i] + pos_cum[i] for i in range(3)]
            new_yaw = _wrap_pi(t["deg_yaw"] + rot_cum)
            new_rot = _quat_normalize(_yaw_to_quat(new_yaw))

            obj["pos"] = [round(x, 6) for x in new_pos]
            obj["rot"] = [round(x, 6) for x in new_rot]

            # remaining（距 GT 还差多少）
            pos_remaining = [(1.0 - p_pos) * dpos[i] for i in range(3)]
            rot_remaining = (1.0 - p_rot) * dyaw

            pos_need = t["pos_total_norm"]
            rot_need = t["rot_total_abs"]

            pos_remaining_norm = _l2(pos_remaining)
            rot_remaining_abs = abs(rot_remaining)

            pos_done_ratio = 1.0 - (pos_remaining_norm / pos_need) if pos_need > 1e-9 else 1.0
            rot_done_ratio = 1.0 - (rot_remaining_abs / rot_need) if rot_need > 1e-9 else 1.0

            pos_score = _score_0_10_from_done_ratio(pos_done_ratio)
            rot_score = _score_0_10_from_done_ratio(rot_done_ratio)
            total_score = round((pos_score + rot_score) / 2.0, 3)

            per_obj.append(
                {
                    "match_key": key,
                    "jid": obj.get("jid"),
                    "alpha_pos": round(alpha_pos, 4),
                    "alpha_rot": round(alpha_rot, 4),
                    "p_pos": round(p_pos, 6),
                    "p_rot": round(p_rot, 6),
                    "delta_pos_step": [round(v, 6) for v in pos_step],
                    "delta_pos_cum": [round(v, 6) for v in pos_cum],
                    "delta_rot_step_rad": round(rot_step, 6),
                    "delta_rot_cum_rad": round(rot_cum, 6),
                    "pos_remaining": [round(v, 6) for v in pos_remaining],
                    "pos_remaining_norm": round(pos_remaining_norm, 6),
                    "rot_remaining_rad": round(rot_remaining, 6),
                    "rot_remaining_abs": round(rot_remaining_abs, 6),
                    "pos_score_0_10": pos_score,
                    "rot_score_0_10": rot_score,
                    "total_score_0_10": total_score,
                }
            )

            total_pos_remaining += pos_remaining_norm
            total_pos_need += pos_need
            total_rot_remaining += rot_remaining_abs
            total_rot_need += rot_need

        if not per_obj:
            global_pos_score = 0.0
            global_rot_score = 0.0
            global_total_score = 0.0
        else:
            global_pos_done = 1.0 - (total_pos_remaining / total_pos_need) if total_pos_need > 1e-9 else 1.0
            global_rot_done = 1.0 - (total_rot_remaining / total_rot_need) if total_rot_need > 1e-9 else 1.0
            global_pos_score = _score_0_10_from_done_ratio(global_pos_done)
            global_rot_score = _score_0_10_from_done_ratio(global_rot_done)
            global_total_score = round((global_pos_score + global_rot_score) / 2.0, 3)

        step_metrics.append(
            {
                "step": step_idx,
                "progress_p": round(base_p, 6),
                "progress_step": round(base_p - prev_base_p, 6),
                "global_pos_score_0_10": global_pos_score,
                "global_rot_score_0_10": global_rot_score,
                "global_total_score_0_10": global_total_score,
                "per_object": per_obj,
            }
        )

        scenes_steps.append(step_scene)
        prev_base_p = base_p

    return scenes_steps, step_metrics


def run_dir(
    gt_dir: str,
    degraded_dir: str,
    out_dir: str,
    n_steps: int = 5,
    seed: int = 42,
    recursive: bool = False,
) -> None:
    gt_root = Path(gt_dir)
    deg_root = Path(degraded_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.json" if recursive else "*.json"
    files = sorted(deg_root.glob(pattern))

    print(f"Processing {len(files)} degraded scenes from {deg_root}")

    for i, deg_path in enumerate(tqdm(files, desc="Building growth trajectories", unit="scene")):
        rel = deg_path.relative_to(deg_root)
        gt_path = gt_root / rel
        if not gt_path.exists():
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_scene = json.load(f)
        with open(deg_path, "r", encoding="utf-8") as f:
            deg_scene = json.load(f)

        scene_seed = seed + i * 10007

        scenes_steps, step_metrics = build_growth_steps_for_scene(
            gt_scene=gt_scene,
            deg_scene=deg_scene,
            n_steps=n_steps,
            seed=scene_seed,
        )

        scene_out_dir = out_root / rel.with_suffix("")
        scene_out_dir.mkdir(parents=True, exist_ok=True)

        # step_00 = degraded 初始状态，step_01..step_N 为优化轨迹
        for s_idx, sc in enumerate(scenes_steps):
            out_scene_path = scene_out_dir / f"step_{s_idx:02d}.json"
            with open(out_scene_path, "w", encoding="utf-8") as f:
                json.dump(sc, f, ensure_ascii=False, indent=2)

            out_metrics_path = scene_out_dir / f"delta_metrics_step_{s_idx:02d}.json"
            with open(out_metrics_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "gt_path": str(gt_path),
                        "degraded_path": str(deg_path),
                        "step": s_idx,
                        "n_steps": n_steps,
                        "seed": scene_seed,
                        "metrics": step_metrics[s_idx],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--degraded_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--steps", type=int, default=5, help="4~5 比较合适")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recursive", action="store_true")
    args = p.parse_args()

    run_dir(
        gt_dir=args.gt_dir,
        degraded_dir=args.degraded_dir,
        out_dir=args.out_dir,
        n_steps=args.steps,
        seed=args.seed,
        recursive=args.recursive,
    )