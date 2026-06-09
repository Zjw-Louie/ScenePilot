import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


# =========================
# Quaternion utils
# =========================

def _quat_mul(q1: List[float], q2: List[float]) -> List[float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def _quat_normalize(q: List[float]) -> List[float]:
    n = math.sqrt(sum(v * v for v in q))
    if n == 0:
        return [0.0, 0.0, 0.0, 1.0]
    return [v / n for v in q]


def _yaw_quat(delta_yaw_rad: float) -> List[float]:
    half = delta_yaw_rad / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]


# =========================
# Bounds utils
# =========================

def _compute_nx_nz_from_bounds(scene: Dict[str, Any]) -> Tuple[float, float]:
    bounds = scene.get("bounds_top") or scene.get("bounds") or scene.get("bounds_bottom")
    if not isinstance(bounds, list) or not bounds:
        raise ValueError("scene 中未找到可用的 bounds_top / bounds / bounds_bottom")

    xs: List[float] = []
    zs: List[float] = []
    for p in bounds:
        if isinstance(p, list) and len(p) >= 3:
            xs.append(float(p[0]))
            zs.append(float(p[2]))

    if not xs or not zs:
        raise ValueError("bounds 点格式不正确，无法提取 x / z")

    nx = (max(xs) - min(xs)) / 2.0
    nz = (max(zs) - min(zs)) / 2.0
    return nx, nz


# =========================
# Scale / size sync utils
# =========================

def _safe_ratio(new_v: float, old_v: float) -> float:
    if abs(old_v) < 1e-8:
        return 1.0
    return new_v / old_v


def _get_current_scale(obj: Dict[str, Any]) -> List[float]:
    scale = obj.get("scale")
    if isinstance(scale, list) and len(scale) == 3:
        return [float(scale[0]), float(scale[1]), float(scale[2])]
    return [1.0, 1.0, 1.0]


def _sync_scale_and_size_only(
    obj: Dict[str, Any],
    old_scale: List[float],
    new_scale: List[float],
    update_size: bool = True,
) -> None:
    obj["scale"] = [float(new_scale[0]), float(new_scale[1]), float(new_scale[2])]

    if update_size:
        size = obj.get("size")
        if isinstance(size, list) and len(size) == 3:
            rx = _safe_ratio(new_scale[0], old_scale[0])
            ry = _safe_ratio(new_scale[1], old_scale[1])
            rz = _safe_ratio(new_scale[2], old_scale[2])
            obj["size"] = [
                float(size[0]) * rx,
                float(size[1]) * ry,
                float(size[2]) * rz,
            ]


# =========================
# Object selection
# =========================

def _is_hanging_object(obj: Dict[str, Any]) -> bool:
    category = str(obj.get("category", "") or "").lower()
    super_category = str(obj.get("super_category", "") or "").lower()
    desc = str(obj.get("desc", "") or "").lower()

    keywords = [
        "pendant lamp",
        "ceiling lamp",
        "chandelier",
        "hanging lamp",
        "ceiling light",
    ]
    text = " ".join([category, super_category, desc])
    return any(k in text for k in keywords)


def _build_candidate_indices(
    objs: List[Dict[str, Any]],
    enable_pos: bool,
    enable_rot: bool,
    enable_scale: bool,
    skip_hanging_for_scale: bool = True,
) -> List[int]:
    indices = []
    for i, obj in enumerate(objs):
        if not isinstance(obj, dict):
            continue

        has_pos = isinstance(obj.get("pos"), list) and len(obj.get("pos")) == 3
        has_rot = isinstance(obj.get("rot"), list) and len(obj.get("rot")) == 4
        has_size = isinstance(obj.get("size"), list) and len(obj.get("size")) == 3

        usable = False
        if enable_pos and has_pos:
            usable = True
        if enable_rot and has_rot:
            usable = True
        if enable_scale and has_size:
            if skip_hanging_for_scale and _is_hanging_object(obj):
                pass
            else:
                usable = True

        if usable:
            indices.append(i)
    return indices


# =========================
# Main perturb logic
# =========================

def perturb_objects(
    scene: Dict[str, Any],
    enable_pos: bool = True,
    enable_rot: bool = True,
    enable_scale: bool = False,
    pos_noise_xyz: Tuple[Optional[float], float, Optional[float]] = (None, 0.0, None),
    yaw_noise_deg: float = 5.0,
    scale_noise_xyz: Tuple[float, float, float] = (0.15, 0.08, 0.15),
    min_scale_xyz: Tuple[float, float, float] = (0.6, 0.8, 0.6),
    max_scale_xyz: Tuple[float, float, float] = (1.6, 1.2, 1.6),
    seed: Optional[int] = None,
    auto_nx_nz: bool = True,
    update_size: bool = True,
    n_perturb_objects: int = 1,
    skip_hanging_for_scale: bool = True,
    write_meta: bool = False,
) -> None:
    rng = random.Random(seed)

    objs = scene.get("objects", [])
    if not isinstance(objs, list) or len(objs) == 0:
        return

    nx, ny, nz = pos_noise_xyz
    if enable_pos and auto_nx_nz and (nx is None or nz is None):
        auto_nx, auto_nz = _compute_nx_nz_from_bounds(scene)
        if nx is None:
            nx = auto_nx
        if nz is None:
            nz = auto_nz

    nx = 0.0 if nx is None else float(nx)
    ny = 0.0 if ny is None else float(ny)
    nz = 0.0 if nz is None else float(nz)

    sx_noise, sy_noise, sz_noise = scale_noise_xyz
    min_sx, min_sy, min_sz = min_scale_xyz
    max_sx, max_sy, max_sz = max_scale_xyz

    candidate_indices = _build_candidate_indices(
        objs=objs,
        enable_pos=enable_pos,
        enable_rot=enable_rot,
        enable_scale=enable_scale,
        skip_hanging_for_scale=skip_hanging_for_scale,
    )

    if not candidate_indices:
        return

    n_pick = max(1, min(int(n_perturb_objects), len(candidate_indices)))
    chosen_indices = rng.sample(candidate_indices, n_pick)

    meta_records = []

    for idx in chosen_indices:
        obj = objs[idx]
        obj_meta: Dict[str, Any] = {
            "object_index": idx,
            "desc": obj.get("desc"),
            "category": obj.get("category"),
            "super_category": obj.get("super_category"),
            "changed": [],
        }

        # ---- pos ----
        if enable_pos:
            pos = obj.get("pos")
            if isinstance(pos, list) and len(pos) == 3:
                old_pos = [float(pos[0]), float(pos[1]), float(pos[2])]
                new_pos = [
                    old_pos[0] + rng.uniform(-nx, nx),
                    old_pos[1] + rng.uniform(-ny, ny),
                    old_pos[2] + rng.uniform(-nz, nz),
                ]
                obj["pos"] = new_pos
                obj_meta["changed"].append("pos")
                obj_meta["old_pos"] = old_pos
                obj_meta["new_pos"] = new_pos

        # ---- rot ----
        if enable_rot:
            rot = obj.get("rot")
            if isinstance(rot, list) and len(rot) == 4:
                old_rot = [float(r) for r in rot]
                delta_yaw_deg = rng.uniform(-yaw_noise_deg, yaw_noise_deg)
                delta_yaw = math.radians(delta_yaw_deg)
                dq = _yaw_quat(delta_yaw)
                new_rot = _quat_normalize(_quat_mul(dq, old_rot))
                obj["rot"] = new_rot
                obj_meta["changed"].append("rot")
                obj_meta["old_rot"] = old_rot
                obj_meta["new_rot"] = new_rot
                obj_meta["delta_yaw_deg"] = delta_yaw_deg

        # ---- scale ----
        if enable_scale:
            size = obj.get("size")
            if isinstance(size, list) and len(size) == 3:
                if not (skip_hanging_for_scale and _is_hanging_object(obj)):
                    old_scale = _get_current_scale(obj)
                    fx = rng.uniform(1.0 - sx_noise, 1.0 + sx_noise)
                    fy = rng.uniform(1.0 - sy_noise, 1.0 + sy_noise)
                    fz = rng.uniform(1.0 - sz_noise, 1.0 + sz_noise)

                    new_scale = [
                        max(min_sx, min(max_sx, old_scale[0] * fx)),
                        max(min_sy, min(max_sy, old_scale[1] * fy)),
                        max(min_sz, min(max_sz, old_scale[2] * fz)),
                    ]

                    _sync_scale_and_size_only(
                        obj=obj,
                        old_scale=old_scale,
                        new_scale=new_scale,
                        update_size=update_size,
                    )

                    obj_meta["changed"].append("scale")
                    obj_meta["old_scale"] = old_scale
                    obj_meta["new_scale"] = new_scale
                    obj_meta["scale_factors"] = [fx, fy, fz]

        meta_records.append(obj_meta)

    if write_meta:
        scene["_perturb_meta"] = {
            "enable_pos": enable_pos,
            "enable_rot": enable_rot,
            "enable_scale": enable_scale,
            "n_perturb_objects": n_pick,
            "records": meta_records,
        }


def process_file(
    in_path: Path,
    out_path: Path,
    enable_pos: bool,
    enable_rot: bool,
    enable_scale: bool,
    pos_noise_xyz: Tuple[Optional[float], float, Optional[float]],
    yaw_noise_deg: float,
    scale_noise_xyz: Tuple[float, float, float],
    min_scale_xyz: Tuple[float, float, float],
    max_scale_xyz: Tuple[float, float, float],
    seed: Optional[int],
    auto_nx_nz: bool,
    update_size: bool,
    n_perturb_objects: int,
    skip_hanging_for_scale: bool,
    write_meta: bool,
) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        scene = json.load(f)

    perturb_objects(
        scene=scene,
        enable_pos=enable_pos,
        enable_rot=enable_rot,
        enable_scale=enable_scale,
        pos_noise_xyz=pos_noise_xyz,
        yaw_noise_deg=yaw_noise_deg,
        scale_noise_xyz=scale_noise_xyz,
        min_scale_xyz=min_scale_xyz,
        max_scale_xyz=max_scale_xyz,
        seed=seed,
        auto_nx_nz=auto_nx_nz,
        update_size=update_size,
        n_perturb_objects=n_perturb_objects,
        skip_hanging_for_scale=skip_hanging_for_scale,
        write_meta=write_meta,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False, indent=2)


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--in_dir", required=True, help="输入目录")
    p.add_argument("--out_dir", required=True, help="输出目录")
    p.add_argument("--recursive", action="store_true")

    # ---- switches ----
    p.add_argument("--enable_pos", dest="enable_pos", action="store_true")
    p.add_argument("--disable_pos", dest="enable_pos", action="store_false")
    p.set_defaults(enable_pos=True)

    p.add_argument("--enable_rot", dest="enable_rot", action="store_true")
    p.add_argument("--disable_rot", dest="enable_rot", action="store_false")
    p.set_defaults(enable_rot=True)

    p.add_argument("--enable_scale", dest="enable_scale", action="store_true")
    p.add_argument("--disable_scale", dest="enable_scale", action="store_false")
    p.set_defaults(enable_scale=False)

    # ---- pos ----
    p.add_argument("--nx", type=float, default=None, help="x 方向最大位移扰动幅度")
    p.add_argument("--ny", type=float, default=0.0, help="y 方向最大位移扰动幅度")
    p.add_argument("--nz", type=float, default=None, help="z 方向最大位移扰动幅度")

    p.add_argument("--auto_nx_nz", action="store_true")
    p.add_argument("--no_auto_nx_nz", dest="auto_nx_nz", action="store_false")
    p.set_defaults(auto_nx_nz=True)

    # ---- rot ----
    p.add_argument("--yaw_deg", type=float, default=5.0, help="yaw 扰动范围 [-yaw_deg, yaw_deg]")

    # ---- scale ----
    p.add_argument("--sx", type=float, default=0.15, help="x 轴相对 scale 扰动幅度")
    p.add_argument("--sy", type=float, default=0.08, help="y 轴相对 scale 扰动幅度")
    p.add_argument("--sz", type=float, default=0.15, help="z 轴相对 scale 扰动幅度")

    p.add_argument("--min_sx", type=float, default=0.6)
    p.add_argument("--min_sy", type=float, default=0.8)
    p.add_argument("--min_sz", type=float, default=0.6)

    p.add_argument("--max_sx", type=float, default=1.6)
    p.add_argument("--max_sy", type=float, default=1.2)
    p.add_argument("--max_sz", type=float, default=1.6)

    p.add_argument("--update_size", action="store_true")
    p.add_argument("--no_update_size", dest="update_size", action="store_false")
    p.set_defaults(update_size=True)

    # ---- object selection ----
    p.add_argument("--n_perturb_objects", type=int, default=1, help="每个场景扰动多少个物体")
    p.add_argument("--skip_hanging_for_scale", action="store_true")
    p.add_argument("--no_skip_hanging_for_scale", dest="skip_hanging_for_scale", action="store_false")
    p.set_defaults(skip_hanging_for_scale=True)

    # ---- misc ----
    p.add_argument("--write_meta", action="store_true", help="在 scene 中写入 _perturb_meta")
    p.add_argument("--seed", type=int, default=None)

    args = p.parse_args()

    in_root = Path(args.in_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.json" if args.recursive else "*.json"
    files = sorted(in_root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"在 {in_root} 下未找到任何 json 文件")

    pos_noise_xyz = (args.nx, args.ny, args.nz)
    scale_noise_xyz = (args.sx, args.sy, args.sz)
    min_scale_xyz = (args.min_sx, args.min_sy, args.min_sz)
    max_scale_xyz = (args.max_sx, args.max_sy, args.max_sz)

    for i, in_path in enumerate(tqdm(files, desc="Perturb scenes", unit="scene")):
        rel = in_path.relative_to(in_root)
        out_path = out_root / rel
        file_seed = None if args.seed is None else (args.seed + i)

        process_file(
            in_path=in_path,
            out_path=out_path,
            enable_pos=args.enable_pos,
            enable_rot=args.enable_rot,
            enable_scale=args.enable_scale,
            pos_noise_xyz=pos_noise_xyz,
            yaw_noise_deg=args.yaw_deg,
            scale_noise_xyz=scale_noise_xyz,
            min_scale_xyz=min_scale_xyz,
            max_scale_xyz=max_scale_xyz,
            seed=file_seed,
            auto_nx_nz=args.auto_nx_nz,
            update_size=args.update_size,
            n_perturb_objects=args.n_perturb_objects,
            skip_hanging_for_scale=args.skip_hanging_for_scale,
            write_meta=args.write_meta,
        )


if __name__ == "__main__":
    main()