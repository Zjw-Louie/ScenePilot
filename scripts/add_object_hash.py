import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm


def _quantize(v: float, step: float) -> int:
    return int(round(v / step))


def _make_match_key_from_pos_rot(pos: List[float], rot: List[float], pos_step: float, rot_step: float) -> str:
    qp = (_quantize(float(pos[0]), pos_step), _quantize(float(pos[1]), pos_step), _quantize(float(pos[2]), pos_step))
    qr = (
        _quantize(float(rot[0]), rot_step),
        _quantize(float(rot[1]), rot_step),
        _quantize(float(rot[2]), rot_step),
        _quantize(float(rot[3]), rot_step),
    )
    raw = f"p:{qp}|r:{qr}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _pos_valid(p: Any) -> bool:
    return isinstance(p, list) and len(p) == 3 and all(isinstance(v, (int, float)) for v in p)


def _rot_valid(r: Any) -> bool:
    return isinstance(r, list) and len(r) == 4 and all(isinstance(v, (int, float)) for v in r)


def attach_match_keys(scene: Dict[str, Any], pos_step: float, rot_step: float, overwrite: bool) -> int:
    objs = scene.get("objects", [])
    if not isinstance(objs, list):
        return 0

    n = 0
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        if (not overwrite) and obj.get("match_key"):
            continue
        pos = obj.get("pos")
        rot = obj.get("rot")
        if _pos_valid(pos) and _rot_valid(rot):
            obj["match_key"] = _make_match_key_from_pos_rot(pos, rot, pos_step=pos_step, rot_step=rot_step)
            n += 1
    return n


def run_dir(in_dir: str, out_dir: str, pos_step: float, rot_step: float, overwrite: bool, recursive: bool) -> None:
    in_root = Path(in_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.json" if recursive else "*.json"
    files = sorted(in_root.glob(pattern))

    for p in tqdm(files, desc="Add match_key", unit="scene"):
        rel = p.relative_to(in_root)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "r", encoding="utf-8") as f:
            scene = json.load(f)

        attach_match_keys(scene, pos_step=pos_step, rot_step=rot_step, overwrite=overwrite)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scene, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--mk_pos_step", type=float, default=1e-3)
    ap.add_argument("--mk_rot_step", type=float, default=1e-4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    run_dir(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        pos_step=args.mk_pos_step,
        rot_step=args.mk_rot_step,
        overwrite=args.overwrite,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()