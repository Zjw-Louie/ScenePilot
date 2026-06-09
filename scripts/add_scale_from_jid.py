import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OBJECT_KEYS = ["objects", "Objects", "object_list", "instance_list", "instances"]

# Matches: <uuid>-(0.99)-(1.0)-(1.0)  (allow spaces, +/- and scientific notation)
SCALE_RE = re.compile(
    r"""
    -\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\)
    -\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\)
    -\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\)\s*$
    """.strip(),
    re.VERBOSE,
)

DEFAULT_SCALE: Tuple[float, float, float] = (1.0, 1.0, 1.0)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _get_objects_list(scene: Dict[str, Any]) -> Tuple[Optional[str], Optional[List[Any]]]:
    for k in OBJECT_KEYS:
        v = scene.get(k)
        if isinstance(v, list):
            return k, v
    return None, None


def _scale_from_jid(jid: Any) -> Tuple[float, float, float]:
    if not isinstance(jid, str) or not jid.strip():
        return DEFAULT_SCALE
    m = SCALE_RE.search(jid.strip())
    if not m:
        return DEFAULT_SCALE
    try:
        return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    except Exception:
        return DEFAULT_SCALE


def main() -> None:
    ap = argparse.ArgumentParser(description="Add per-object scale field parsed from jid.")
    ap.add_argument(
        "src_root",
        type=Path,
        help="Source root dir, e.g. /home2/zhangjiawei/respace/scenes_SSR_v2",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="If set, write modified jsons under this root (keep relative paths). If not set, use --inplace.",
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Modify files in place. Required if --out-root is not provided.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Do not write files; only print stats.")
    ap.add_argument("--ext", type=str, default=".json")
    args = ap.parse_args()

    src_root: Path = args.src_root
    out_root: Optional[Path] = args.out_root

    if not src_root.exists():
        raise SystemExit(f"[ERR] src_root not found: {src_root}")

    if out_root is None and not args.inplace:
        raise SystemExit("[ERR] Provide --out-root or use --inplace")

    paths = sorted(src_root.rglob(f"*{args.ext}"))

    n_files = 0
    n_written = 0
    n_objs = 0
    n_set = 0
    n_default = 0
    n_skipped = 0

    for p in paths:
        n_files += 1
        scene = _load_json(p)
        if scene is None:
            n_skipped += 1
            continue

        k, objs = _get_objects_list(scene)
        if objs is None:
            n_skipped += 1
            continue

        changed = False
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            n_objs += 1
            scale = _scale_from_jid(obj.get("jid"))
            if scale == DEFAULT_SCALE:
                n_default += 1
            obj["scale"] = [scale[0], scale[1], scale[2]]
            n_set += 1
            changed = True

        if not changed:
            continue

        if args.dry_run:
            continue

        if out_root is not None:
            rel = p.relative_to(src_root)
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _dump_json(out_path, scene)
        else:
            _dump_json(p, scene)

        n_written += 1

    print("[DONE]")
    print(f"  files_scanned : {n_files}")
    print(f"  files_written : {n_written} {'(dry-run)' if args.dry_run else ''}")
    print(f"  scenes_skipped: {n_skipped}")
    print(f"  objects_seen  : {n_objs}")
    print(f"  scale_set     : {n_set}")
    print(f"  default_scale : {n_default}")
    if out_root is not None:
        print(f"  out_root      : {out_root}")
    else:
        print("  mode          : inplace")


if __name__ == "__main__":
    main()