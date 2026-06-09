import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_dirname(name: str) -> str:
    # make safe as folder name
    name = name.strip()
    name = re.sub(r"[\/\0]", "_", name)
    return name or "UNKNOWN_ROOM_ID"


def _normalize_room_id(room_id: str) -> str:
    """
    e.g. "DiningRoom-11628" -> "DiningRoom"
    Only strips a trailing "-<digits>" suffix.
    """
    room_id = room_id.strip()
    return re.sub(r"-\d+$", "", room_id)


def _extract_room_id(scene: Dict[str, Any]) -> Optional[str]:
    rid = scene.get("room_id")
    if isinstance(rid, str) and rid.strip():
        return rid.strip()
    return None


def _build_scenes_index(scenes_root: Path) -> Dict[str, Path]:
    """
    Build index by filename -> first path found.
    If you have duplicate filenames under scenes_root, the first one wins.
    """
    idx: Dict[str, Path] = {}
    for p in scenes_root.rglob("*.json"):
        idx.setdefault(p.name, p)
    return idx


def _find_matching_scene_json(
    empty_path: Path, scenes_root: Path, scenes_index: Optional[Dict[str, Path]]
) -> Optional[Path]:
    """
    Match priority:
    1) same filename directly under scenes_root
    2) index lookup by filename (recursive)
    3) index lookup by same stem
    """
    filename = empty_path.name

    direct = scenes_root / filename
    if direct.exists():
        return direct

    if scenes_index is None:
        return None

    if filename in scenes_index:
        return scenes_index[filename]

    stem = empty_path.stem
    for k, v in scenes_index.items():
        if Path(k).stem == stem:
            return v

    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-bucket empty_scenes by room_id found in scenes."
    )
    ap.add_argument(
        "--empty-root",
        default="/home2/zhangjiawei/respace/empty_scenes",
        help="Path to empty_scenes root (recursive scan).",
    )
    ap.add_argument(
        "--scenes-root",
        default="/home2/zhangjiawei/respace/scenes",
        help="Path to scenes root (recursive scan if --index-scenes).",
    )
    ap.add_argument(
        "--out-root",
        default="/home2/zhangjiawei/respace/empty_scenes_room_id",
        help="Output root; will create <out_root>/<room_id>/<filename>.json",
    )
    ap.add_argument(
        "--mode",
        choices=["copy", "write"],
        default="copy",
        help="copy: copy file bytes; write: load+dump normalized json.",
    )
    ap.add_argument(
        "--index-scenes",
        action="store_true",
        help="Build a filename index for scenes_root (recommended if scenes_root isn't flat).",
    )
    args = ap.parse_args()

    empty_root = Path(args.empty_root)
    scenes_root = Path(args.scenes_root)
    out_root = Path(args.out_root)

    if not empty_root.exists():
        raise SystemExit(f"[ERR] empty_root not found: {empty_root}")
    if not scenes_root.exists():
        raise SystemExit(f"[ERR] scenes_root not found: {scenes_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    scenes_index = _build_scenes_index(scenes_root) if args.index_scenes else None

    empty_files = sorted(empty_root.rglob("*.json"))
    if not empty_files:
        print(f"[WARN] no json found under: {empty_root}")
        return

    total = 0
    ok = 0
    no_match = 0
    no_room_id = 0

    missing: list[str] = []
    missing_room: list[str] = []

    for empty_path in empty_files:
        total += 1

        match_path = _find_matching_scene_json(empty_path, scenes_root, scenes_index)
        if match_path is None or not match_path.exists():
            no_match += 1
            missing.append(str(empty_path))
            continue

        try:
            scene = _load_json(match_path)
        except Exception as e:
            no_match += 1
            missing.append(f"{empty_path}  (matched={match_path}, json_error={e})")
            continue

        room_id = _extract_room_id(scene)
        if not room_id:
            no_room_id += 1
            missing_room.append(f"{empty_path}  (matched={match_path})")
            continue

        room_id_norm = _normalize_room_id(room_id)
        room_dir = out_root / _safe_dirname(room_id_norm)
        room_dir.mkdir(parents=True, exist_ok=True)
        out_path = room_dir / empty_path.name

        if args.mode == "copy":
            shutil.copy2(empty_path, out_path)
        else:
            try:
                empty_scene = _load_json(empty_path)
            except Exception as e:
                no_match += 1
                missing.append(f"{empty_path}  (empty_json_error={e})")
                continue

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(empty_scene, f, ensure_ascii=False, indent=2)

        ok += 1

    print("[DONE]")
    print(f"  empty_root : {empty_root}")
    print(f"  scenes_root: {scenes_root}")
    print(f"  out_root   : {out_root}")
    print(f"  total      : {total}")
    print(f"  ok         : {ok}")
    print(f"  no_match   : {no_match}")
    print(f"  no_room_id : {no_room_id}")

    if missing:
        miss_path = out_root / "_missing_scene_match.txt"
        miss_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"  wrote      : {miss_path}")

    if missing_room:
        miss_room_path = out_root / "_missing_room_id.txt"
        miss_room_path.write_text("\n".join(missing_room) + "\n", encoding="utf-8")
        print(f"  wrote      : {miss_room_path}")


if __name__ == "__main__":
    main()