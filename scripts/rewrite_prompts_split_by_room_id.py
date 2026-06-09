import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple


def normalize_room_id(room_id: str) -> str:
    """
    e.g. "DiningRoom-11628" -> "DiningRoom"
    Only strips a trailing "-<digits>" suffix.
    """
    room_id = room_id.strip()
    return re.sub(r"-\d+$", "", room_id)


def build_scene_to_room(scenes_root: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for p in scenes_root.rglob("*.json"):
        scene_id = p.stem
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue
        rid = obj.get("room_id")
        if isinstance(rid, str) and rid.strip():
            mapping.setdefault(scene_id, normalize_room_id(rid))
    return mapping


def extract_scene_id(rec: dict) -> Optional[str]:
    # common keys: scene_id / sceneId / id
    for k in ("scene_id", "sceneId", "id"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Rewrite prompts jsonl split by scenes room_id.")
    ap.add_argument(
        "--scenes-root",
        type=Path,
        default="/home2/zhangjiawei/respace/scenes",
        help="Directory containing scene jsons with room_id.",
    )
    ap.add_argument(
        "--in-jsonl",
        type=Path,
        default="/home2/zhangjiawei/respace/benchmark/room_prompts_qwen3_vl/all_prompts.jsonl",
        help="Input prompts jsonl.",
    )
    ap.add_argument(
        "--out-jsonl",
        type=Path,
        default="/home2/zhangjiawei/respace/benchmark/room_prompts_qwen3_vl/all_prompt_room_id.jsonl",
        help="Output jsonl with updated split.",
    )
    args = ap.parse_args()

    if not args.scenes_root.exists():
        raise SystemExit(f"[ERR] scenes-root not found: {args.scenes_root}")
    if not args.in_jsonl.exists():
        raise SystemExit(f"[ERR] in-jsonl not found: {args.in_jsonl}")

    scene2room = build_scene_to_room(args.scenes_root)
    if not scene2room:
        raise SystemExit("[ERR] scene->room mapping is empty (no readable scenes with room_id?)")

    total = 0
    updated = 0
    no_scene_id = 0
    no_mapping = 0

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.in_jsonl.open("r", encoding="utf-8") as fin, args.out_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            s = line.strip()
            if not s:
                continue

            try:
                rec = json.loads(s)
            except Exception:
                # keep original line if it's not valid json, but count it in total
                fout.write(line)
                continue

            scene_id = extract_scene_id(rec)
            if not scene_id:
                no_scene_id += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            room = scene2room.get(scene_id)
            if not room:
                no_mapping += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            rec["split"] = room
            updated += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[DONE]")
    print(f"  scenes_root : {args.scenes_root}")
    print(f"  in_jsonl    : {args.in_jsonl}")
    print(f"  out_jsonl   : {args.out_jsonl}")
    print(f"  total_lines : {total}")
    print(f"  updated     : {updated}")
    print(f"  no_scene_id : {no_scene_id}")
    print(f"  no_mapping  : {no_mapping}")


if __name__ == "__main__":
    main()