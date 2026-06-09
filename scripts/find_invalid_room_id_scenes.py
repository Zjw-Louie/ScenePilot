import argparse
import json
from pathlib import Path
from typing import List, Set, Tuple

from tqdm import tqdm


def load_invalid_room_ids(pth: Path) -> Set[str]:
    invalid: Set[str] = set()
    for line in pth.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("//"):
            continue
        invalid.add(s)
    return invalid


def iter_scene_files(root: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.json" if recursive else "*.json"
    return sorted([p for p in root.glob(pattern) if p.is_file()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes_dir", default="/home2/zhangjiawei/respace/scenes")
    ap.add_argument(
        "--invalid_room_ids",
        default="/home2/zhangjiawei/respace/data/metadata/invalid_threed_front_rooms.txt",
    )
    ap.add_argument(
        "--out_report",
        default="/home2/zhangjiawei/respace/data/metadata/scenes_with_invalid_room_id.txt",
    )
    ap.add_argument(
        "--out_missing",
        default="/home2/zhangjiawei/respace/data/metadata/scenes_missing_room_id.txt",
    )
    ap.add_argument("--recursive", action="store_true", default=True)
    ap.add_argument("--no_recursive", dest="recursive", action="store_false")
    args = ap.parse_args()

    scenes_root = Path(args.scenes_dir)
    invalid_path = Path(args.invalid_room_ids)
    out_report = Path(args.out_report)
    out_missing = Path(args.out_missing)

    invalid_ids = load_invalid_room_ids(invalid_path)
    files = iter_scene_files(scenes_root, recursive=args.recursive)

    hits: List[Tuple[str, str]] = []
    missing: List[str] = []
    bad_json: List[Tuple[str, str]] = []

    for p in tqdm(files, desc="Scan scenes", unit="scene"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                scene = json.load(f)
        except Exception as e:
            bad_json.append((str(p), repr(e)))
            continue

        room_id = scene.get("room_id")
        if not isinstance(room_id, str) or not room_id:
            missing.append(str(p))
            continue

        if room_id in invalid_ids:
            hits.append((str(p), room_id))

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_missing.parent.mkdir(parents=True, exist_ok=True)

    out_report.write_text(
        "\n".join([f"{pth}\t{rid}" for pth, rid in hits]) + ("\n" if hits else ""),
        encoding="utf-8",
    )
    out_missing.write_text(
        "\n".join(missing) + ("\n" if missing else ""),
        encoding="utf-8",
    )

    if bad_json:
        bad_path = out_report.with_suffix(".bad_json.txt")
        bad_path.write_text(
            "\n".join([f"{pth}\t{err}" for pth, err in bad_json]) + "\n",
            encoding="utf-8",
        )

    print(
        f"done. total={len(files)} hits={len(hits)} missing_room_id={len(missing)} bad_json={len(bad_json)}\n"
        f"report: {out_report}\n"
        f"missing: {out_missing}"
    )


if __name__ == "__main__":
    main()