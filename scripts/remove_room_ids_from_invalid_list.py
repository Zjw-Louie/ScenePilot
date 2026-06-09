import argparse
from pathlib import Path
from typing import Set


def load_room_ids_from_report(p: Path) -> Set[str]:
    ids: Set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        # 兼容两种格式：
        # 1) "<scene_path>\t<room_id>"
        # 2) "<room_id>"
        if "\t" in s:
            parts = [x for x in s.split("\t") if x]
            if parts:
                ids.add(parts[-1].strip())
        else:
            ids.add(s)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--report",
        default="/home2/zhangjiawei/respace/data/metadata/scenes_with_invalid_room_id.txt",
    )
    ap.add_argument(
        "--invalid_list",
        default="/home2/zhangjiawei/respace/data/metadata/invalid_threed_front_rooms.txt",
    )
    args = ap.parse_args()

    report_path = Path(args.report)
    invalid_path = Path(args.invalid_list)

    to_remove = load_room_ids_from_report(report_path)
    lines = invalid_path.read_text(encoding="utf-8").splitlines()

    kept = []
    removed = 0
    for line in lines:
        s = line.strip()
        if s in to_remove:
            removed += 1
            continue
        kept.append(line)

    invalid_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"done. removed={removed} from {invalid_path}")


if __name__ == "__main__":
    main()