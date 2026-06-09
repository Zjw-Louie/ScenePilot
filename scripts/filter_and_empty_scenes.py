import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm

SRC_DIR = Path("/home2/zhangjiawei/respace/scenes_SSR_v2")
DST_DIR = Path("/home2/zhangjiawei/respace/scenes_filter")

MIN_LEN = 4.0
MIN_WID = 4.0
MIN_OBJECTS = 5  # 场景中物体数量 <= 6 的先过滤掉

OBJECT_KEYS = [
    "objects",
    "Objects",
    "object_list",
    "instance_list",
    "instances",
]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        tqdm.write(f"[WARN] 读取失败: {path} ({e})")
        return None


def _get_bounds_points(scene: Dict[str, Any]) -> Optional[List[List[float]]]:
    """
    返回 bounds_top/bounds_bottom 的所有点（支持多边形，>=3点）。
    """
    for k in ("bounds_top", "bounds_bottom"):
        pts = scene.get(k)
        if isinstance(pts, list) and len(pts) >= 3:
            out: List[List[float]] = []
            ok = True
            for p in pts:
                if not (isinstance(p, list) and len(p) >= 3):
                    ok = False
                    break
                out.append([float(p[0]), float(p[1]), float(p[2])])
            if ok:
                return out
    return None


def _dims_from_bounds(pts: List[List[float]]) -> Tuple[float, float]:
    """
    用全部点的 (x, z) 轴对齐包围盒估算长宽。
    """
    xs = [p[0] for p in pts]
    zs = [p[2] for p in pts]
    len_x = max(xs) - min(xs)
    len_z = max(zs) - min(zs)
    return float(len_x), float(len_z)


def _count_objects(scene: Dict[str, Any]) -> int:
    """
    统计场景中的物体数量。
    返回 OBJECT_KEYS 中第一个存在且为 list 的字段长度。
    如果都没有，则返回 0。
    """
    for k in OBJECT_KEYS:
        objs = scene.get(k)
        if isinstance(objs, list):
            return len(objs)
    return 0


def _clear_objects(scene: Dict[str, Any]) -> None:
    cleared = False
    for k in OBJECT_KEYS:
        if k in scene:
            scene[k] = []
            cleared = True
    if not cleared:
        scene["objects"] = []


def _sanitize_room_type(room_id: Any) -> str:
    """
    使用 room_id / room_type 风格字符串进行清洗：
      "bedroom-12532" -> "bedroom"
      "livingroom_88" -> "livingroom"
      "Living Room-123" -> "living_room"
      None/"" -> "unknown"
    """
    if room_id is None:
        return "unknown"

    text = str(room_id).strip().lower()
    if not text:
        return "unknown"

    # 先把空白变成下划线，保证 "Living Room-123" -> "living_room-123"
    text = re.sub(r"\s+", "_", text)

    # 如果形如 "<prefix>-<digits>" 或 "<prefix>_<digits>"，提取 prefix
    m = re.match(r"^([a-z0-9_]+)[\-_]\d+.*$", text)
    if m:
        text = m.group(1)

    # 最终再做一次路径安全清洗
    text = re.sub(r"[^a-z0-9_\-]", "", text)

    return text if text else "unknown"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Filter scenes by object count and room size; optionally empty objects.")
    ap.add_argument(
        "--empty",
        dest="do_empty",
        action="store_true",
        default=True,
        help="Clear objects in output scenes (default: enabled).",
    )
    ap.add_argument(
        "--no-empty",
        dest="do_empty",
        action="store_false",
        help="Only filter scenes; keep objects unchanged.",
    )
    args = ap.parse_args()

    if not SRC_DIR.exists():
        raise SystemExit(f"源目录不存在: {SRC_DIR}")

    DST_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(SRC_DIR.rglob("*.json"))

    total = len(paths)
    passed_object_filter = 0
    passed_size_filter = 0
    written = 0

    room_type_stats: Dict[str, int] = {}

    for path in tqdm(paths, desc="过滤并分类输出", unit="scene", dynamic_ncols=True):
        scene = _load_json(path)
        if scene is None or not isinstance(scene, dict):
            continue

        # 1) 先过滤物体数量
        num_objects = _count_objects(scene)
        if num_objects <= MIN_OBJECTS:
            continue
        passed_object_filter += 1

        # 2) 再过滤场景尺寸
        pts = _get_bounds_points(scene)
        if pts is None:
            continue

        len_x, len_z = _dims_from_bounds(pts)
        long_side = max(len_x, len_z)
        short_side = min(len_x, len_z)

        if long_side < MIN_LEN or short_side < MIN_WID:
            continue
        passed_size_filter += 1

        # 3) 可选：清空 objects
        if args.do_empty:
            _clear_objects(scene)

        # 4) 按 room_type 分类输出
        room_type = _sanitize_room_type(scene.get("room_id", "unknown"))

        # 保留原相对路径，避免同名文件覆盖
        rel = path.relative_to(SRC_DIR)
        out_path = DST_DIR / room_type / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            out_path.write_text(
                json.dumps(scene, ensure_ascii=False, indent=4) + "\n",
                encoding="utf-8",
            )
            written += 1
            room_type_stats[room_type] = room_type_stats.get(room_type, 0) + 1
        except Exception as e:
            tqdm.write(f"[WARN] 写入失败: {out_path} ({e})")

    print(f"总计 scene: {total}")
    print(f"物体数量 > {MIN_OBJECTS} 的 scene: {passed_object_filter}")
    print(f"同时满足尺寸 >= {MIN_LEN} x {MIN_WID} 的 scene: {passed_size_filter}")
    if args.do_empty:
        print(f"已写入空 objects 的 scene: {written}")
    else:
        print(f"已写入（不清空 objects）的 scene: {written}")
    print(f"输出根目录: {DST_DIR}")

    if room_type_stats:
        print("\n各 room_type 写入数量：")
        for rt, cnt in sorted(room_type_stats.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {rt}: {cnt}")


if __name__ == "__main__":
    main()