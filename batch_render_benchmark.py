from src.respace import ReSpace
from src.viz import render_annotated_top_view

from pathlib import Path
import json
import traceback


# =========================
# 1. 路径配置
# =========================
jsonl_path = Path("/home2/zhangjiawei/respace/benchmark/sample_benchmark_by_ratio_vlm.jsonl")
scenes_root = Path("/home2/zhangjiawei/respace/benchmark/scenes_filter")
output_root = Path("/home2/zhangjiawei/respace/benchmark/rendered_benchmark")

# 是否跳过已经渲染过的场景
SKIP_IF_EXISTS = True


# =========================
# 2. 初始化 ReSpace
# =========================
respace = ReSpace()


# =========================
# 3. 工具函数
# =========================
def load_jsonl(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] jsonl 第 {line_idx} 行解析失败: {e}")
    return data


def get_room_folder(item: dict) -> str:
    """
    优先使用 empty_scene_room_id_folder。
    兼容以下几种情况：
    1) "livingroom"
    2) "/home2/.../empty_scenes/livingroom"
    3) "LivingRoom"
    """
    val = item.get("empty_scene_room_id_folder", "")
    if val in [None, ""]:
        return ""

    val = str(val).strip()

    # 如果是完整路径，取最后一级目录名
    folder_name = Path(val).name if ("/" in val or "\\" in val) else val

    # 统一成 scenes_filter 下的目录风格
    return folder_name.strip().lower()


def find_scene_json(scenes_root: Path, room_folder: str, scene_id: str) -> Path | None:
    """
    在 scenes_filter/<room_folder>/ 下查找 scene_id.json
    """
    room_dir = scenes_root / room_folder
    if not room_dir.exists():
        return None

    # 最常见情况：直接是 <scene_id>.json
    direct_path = room_dir / f"{scene_id}.json"
    if direct_path.exists():
        return direct_path

    # 递归搜一下
    matches = list(room_dir.rglob(f"{scene_id}.json"))
    if matches:
        return matches[0]

    return None


def already_rendered(out_dir: Path) -> bool:
    """
    判断是否已经渲染过。
    """
    annotated_jpg = out_dir / "top-annotated" / "frame_annotated_top.jpg"
    annotated_png = out_dir / "top-annotated" / "frame_annotated_top.png"

    if annotated_jpg.exists() or annotated_png.exists():
        return True

    img_files = (
        list(out_dir.rglob("*.jpg")) +
        list(out_dir.rglob("*.jpeg")) +
        list(out_dir.rglob("*.png"))
    )
    return len(img_files) > 0


def render_one_scene(scene_json_path: Path, out_dir: Path):
    """
    渲染单个场景：
    1) render_scene_frame
    2) render_annotated_top_view
    """
    scene = json.loads(scene_json_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 常规渲染
    respace.render_scene_frame(
        scene,
        filename="frame",
        pth_viz_output=out_dir,
    )

    # 标注顶视图
    render_annotated_top_view(
        scene,
        filename="frame_annotated_top",
        pth_viz_output=out_dir,
        resolution=(1024, 1024),
        use_dynamic_zoom=True,
        camera_height=None,
        show_assets=True,
        font_size=14,
        bg_color=None,
        show_bboxes=False,
    )


# =========================
# 4. 主流程
# =========================
def main():
    output_root.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(jsonl_path)
    print(f"[INFO] 共读取 {len(items)} 条样本")

    success_count = 0
    skip_count = 0
    fail_count = 0
    miss_count = 0

    for idx, item in enumerate(items, 1):
        scene_id = str(item.get("scene_id", "")).strip()
        if not scene_id:
            print(f"[WARN] 第 {idx} 条缺少 scene_id，跳过")
            fail_count += 1
            continue

        room_folder = get_room_folder(item)
        if not room_folder:
            print(f"[WARN] scene_id={scene_id} 缺少 empty_scene_room_id_folder，跳过")
            fail_count += 1
            continue

        scene_json_path = find_scene_json(scenes_root, room_folder, scene_id)
        if scene_json_path is None:
            print(f"[MISS] 找不到场景: room_folder={room_folder}, scene_id={scene_id}")
            miss_count += 1
            continue

        out_dir = output_root / room_folder / scene_id

        if SKIP_IF_EXISTS and already_rendered(out_dir):
            print(f"[SKIP] 已存在渲染结果: {out_dir}")
            skip_count += 1
            continue

        print(f"\n[{idx}/{len(items)}]")
        print(f"  scene_id   : {scene_id}")
        print(f"  room_folder: {room_folder}")
        print(f"  scene_json : {scene_json_path}")
        print(f"  out_dir    : {out_dir}")

        try:
            render_one_scene(scene_json_path, out_dir)
            print(f"[OK] 渲染完成: {scene_id}")
            success_count += 1
        except Exception as e:
            print(f"[FAIL] scene_id={scene_id}, error={e}")
            traceback.print_exc()
            fail_count += 1

    print("\n========== SUMMARY ==========")
    print(f"success : {success_count}")
    print(f"skip    : {skip_count}")
    print(f"missing : {miss_count}")
    print(f"fail    : {fail_count}")
    print(f"output  : {output_root}")


if __name__ == "__main__":
    main()