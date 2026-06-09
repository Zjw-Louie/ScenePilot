from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

from tqdm import tqdm

from src.viz import render_annotated_top_view


def iter_step_jsons(root: Path):
    # 形如 scenes_growth/<scene_id>/step_01.json
    yield from sorted(root.rglob("step_*.json"))


def out_path_for(scene_json: Path, in_root: Path, out_root: Path) -> Path:
    # scene_json: <in_root>/<scene_id>/step_01.json
    rel = scene_json.relative_to(in_root)
    scene_id = rel.parts[0]
    step_name = scene_json.stem  # step_01
    return out_root / scene_id / "annotated_top" / f"{step_name}.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_root",
        type=Path,
        default=Path("/home2/zhangjiawei/respace/training_data/scenes_growth"),
    )
    ap.add_argument(
        "--out_root",
        type=Path,
        default=Path("/home2/zhangjiawei/respace/training_data/growth2_frames"),
    )
    ap.add_argument(
        "--assets_root",
        type=Path,
        default=None,
        help="3D-FUTURE 资产根目录（会写入环境变量 PTH_3DFUTURE_ASSETS）",
    )
    ap.add_argument("--resolution", type=int, nargs=2, default=(1024, 1024))
    ap.add_argument("--font_size", type=int, default=14)
    ap.add_argument("--show_assets", action="store_true", help="加载 mesh（缺资产会慢/可能失败）")
    args = ap.parse_args()

    in_root: Path = args.in_root
    out_root: Path = args.out_root
    if args.assets_root is not None:
        os.environ["PTH_3DFUTURE_ASSETS"] = str(args.assets_root)
    elif args.show_assets and not os.getenv("PTH_3DFUTURE_ASSETS"):
        print("[render_growth] WARN PTH_3DFUTURE_ASSETS not set; fallback to show_assets=False")
        args.show_assets = False

    scene_paths = list(iter_step_jsons(in_root))
    if not scene_paths:
        print(f"[render_growth] no step_*.json found under {in_root}")
        return

    for scene_json in tqdm(scene_paths, desc="render annotated_top"):
        scene = json.loads(scene_json.read_text(encoding="utf-8"))

        out_jpg = out_path_for(scene_json, in_root, out_root)
        out_dir = out_jpg.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # render_annotated_top_view 会自己创建 {pth_viz_output}/top-annotated/
        # 为了让它输出到你指定的 annotated_top 目录：传 pth_viz_output=out_dir.parent
        # 然后 filename 用 step_01，最后把生成文件名设为 step_01.jpg
        # 最简单：直接让 pth_viz_output=out_dir，并在 viz 内部保存到 top-annotated；
        # 所以这里用一个临时约定：把 annotated_top 当作 pth_viz_output，并把 top-annotated 目录改名为 annotated_top
        # ——不改 viz 的话，我们就渲染到 out_dir/../top-annotated，再移动/重命名到 annotated_top。
        tmp_root = out_dir.parent  # .../<scene_id>/
        tmp_root.mkdir(parents=True, exist_ok=True)

        tmp_path = render_annotated_top_view(
            scene,
            filename=scene_json.stem,  # step_01
            pth_viz_output=tmp_root,
            resolution=tuple(args.resolution),
            use_dynamic_zoom=True,
            camera_height=None,
            show_assets=args.show_assets,
            font_size=args.font_size,
            bg_color=None,
        )

        # tmp_path: .../<scene_id>/top-annotated/step_01.jpg
        # move to: .../<scene_id>/annotated_top/step_01.jpg
        if tmp_path is not None:
            out_jpg.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.replace(out_jpg)

    print("[render_growth] done")


if __name__ == "__main__":
    main()