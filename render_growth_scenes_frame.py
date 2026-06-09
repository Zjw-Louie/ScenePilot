import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.respace import ReSpace


def iter_scene_files(root: Path, recursive: bool = True):
    pattern = "**/step_*.json" if recursive else "step_*.json"
    yield from sorted(p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() == ".json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes_root", default="/home2/zhangjiawei/respace/training_data/scenes_growth")
    ap.add_argument(
        "--out_root",
        default=None,
        help="输出根目录；默认写到每个 step 文件同级目录下的 renders/",
    )
    ap.add_argument("--recursive", action="store_true", default=True)
    ap.add_argument("--no_recursive", dest="recursive", action="store_false")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scenes_root = Path(args.scenes_root)
    out_root = Path(args.out_root) if args.out_root else None

    respace = ReSpace()

    files = list(iter_scene_files(scenes_root, recursive=args.recursive))
    if not files:
        print(f"No step_*.json found under {scenes_root}")
        return

    for scene_path in tqdm(files, desc="Render frames", unit="scene"):
        with open(scene_path, "r", encoding="utf-8") as f:
            scene = json.load(f)

        scene_dir = scene_path.parent
        render_dir = (out_root / scene_dir.relative_to(scenes_root) if out_root else (scene_dir / "renders"))
        render_dir.mkdir(parents=True, exist_ok=True)

        filename = scene_path.stem  # step_01

        expected_diag = render_dir / "diag" / f"{filename}.jpg"
        expected_top = render_dir / "top" / f"{filename}.jpg"
        if not args.overwrite and expected_diag.exists() and expected_top.exists():
            tqdm.write(f"Skip existing: {expected_diag} and {expected_top}")
            continue

        respace.render_scene_frame(
            scene,
            filename=filename,
            pth_viz_output=render_dir,
        )


if __name__ == "__main__":
    main()