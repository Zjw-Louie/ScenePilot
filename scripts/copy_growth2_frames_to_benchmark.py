import argparse
import shutil
from pathlib import Path


def _unique_dest(dst_dir: Path, base_name: str) -> Path:
    cand = dst_dir / base_name
    if not cand.exists():
        return cand

    p = Path(base_name)
    stem = p.stem
    suffix = p.suffix
    i = 1
    while True:
        cand = dst_dir / f"{stem}__{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backup (copy) or move growth2_frames renders into benchmark/scenes_frame/{diag,top}."
    )
    ap.add_argument(
        "--src-root",
        type=Path,
        default=Path("/home2/zhangjiawei/respace/training_data/growth2_frames"),
        help="Source root containing <scene_id>/{diag,top}/*.jpg",
    )
    ap.add_argument(
        "--dst-root",
        type=Path,
        default=Path("/home2/zhangjiawei/respace/benchmark/scenes_frame"),
        help="Destination root containing diag/ and top/ subfolders",
    )
    ap.add_argument("--ext", type=str, default=".jpg", help="File extension to copy/move (default: .jpg)")
    ap.add_argument(
        "--step",
        type=str,
        default="step_05",
        help="Only handle images whose filename contains this token (default: step_05).",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="copy",
        choices=["copy", "move"],
        help="copy: backup to dst (default); move: move files to dst.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print planned operations without changing files")
    args = ap.parse_args()

    src_root: Path = args.src_root
    dst_root: Path = args.dst_root
    ext = args.ext.lower()
    step_token = args.step

    if not src_root.exists():
        raise SystemExit(f"[ERR] src-root not found: {src_root}")

    dst_diag = dst_root / "diag"
    dst_top = dst_root / "top"
    dst_diag.mkdir(parents=True, exist_ok=True)
    dst_top.mkdir(parents=True, exist_ok=True)

    op = shutil.copy2 if args.mode == "copy" else shutil.move

    handled = 0
    skipped = 0

    for scene_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        scene_id = scene_dir.name

        for view in ("diag", "top"):
            src_view_dir = scene_dir / view
            if not src_view_dir.exists():
                continue

            dst_view_dir = dst_diag if view == "diag" else dst_top

            for img in sorted(src_view_dir.glob(f"*{ext}")):
                if not img.is_file():
                    continue
                if step_token not in img.name:
                    continue

                base_name = f"{scene_id}__{img.name}"
                dst_path = _unique_dest(dst_view_dir, base_name)

                if args.dry_run:
                    print(f"[DRY] {args.mode.upper()} {img} -> {dst_path}")
                    handled += 1
                    continue

                try:
                    op(str(img), str(dst_path))
                    handled += 1
                except Exception as e:
                    print(f"[WARN] failed {args.mode}: {img} -> {dst_path} ({e})")
                    skipped += 1

    print(f"[DONE] mode={args.mode} handled={handled} skipped={skipped}")
    print(f"  dst_diag: {dst_diag}")
    print(f"  dst_top : {dst_top}")


if __name__ == "__main__":
    main()