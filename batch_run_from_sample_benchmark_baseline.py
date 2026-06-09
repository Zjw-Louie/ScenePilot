import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield ln, json.loads(s)
            except Exception as e:
                raise ValueError(f"Invalid JSON at line {ln}: {e}") from e


def index_empty_scenes(root: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in root.rglob("*.json"):
        # scene_id is filename stem
        idx.setdefault(p.stem, p)
    return idx


def pick_scene_id(rec: Dict[str, Any]) -> Optional[str]:
    for k in ("scene_id", "sceneId", "id"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def pick_prompt(rec: Dict[str, Any]) -> Optional[str]:
    # try common keys
    for k in ("prompt", "room_prompt", "ROOM_PROMPT", "text"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # sometimes prompt nested (best-effort)
    v = rec.get("data")
    if isinstance(v, dict):
        for k in ("prompt", "room_prompt", "text"):
            vv = v.get(k)
            if isinstance(vv, str) and vv.strip():
                return vv.strip()
    return None


def load_latest_status(results_path: Path) -> Dict[str, str]:
    """
    scene_id -> latest status from existing results.jsonl (last occurrence wins).
    Lines that are not valid JSON are ignored.
    """
    if not results_path.exists():
        return {}
    last: Dict[str, str] = {}
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            sid = rec.get("scene_id")
            status = rec.get("status")
            if isinstance(sid, str) and isinstance(status, str):
                last[sid] = status
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch run ReSpace from sample_benchmark_by_ratio_vlm.jsonl")
    ap.add_argument(
        "--in-jsonl",
        type=Path,
        default="/home2/zhangjiawei/respace/benchmark/sample_benchmark_by_ratio_vlm.jsonl",
        help="Input jsonl containing scene_id + prompt.",
    )
    ap.add_argument(
        "--empty-root",
        type=Path,
        default="/home2/zhangjiawei/respace/benchmark/empty_scenes",
        help="Root dir containing empty scenes (recursive).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default="/home2/zhangjiawei/respace/results/batch_outputs_baseline_456",
        help="Output directory for updated scenes and run logs.",
    )
    ap.add_argument("--limit", type=int, default=0, help="If >0, only run first N records.")
    ap.add_argument("--resume", action="store_true", help="Skip scenes that already have output.")
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="When used with --resume: only skip scenes whose latest status is ok; re-run model_failed/exception/etc.",
    )
    ap.add_argument(
        "--render-frame",
        action="store_true",
        help="Render a single RGB frame for each updated scene (respace.render_scene_frame).",
    )
    ap.add_argument(
        "--render-top",
        action="store_true",
        help="Render annotated top view for each updated scene (render_annotated_top_view).",
    )
    args = ap.parse_args()

    if not args.in_jsonl.exists():
        raise SystemExit(f"[ERR] in-jsonl not found: {args.in_jsonl}")
    if not args.empty_root.exists():
        raise SystemExit(f"[ERR] empty-root not found: {args.empty_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_scenes_dir = args.out_dir / "updated_scenes"
    out_scenes_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = args.out_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = args.out_dir / "results.jsonl"

    # Only needed when retrying: judge whether last run was ok
    last_status = load_latest_status(run_log_path) if (args.resume and args.retry_failed) else {}

    # Import here so env is set up before loading heavy deps
    from src.respace import ReSpace  # type: ignore
    from src.viz import render_annotated_top_view  # type: ignore

    respace = ReSpace()
    empty_idx = index_empty_scenes(args.empty_root)

    n_total = 0
    n_ok = 0
    n_fail = 0
    n_skip = 0
    t0 = time.time()

    with run_log_path.open("a", encoding="utf-8") as flog:
        for ln, rec in load_jsonl(args.in_jsonl):
            n_total += 1
            if args.limit > 0 and n_total > args.limit:
                break

            scene_id = pick_scene_id(rec)
            prompt = pick_prompt(rec)

            if not scene_id or not prompt:
                n_fail += 1
                flog.write(
                    json.dumps(
                        {
                            "line": ln,
                            "status": "bad_record",
                            "scene_id": scene_id,
                            "has_prompt": bool(prompt),
                            "record": rec,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            scene_path = empty_idx.get(scene_id)
            if scene_path is None:
                n_fail += 1
                flog.write(
                    json.dumps(
                        {
                            "line": ln,
                            "status": "missing_scene",
                            "scene_id": scene_id,
                            "prompt": prompt,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            out_scene_path = out_scenes_dir / f"{scene_id}_updated.json"
            per_scene_render_dir = renders_dir / scene_id

            if args.resume and out_scene_path.exists():
                if args.retry_failed:
                    prev = last_status.get(scene_id)
                    if prev == "ok":
                        n_skip += 1
                        flog.write(
                            json.dumps(
                                {
                                    "line": ln,
                                    "status": "skipped_existing_ok",
                                    "scene_id": scene_id,
                                    "prev_status": prev,
                                    "scene_path": str(scene_path),
                                    "out_scene_path": str(out_scene_path),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue
                else:
                    n_skip += 1
                    flog.write(
                        json.dumps(
                            {
                                "line": ln,
                                "status": "skipped_existing",
                                "scene_id": scene_id,
                                "scene_path": str(scene_path),
                                "out_scene_path": str(out_scene_path),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

            try:
                scene = json.loads(scene_path.read_text(encoding="utf-8"))
                updated_scene, is_success = respace.handle_prompt(prompt, scene)

                out_scene_path.write_text(
                    json.dumps(updated_scene, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                render_errors = []
                if args.render_frame or args.render_top:
                    per_scene_render_dir.mkdir(parents=True, exist_ok=True)

                # 1) render one frame
                if args.render_frame:
                    try:
                        respace.render_scene_frame(
                            updated_scene, filename="frame", pth_viz_output=per_scene_render_dir
                        )
                    except Exception as e:
                        render_errors.append({"what": "render_scene_frame", "error": repr(e)})

                # 2) render annotated top view
                if args.render_top:
                    try:
                        render_annotated_top_view(
                            updated_scene,
                            filename="frame_annotated_top",
                            pth_viz_output=per_scene_render_dir,
                            resolution=(1024, 1024),
                            use_dynamic_zoom=True,
                            camera_height=None,
                            show_assets=True,
                            font_size=14,
                            bg_color=None,
                        )
                    except Exception as e:
                        render_errors.append({"what": "render_annotated_top_view", "error": repr(e)})

                status = "ok" if is_success else "model_failed"
                if is_success:
                    n_ok += 1
                else:
                    n_fail += 1

                flog.write(
                    json.dumps(
                        {
                            "line": ln,
                            "status": status,
                            "scene_id": scene_id,
                            "prompt": prompt,
                            "scene_path": str(scene_path),
                            "out_scene_path": str(out_scene_path),
                            "is_success": bool(is_success),
                            "render_dir": str(per_scene_render_dir) if (args.render_frame or args.render_top) else None,
                            "render_errors": render_errors,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                # Update in-memory latest status so later duplicates in the same run are handled
                if args.resume and args.retry_failed:
                    last_status[scene_id] = status

            except Exception as e:
                n_fail += 1
                flog.write(
                    json.dumps(
                        {
                            "line": ln,
                            "status": "exception",
                            "scene_id": scene_id,
                            "prompt": prompt,
                            "scene_path": str(scene_path),
                            "out_scene_path": str(out_scene_path),
                            "error": repr(e),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if args.resume and args.retry_failed:
                    last_status[scene_id] = "exception"

    dt = time.time() - t0
    print("[DONE]")
    print(f"  in_jsonl   : {args.in_jsonl}")
    print(f"  empty_root : {args.empty_root}")
    print(f"  out_dir    : {args.out_dir}")
    print(f"  total      : {n_total}")
    print(f"  ok         : {n_ok}")
    print(f"  fail       : {n_fail}")
    print(f"  skip       : {n_skip}")
    print(f"  seconds    : {dt:.1f}")
    print(f"  results    : {run_log_path}")


if __name__ == "__main__":
    main()