from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from src.eval import eval_scene
from src.respace import ReSpace


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    scene_path = Path(os.getenv("SCENE_PATH", "./scene_after_gpt_eval.json")).expanduser()
    out_dir = Path(os.getenv("OUT_DIR", "./evaluate/scene_after_gpt_eval")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not scene_path.exists():
        raise FileNotFoundError(f"SCENE_PATH not found: {scene_path}")

    scene: Dict[str, Any] = json.loads(scene_path.read_text(encoding="utf-8"))

    # 确保 eval_scene 找得到 mesh（你现在的路径是 /home2/...，这里保持一致）
    if not os.getenv("PTH_3DFUTURE_ASSETS"):
        os.environ["PTH_3DFUTURE_ASSETS"] = "/home2/zhangjiawei/respace/dataset/3D-FUTURE-model"

    # 1) render
    respace = ReSpace()
    respace.render_scene_frame(scene, filename="scene_after_gpt_eval", pth_viz_output=out_dir)

    diag_path = out_dir / "diag" / "scene_after_gpt_eval.jpg"
    top_path = out_dir / "top" / "scene_after_gpt_eval.jpg"

    # 2) metrics
    metrics = eval_scene(scene, is_debug=False)
    _write_json(out_dir / "metrics.json", metrics)

    print(f"rendered diag: {diag_path if diag_path.exists() else 'MISSING'}")
    print(f"rendered top:  {top_path  if top_path.exists()  else 'MISSING'}")
    print(f"total_oob_loss: {metrics.get('total_oob_loss')}")
    print(f"total_mbl_loss: {metrics.get('total_mbl_loss')}")
    print(f"total_pbl_loss: {metrics.get('total_pbl_loss')}")
    print(f"is_valid_scene_pbl: {metrics.get('is_valid_scene_pbl')}")
    print(f"metrics saved: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()