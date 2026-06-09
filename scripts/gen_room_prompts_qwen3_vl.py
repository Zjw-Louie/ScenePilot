from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

EMPTY_SCENES_DIR = Path("/home2/zhangjiawei/respace/empty_scenes")
FRAMES_DIR = Path("/home2/zhangjiawei/respace/training_data/growth2_frames")
SCENES_JSON_DIR = Path("/home2/zhangjiawei/respace/scenes_hash2_with_cat")
MODEL_DIR = Path("/home2/zhangjiawei/respace/model/qwen3-vl-8B-instruct")

OUT_DIR = Path("/home2/zhangjiawei/respace/evaluate_date/room_prompts_qwen3_vl")
FRAME_STEP = "step_05.jpg"  # 你要的 step05（示例是 step_05.jpg）

MAX_NEW_TOKENS = 128
_ONE_LINE = re.compile(r"\s+")


def _scene_id_from_empty_scene(path: Path) -> str:
    return path.stem


def _split_from_empty_scene(path: Path) -> str:
    """
    按 empty_scenes 下的一级目录分：bedroom/livingroom/other
    若不在这三类，归为 other。
    """
    rel = path.relative_to(EMPTY_SCENES_DIR)
    parts = rel.parts
    if len(parts) >= 2:
        cand = parts[0].lower()
        if cand in {"bedroom", "livingroom", "other"}:
            return cand
    return "other"


def _frame_path(scene_id: str, view: str) -> Optional[Path]:
    """
    growth2_frames/<scene_id>/<view>/step_05.jpg
    view in {"diag","top"}
    """
    p = FRAMES_DIR / scene_id / view / FRAME_STEP
    return p if p.exists() else None


def _load_scene_json(scene_id: str) -> Optional[Dict[str, Any]]:
    p = SCENES_JSON_DIR / f"{scene_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _compact_scene(scene: Dict[str, Any], max_objects: int = 80) -> Dict[str, Any]:
    objs = scene.get("objects", [])
    out_objs: List[Dict[str, Any]] = []
    if isinstance(objs, list):
        for i, o in enumerate(objs[:max_objects]):
            if not isinstance(o, dict):
                continue
            out_objs.append(
                {
                    "idx": i,
                    "category": o.get("category") or o.get("type"),
                    "super_category": o.get("super-category") or o.get("super_category"),
                    "desc": o.get("desc") or o.get("description") or o.get("style_description"),
                    "size": o.get("size"),
                }
            )
    return {
        "room_type": scene.get("room_type"),
        "bounds_bottom": scene.get("bounds_bottom"),
        "objects": out_objs,
    }


def _make_instruction(scene_compact: Dict[str, Any]) -> str:
    scene_str = json.dumps(scene_compact, ensure_ascii=False, separators=(",", ":"))
    return f"""You are an expert interior designer prompt writer.

Task:
Write ONE single-line English prompt describing the room layout and key furniture.

Format requirements:
- Output ONE line only. No quotes, no bullet points.
- Start with: "create a <style> <room_type> include ..."
- Mention 6-14 key objects (most salient).
- Avoid ids/jids and avoid coordinates.

SCENE_JSON:
{scene_str}
"""


def _normalize_one_line(text: str) -> str:
    text = text.strip().replace("```", "")
    return _ONE_LINE.sub(" ", text).strip()

def _load_done_scene_ids(all_jsonl: Path) -> set[str]:
    done: set[str] = set()
    if not all_jsonl.exists():
        return done
    try:
        with all_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                sid = rec.get("scene_id")
                if isinstance(sid, str) and sid:
                    done.add(sid)
    except Exception:
        pass
    return done


def main() -> None:
    empty_scene_paths = sorted(EMPTY_SCENES_DIR.rglob("*.json"))
    if not empty_scene_paths:
        raise SystemExit(f"empty_scenes 下未找到 json: {EMPTY_SCENES_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_out_path = OUT_DIR / "all_prompts.jsonl"
    done_scene_ids = _load_done_scene_ids(all_out_path)
    if done_scene_ids:
        print(f"found existing outputs: {len(done_scene_ids)} scene_ids in {all_out_path}")

    # === Qwen3-VL transformers 推理 ===
    from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore
    import torch  # type: ignore
    from PIL import Image  # type: ignore

    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_DIR),
        device_map="auto",
        trust_remote_code=True,
    )

    # 追加写，避免覆盖历史结果
    outs = {
        "bedroom": (OUT_DIR / "bedroom_prompts.jsonl").open("a", encoding="utf-8"),
        "livingroom": (OUT_DIR / "livingroom_prompts.jsonl").open("a", encoding="utf-8"),
        "other": (OUT_DIR / "other_prompts.jsonl").open("a", encoding="utf-8"),
        "all": all_out_path.open("a", encoding="utf-8"),
    }

    written = 0
    skipped = 0
    skipped_done = 0

    try:
        for empty_path in tqdm(empty_scene_paths, desc="gen prompts", unit="scene"):
            scene_id = _scene_id_from_empty_scene(empty_path)
            if scene_id in done_scene_ids:
                skipped_done += 1
                continue

            split = _split_from_empty_scene(empty_path)

            diag = _frame_path(scene_id, "diag")
            top = _frame_path(scene_id, "top")
            scene = _load_scene_json(scene_id)

            if diag is None or top is None or scene is None:
                skipped += 1
                continue

            instruction = _make_instruction(_compact_scene(scene))
            diag_img = Image.open(diag).convert("RGB")
            top_img = Image.open(top).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": diag_img},
                        {"type": "image", "image": top_img},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]

            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = processor(
                text=[prompt_text],
                images=[[diag_img, top_img]],
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

            input_len = inputs["input_ids"].shape[-1]
            gen_ids = out[0][input_len:]
            text = processor.decode(gen_ids, skip_special_tokens=True)
            prompt = _normalize_one_line(text)

            rec = {
                "scene_id": scene_id,
                "split": split,
                "prompt": prompt,
                "diag_path": str(diag),
                "top_path": str(top),
                "scene_json_path": str(SCENES_JSON_DIR / f"{scene_id}.json"),
                "empty_scene_path": str(empty_path),
            }

            line = json.dumps(rec, ensure_ascii=False) + "\n"
            outs["all"].write(line)
            outs[split].write(line)
            outs["all"].flush()
            outs[split].flush()

            done_scene_ids.add(scene_id)
            written += 1
    finally:
        for f in outs.values():
            f.close()

    print(
        f"done. written={written} skipped_missing={skipped} skipped_done={skipped_done} out_dir={OUT_DIR}"
    )


if __name__ == "__main__":
    main()