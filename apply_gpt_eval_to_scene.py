from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_MESH_CANDIDATES = (
    "raw_model.glb",
    "normalized_model.glb",
    "model.glb",
    "raw_model.obj",
    "normalized_model.obj",
    "model.obj",
)


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fill_asset_paths(scene: Dict[str, Any], asset_root: Path) -> Dict[str, Any]:
    for obj in scene.get("objects", []):
        jid = obj.get("sampled_asset_jid") or obj.get("jid") or obj.get("sampled_jid")
        if not isinstance(jid, str) or not jid:
            continue
        if obj.get("asset_path") and Path(obj["asset_path"]).exists():
            continue

        found: Optional[Path] = None
        jid_dir = asset_root / jid
        if not jid_dir.exists():
            candidates = sorted(asset_root.glob(f"{jid}*"))
            if candidates:
                jid_dir = candidates[0]

        for mesh_name in _MESH_CANDIDATES:
            candidate = jid_dir / mesh_name
            if candidate.exists():
                found = candidate
                break

        if found is not None:
            obj["asset_path"] = str(found)

    return scene


def _apply_patch_to_scene(
    scene: Dict[str, Any], eval_json: Dict[str, Any]
) -> Tuple[Dict[str, Any], int, List[Dict[str, Any]]]:
    per = eval_json.get("per_object", [])
    if not isinstance(per, list):
        return scene, 0, []

    # 1) 精确映射：优先用 (jid, match_key) -> patch
    patches_by_jid_mk: Dict[Tuple[str, str], Dict[str, Any]] = {}
    # 2) 退化映射：只有 jid（无 match_key）的 patch，按顺序消费
    patches_by_jid_seq: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for it in per:
        if not isinstance(it, dict):
            continue
        jid = it.get("jid")
        mk = it.get("match_key")
        patch = it.get("patch")

        if not isinstance(jid, str) or not isinstance(patch, dict):
            continue

        if isinstance(mk, str) and mk:
            patches_by_jid_mk[(jid, mk)] = patch
        else:
            patches_by_jid_seq[jid].append(patch)

    consume_idx: Dict[str, int] = defaultdict(int)
    applied = 0
    changes: List[Dict[str, Any]] = []

    for obj in scene.get("objects", []):
        jid = obj.get("sampled_asset_jid") or obj.get("jid") or obj.get("sampled_jid")
        mk = obj.get("match_key")
        if not isinstance(jid, str):
            continue

        patch: Optional[Dict[str, Any]] = None

        # A) 先走精确匹配
        if isinstance(mk, str) and mk and (jid, mk) in patches_by_jid_mk:
            patch = patches_by_jid_mk[(jid, mk)]
        else:
            # B) 再走按 jid 顺序消费（兼容老 eval 输出不带 match_key）
            patch_list = patches_by_jid_seq.get(jid)
            if patch_list:
                idx = consume_idx[jid]
                if idx >= len(patch_list):
                    idx = len(patch_list) - 1
                patch = patch_list[idx]
                consume_idx[jid] += 1

        if not patch:
            continue

        pos = patch.get("pos")
        rot = patch.get("rot")

        before_pos = obj.get("pos")
        before_rot = obj.get("rot")

        if isinstance(pos, list) and len(pos) == 3:
            new_pos = [float(pos[0]), float(pos[1]), float(pos[2])]
            if obj.get("pos") != new_pos:
                obj["pos"] = new_pos
                applied += 1
                changes.append({
                    "jid": jid,
                    "match_key": mk,
                    "field": "pos",
                    "before": before_pos,
                    "after": new_pos,
                })

        if isinstance(rot, list) and len(rot) == 4:
            new_rot = [float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])]
            if obj.get("rot") != new_rot:
                obj["rot"] = new_rot
                applied += 1
                changes.append({
                    "jid": jid,
                    "match_key": mk,
                    "field": "rot",
                    "before": before_rot,
                    "after": new_rot,
                })

    return scene, applied, changes


def main() -> None:
    # 输入：scene + 你贴的 eval 文本（文件 or env）
    scene_json_path = Path(os.getenv("SCENE_JSON_PATH", "")).expanduser()
    eval_text_path = Path(os.getenv("EVAL_TEXT_PATH", "")).expanduser()
    out_scene_path = Path(os.getenv("OUT_SCENE_PATH", "./scene_after_gpt_eval.json")).expanduser()

    if not scene_json_path.exists():
        raise FileNotFoundError(f"SCENE_JSON_PATH not found: {scene_json_path}")
    if not eval_text_path.exists():
        raise FileNotFoundError(f"EVAL_TEXT_PATH not found: {eval_text_path}")

    scene = json.loads(scene_json_path.read_text(encoding="utf-8"))
    eval_text = eval_text_path.read_text(encoding="utf-8").strip()
    eval_json = json.loads(eval_text)

    # 可选：补 asset_path，避免后续 eval/渲染找不到 mesh
    asset_root = Path(os.getenv("ASSET_ROOT", "/home2/zhangjiawei/respace/dataset/3D-FUTURE-model"))
    scene = _fill_asset_paths(scene, asset_root=asset_root)

    scene2 = json.loads(json.dumps(scene))
    scene2, applied, changes = _apply_patch_to_scene(scene2, eval_json)

    _write_json(out_scene_path, scene2)
    _write_json(out_scene_path.with_suffix(".changes.json"), {"applied_fields": applied, "applied_changes": changes})

    print(f"written: {out_scene_path}")
    print(f"applied_fields: {applied}")
    print(f"changes: {out_scene_path.with_suffix('.changes.json')}")


if __name__ == "__main__":
    main()