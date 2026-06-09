import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tqdm import tqdm


UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)


def _extract_model_id_from_jid(jid: Any) -> Optional[str]:
    """
    3D-FUTURE 的 object jid 常见形式：
      "<uuid>" 或 "<uuid>-...(scale/size)..."
    这里取第一个 '-' 前 36 长度 uuid 片段更稳：用 split('-') 会把 uuid 自身拆烂。
    """
    if not isinstance(jid, str) or not jid:
        return None

    # 取前 36 个字符尝试看是否 UUID
    head = jid[:36]
    if UUID_RE.match(head):
        return head

    # 兜底：有些 jid 可能就是 uuid
    if UUID_RE.match(jid):
        return jid

    return None


def _load_model_info(model_info_path: Path) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    返回: model_id -> (super_category, category)
    兼容两类字段命名：
      - "super-category" / "category"
      - "super_category" / "category"
    """
    with open(model_info_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

    if isinstance(data, dict):
        # 常见：{"model_id": {...}} 或 {"data":[...]}
        if "data" in data and isinstance(data["data"], list):
            items = data["data"]
        else:
            # 假设键就是 model_id
            for k, v in data.items():
                if isinstance(v, dict):
                    sc = v.get("super-category", v.get("super_category"))
                    cat = v.get("category")
                    mapping[str(k)] = (sc, cat)
            return mapping
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unsupported model_info.json structure: {type(data)}")

    # list[{...}]
    for it in items:
        if not isinstance(it, dict):
            continue
        mid = it.get("model_id") or it.get("id") or it.get("uid")
        if not mid:
            continue
        sc = it.get("super-category", it.get("super_category"))
        cat = it.get("category")
        mapping[str(mid)] = (sc, cat)

    return mapping


def _process_scene(scene: Dict[str, Any], mapping: Dict[str, Tuple[Optional[str], Optional[str]]], overwrite: bool) -> int:
    objs = scene.get("objects", [])
    if not isinstance(objs, list):
        return 0

    n = 0
    for obj in objs:
        if not isinstance(obj, dict):
            continue

        # 已经有了就跳过（避免破坏已有标签）
        if (not overwrite) and ("category" in obj or "super_category" in obj or "super-category" in obj):
            continue

        model_id = _extract_model_id_from_jid(obj.get("jid"))
        if not model_id:
            continue
        if model_id not in mapping:
            continue

        super_cat, cat = mapping[model_id]

        # 按你要求写字段名：super_category + category
        obj["super_category"] = super_cat
        obj["category"] = cat

        # 可选：清理旧字段（避免混用）；仅在 overwrite 时做
        if overwrite and "super-category" in obj:
            obj.pop("super-category", None)

        n += 1

    return n


def run_dir(
    model_info_json: str,
    scenes_dir: str,
    out_dir: Optional[str],
    recursive: bool,
    overwrite_fields: bool,
    inplace: bool,
) -> None:
    model_info_path = Path(model_info_json)
    scenes_root = Path(scenes_dir)

    mapping = _load_model_info(model_info_path)

    pattern = "**/*.json" if recursive else "*.json"
    files = sorted(scenes_root.glob(pattern))

    if inplace:
        out_root = scenes_root
    else:
        if out_dir is None:
            raise ValueError("--out_dir is required unless --inplace is set")
        out_root = Path(out_dir)
        out_root.mkdir(parents=True, exist_ok=True)

    total_tagged = 0
    for p in tqdm(files, desc="Add category/super-category", unit="scene"):
        rel = p.relative_to(scenes_root)
        out_path = (out_root / rel) if not inplace else p

        with open(p, "r", encoding="utf-8") as f:
            scene = json.load(f)

        tagged = _process_scene(scene, mapping=mapping, overwrite=overwrite_fields)
        total_tagged += tagged

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scene, f, ensure_ascii=False, indent=2)

    print(f"Done. scenes={len(files)}, objects_tagged={total_tagged}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_info_json",
        default="/home2/zhangjiawei/respace/dataset_3D_Front/3D-FUTURE-model/model_info.json",
        help="3D-FUTURE model_info.json 路径",
    )
    ap.add_argument(
        "--scenes_dir",
        default="/home2/zhangjiawei/respace/scenes_SSR",
        help="输入 scenes 目录",
    )
    ap.add_argument(
        "--out_dir",
        default="/home2/zhangjiawei/respace/scenes_SSR_v2",
        help="输出目录（非 inplace 时使用）",
    )
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--overwrite_fields", action="store_true", help="覆盖已有 category/super_category 字段")
    ap.add_argument("--inplace", action="store_true", help="原地覆盖写回 scenes_dir（危险）")
    args = ap.parse_args()

    run_dir(
        model_info_json=args.model_info_json,
        scenes_dir=args.scenes_dir,
        out_dir=None if args.inplace else args.out_dir,
        recursive=args.recursive,
        overwrite_fields=args.overwrite_fields,
        inplace=args.inplace,
    )


if __name__ == "__main__":
    main()