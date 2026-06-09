import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

OBJECT_KEYS = ["objects", "Objects", "object_list", "instance_list", "instances"]
ROOM_KEYS = ["room_id", "room_type", "roomType", "type"]

PAT = re.compile(r"washing[\s_\-]*machine", re.IGNORECASE)


def _iter_objects(scene: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for k in OBJECT_KEYS:
        v = scene.get(k)
        if isinstance(v, list):
            for obj in v:
                if isinstance(obj, dict):
                    yield obj
            return
    return


def _get_room(scene: Dict[str, Any]) -> str:
    for k in ROOM_KEYS:
        v = scene.get(k)
        if v is not None and str(v).strip():
            return str(v)
    return "unknown"


def _obj_text(obj: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in ("category", "class", "type", "name", "label", "model_id", "modelId", "asset_id", "assetId", "id", "desc"):
        v = obj.get(k)
        if v is None:
            continue
        if isinstance(v, (str, int, float)):
            parts.append(str(v))
    return " ".join(parts)


def scene_has_washing_machine(scene: Dict[str, Any]) -> bool:
    for obj in _iter_objects(scene):
        if PAT.search(_obj_text(obj)):
            return True
    return False


def main() -> None:
    root = Path("/home2/zhangjiawei/respace/scenes_filter")
    paths = sorted(root.rglob("*.json"))
    hits: List[Tuple[str, str]] = []

    for p in paths:
        try:
            scene = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(scene, dict):
            continue
        if scene_has_washing_machine(scene):
            hits.append((str(p), _get_room(scene)))

    for path, room in hits:
        print(f"{room}\t{path}")

    print(f"\nTOTAL_HITS: {len(hits)} / TOTAL_SCENES: {len(paths)}")


if __name__ == "__main__":
    main()