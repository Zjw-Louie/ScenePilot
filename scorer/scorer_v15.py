from __future__ import annotations

import base64
import json
import math
import os
import re
import torch
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from PIL import Image
from requests.adapters import HTTPAdapter


@dataclass
class GPTMovePromptV5Result:
    raw_text: str
    move_prompt: str
    json_text: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GPTRelationPriorsV5Result:
    raw_text: str
    json_text: str


@dataclass
class ObjectEdit:
    jid_prefix: str = ""
    description: str = ""
    object_index: Optional[int] = None
    hint_pos: Optional[List[float]] = None
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    target_yaw_deg: Optional[float] = None
    relative_yaw_deg: Optional[float] = None
    scale: Optional[Union[float, List[float]]] = None
    scale_delta: Optional[Union[float, List[float]]] = None
    no_movement: bool = False
    no_rotation: bool = False
    no_scale: bool = False
    raw_line: str = ""


@dataclass
class MovePromptParseResult:
    room_name: str = ""
    edits: List[ObjectEdit] = field(default_factory=list)
    header_line: str = ""
    parse_warnings: List[str] = field(default_factory=list)


_ACTION_EPS = 1e-9

_BAD_PROMPT_MARKERS: Tuple[str, ...] = (
    "retrieved group priors",
    "global scene requirement:",
    "room-level priors:",
    "anchor-level priors:",
    "now perform scene planning",
    "existing scene objects:",
    "return only one json object",
    "do not output markdown",
    "do not output explanations",
    "do not use markdown fences",
    "user request:",
    "existing objects:",
    "add object:",
    "role:",
    "zone hint:",
    "allowed actions:",
    "current scene json:",
)

_PROMPT_INSTRUCTION_CUES: Tuple[str, ...] = (
    "user request:",
    "existing objects:",
    "add object:",
    "role:",
    "zone hint:",
    "return only one json object",
    "do not output",
)


def _extract_requested_object_prompt(text: Any) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    flat = re.sub(r"\s+", " ", s)
    patterns = [
        r"Add object:\s*(.+?)(?=\s*(?:Role:|Zone hint:|Existing objects:|$))",
        r"add\s+(.+?)\s+as\s+the\s+anchor\s+object\b",
        r"add\s+(.+?)\s+as\s+part\s+of\s+the\b",
        r"add\s+(.+?)\s+around\s+the\s+anchor\b",
    ]
    for pat in patterns:
        m = re.search(pat, flat, flags=re.I)
        if not m:
            continue
        val = m.group(1).strip(" .,:;\"'")
        if val:
            return val.lower()
    return ""


def _flatten_object_entries(entries: Any) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    if not isinstance(entries, list):
        return flat
    for item in entries:
        if not isinstance(item, dict):
            continue
        nested = item.get("objects")
        if isinstance(nested, list) and nested:
            flat.extend(_flatten_object_entries(nested))
            continue
        flat.append(item)
    return flat


def _normalized_scene_objects(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _flatten_object_entries(scene.get("objects", []))


def _looks_like_planning_blob(text: Any) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    low = s.lower()
    if any(marker in low for marker in _BAD_PROMPT_MARKERS):
        return True
    if "\n" in s:
        return True
    cue_hits = sum(1 for cue in _PROMPT_INSTRUCTION_CUES if cue in low)
    if cue_hits >= 2:
        return True
    if len(s) > 80 and cue_hits >= 1:
        return True
    return len(s) > 120


def _short_object_prompt_from_obj(obj: Dict[str, Any]) -> str:
    raw_prompt = str(obj.get("prompt", "") or "").strip()
    extracted = _extract_requested_object_prompt(raw_prompt)
    if extracted:
        return extracted
    raw_saved = str(obj.get("planning_prompt_raw", "") or "").strip()
    extracted_saved = _extract_requested_object_prompt(raw_saved)
    if extracted_saved:
        return extracted_saved
    for k in ("category", "type"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    for k in ("description", "style_description", "desc", "sampled_asset_desc"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            s = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff \-_]+", " ", v).strip().lower()
            words = s.split()
            if words:
                return " ".join(words[:4])
    return "object"


def _safe_prompt_for_compact_scene(obj: Dict[str, Any]) -> str:
    p = obj.get("prompt")
    if not isinstance(p, str):
        return _short_object_prompt_from_obj(obj)
    p = p.strip()
    if not p or _looks_like_planning_blob(p):
        return _short_object_prompt_from_obj(obj)
    return p


def _scene_signature(scene: Dict[str, Any]) -> str:
    return json.dumps(scene, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _yaw_from_obj(obj: Dict[str, Any]) -> float:
    rot = obj.get("rot", [0.0, 0.0, 0.0, 1.0])
    if isinstance(rot, list) and len(rot) == 4:
        return yaw_from_quaternion(rot)
    return 0.0



def _compact_scene_for_prompt(scene: Dict[str, Any]) -> Dict[str, Any]:
    objects: List[Dict[str, Any]] = []
    for i, obj in enumerate(_normalized_scene_objects(scene)):
        objects.append(
            {
                "object_index": i,
                "idx": i,
                "jid": obj.get("sampled_asset_jid") or obj.get("jid") or obj.get("sampled_jid"),
                "category": obj.get("category") or obj.get("type"),
                "desc": obj.get("description") or obj.get("style_description") or obj.get("desc"),
                "prompt": _safe_prompt_for_compact_scene(obj),
                "pos": obj.get("pos"),
                "rot": obj.get("rot"),
                "yaw_deg": round(_yaw_from_obj(obj), 3),
                "size": obj.get("size"),
                "sampled_asset_size": obj.get("sampled_asset_size"),
                "scale": obj.get("scale"),
            }
        )
    return {
        "room_type": scene.get("room_type"),
        "bounds_bottom": scene.get("bounds_bottom"),
        "bounds_top": scene.get("bounds_top"),
        "objects": objects,
    }


@lru_cache(maxsize=64)
def _img_to_data_url_cached(path_str: str, mtime_ns: int, size: int) -> str:
    p = Path(path_str)
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    suffix = p.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"



def _img_to_data_url(p: Path) -> str:
    stat = p.stat()
    return _img_to_data_url_cached(str(p.resolve()), stat.st_mtime_ns, stat.st_size)



def _post_chat_completions(
    api_base: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout_s: float,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    http = session or requests
    resp = http.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()



def _strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned



def _extract_json_text(text: str) -> str:
    cleaned = _strip_markdown_fence(text)
    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(cleaned[i:])
            return cleaned[i : i + end]
        except Exception:
            continue
    raise ValueError("Model output does not contain valid JSON.")



def _normalize_relation_priors_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list):
        return {"relations": payload}
    if isinstance(payload, dict):
        if isinstance(payload.get("relations"), list):
            return payload
        for key in ("relation_priors", "priors", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return {"relations": value}
        return {"relations": []}
    return {"relations": []}



def _normalize_actions_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"actions": []}

    actions = payload.get("actions")
    if not isinstance(actions, list):
        return {"actions": []}

    normalized: List[Dict[str, Any]] = []
    for item in actions:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip().lower()
        obj_idx = item.get("object_index")
        if not isinstance(obj_idx, int):
            try:
                obj_idx = int(obj_idx)
            except Exception:
                continue

        if action == "move":
            normalized.append(
                {
                    "action": "move",
                    "object_index": obj_idx,
                    "dx": float(item.get("dx", 0.0)),
                    "dy": float(item.get("dy", 0.0)),
                    "dz": float(item.get("dz", 0.0)),
                }
            )
        elif action == "rotate":
            normalized.append(
                {
                    "action": "rotate",
                    "object_index": obj_idx,
                    "yaw_deg": float(item.get("yaw_deg", 0.0)),
                }
            )
        elif action == "scale":
            normalized.append(
                {
                    "action": "scale",
                    "object_index": obj_idx,
                    "sx": float(item.get("sx", 1.0)),
                    "sy": float(item.get("sy", 1.0)),
                    "sz": float(item.get("sz", 1.0)),
                }
            )
    return {"actions": normalized}



def yaw_from_quaternion(q: List[float]) -> float:
    x, y, z, w = q
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (y * y + x * x)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))



def quaternion_from_yaw(yaw_deg: float) -> List[float]:
    yaw_rad = math.radians(yaw_deg)
    return [0.0, math.sin(yaw_rad / 2.0), 0.0, math.cos(yaw_rad / 2.0)]



def _make_object_label(obj: Dict[str, Any], index: int, counts: Dict[str, int], seen: Dict[str, int]) -> str:
    jid = obj.get("sampled_asset_jid") or obj.get("jid") or obj.get("sampled_jid") or "unknown"
    prefix = jid[:6].lower() if len(jid) >= 6 else jid.lower()

    category = obj.get("category") or obj.get("type") or ""
    if not category:
        desc = obj.get("description") or obj.get("style_description") or obj.get("desc") or ""
        words = desc.split()[:3]
        category = " ".join(words) if words else f"Object_{index}"

    pos = obj.get("pos", [0.0, 0.0, 0.0])
    x = float(pos[0]) if len(pos) > 0 else 0.0
    z = float(pos[2]) if len(pos) > 2 else 0.0

    if counts.get(prefix, 1) > 1:
        rank = seen.get(prefix, 0) + 1
        seen[prefix] = rank
        return f"[{index}] {category} #{rank} ({prefix}…) at ({x:.2f}, {z:.2f})"
    return f"[{index}] {category} ({prefix}…) at ({x:.2f}, {z:.2f})"



def build_labeled_scene_summary(scene: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    objs = _normalized_scene_objects(scene)
    counts: Dict[str, int] = {}
    for obj in objs:
        jid = obj.get("sampled_asset_jid") or obj.get("jid") or obj.get("sampled_jid") or ""
        prefix = jid[:6].lower() if len(jid) >= 6 else jid.lower()
        counts[prefix] = counts.get(prefix, 0) + 1

    seen: Dict[str, int] = {}
    lines: List[str] = []
    labeled: List[Dict[str, Any]] = []
    for i, obj in enumerate(objs):
        label = _make_object_label(obj, i, counts, seen)
        pos = obj.get("pos", [0.0, 0.0, 0.0])
        rot = obj.get("rot", [0.0, 0.0, 0.0, 1.0])
        yaw = yaw_from_quaternion(rot) if isinstance(rot, list) and len(rot) == 4 else 0.0
        scale = obj.get("scale")
        scale_str = ""
        if isinstance(scale, list) and len(scale) == 3:
            scale_str = f" | scale=[{float(scale[0]):.2f}, {float(scale[1]):.2f}, {float(scale[2]):.2f}]"
        lines.append(
            f"  - {label} | pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | yaw={yaw:.1f}°{scale_str}"
        )
        labeled.append({**obj, "_label": label, "object_index": i})
    return "OBJECTS IN SCENE:\n" + "\n".join(lines), labeled


_RE_JID = re.compile(r"\(([a-f0-9]{6})[^)]*\)", re.IGNORECASE)
_RE_AT_POS = re.compile(
    r"at\s*\(\s*([+\-−]?\d+(?:\.\d+)?)\s*,\s*([+\-−]?\d+(?:\.\d+)?)\s*(?:,\s*([+\-−]?\d+(?:\.\d+)?))?\s*\)",
    re.IGNORECASE,
)
_RE_MOVE_SEGMENT = re.compile(r"move\s+([+\-−]?\s*\d+(?:\.\d+)?)\s*m\s+along\s+([+\-−]?[XYZxyz])", re.IGNORECASE)
_RE_ROT_ABSOLUTE = re.compile(r"rotate\s+to\s+(?:face\s+)?([+\-−]?\d+(?:\.\d+)?)\s*°?", re.IGNORECASE)
_RE_ROT_RELATIVE = re.compile(r"rotate\s+([+\-−]?\d+(?:\.\d+)?)\s*°?\s*(clockwise|counter[- ]?clockwise|ccw|cw)?", re.IGNORECASE)

_RE_SCALE_TO_VEC = re.compile(
    r"scale\s+to\s*\(\s*([+\-−]?\d+(?:\.\d+)?)\s*,\s*([+\-−]?\d+(?:\.\d+)?)\s*,\s*([+\-−]?\d+(?:\.\d+)?)\s*\)",
    re.IGNORECASE,
)
_RE_SCALE_BY_VEC = re.compile(
    r"scale\s+(?:by|delta)\s*\(\s*([+\-−]?\d+(?:\.\d+)?)\s*,\s*([+\-−]?\d+(?:\.\d+)?)\s*,\s*([+\-−]?\d+(?:\.\d+)?)\s*\)\s*x?",
    re.IGNORECASE,
)
_RE_SCALE_TO_UNI = re.compile(r"scale\s+to\s+([+\-−]?\d+(?:\.\d+)?)\s*x?", re.IGNORECASE)
_RE_SCALE_BY_UNI = re.compile(r"scale\s+(?:by|delta)\s+([+\-−]?\d+(?:\.\d+)?)\s*x?", re.IGNORECASE)

_RE_NO_MOVEMENT = re.compile(r"no\s+movement\s+needed", re.IGNORECASE)
_RE_NO_ROTATION = re.compile(r"no\s+rotation\s+needed", re.IGNORECASE)
_RE_NO_SCALE = re.compile(r"no\s+scale\s+needed", re.IGNORECASE)



def _parse_sign_number(text: str) -> float:
    return float(text.strip().replace("−", "-").replace(" ", ""))



def _parse_axis_sign(text: str) -> Tuple[str, float]:
    text = text.strip().replace("−", "-")
    if text.startswith("+"):
        return text[1:].upper(), 1.0
    if text.startswith("-"):
        return text[1:].upper(), -1.0
    return text.upper(), 1.0



def _parse_scale_triplet(a: str, b: str, c: str) -> List[float]:
    return [_parse_sign_number(a), _parse_sign_number(b), _parse_sign_number(c)]



def _payload_to_edits(payload: Dict[str, Any]) -> Tuple[List[ObjectEdit], List[str]]:
    edits: List[ObjectEdit] = []
    warnings: List[str] = []
    actions = payload.get("actions", [])
    if not isinstance(actions, list):
        return edits, ["payload.actions is not a list"]

    for action_item in actions:
        if not isinstance(action_item, dict):
            warnings.append(f"skip non-dict action: {action_item!r}")
            continue

        action = str(action_item.get("action", "")).strip().lower()
        obj_idx = action_item.get("object_index")
        if not isinstance(obj_idx, int):
            try:
                obj_idx = int(obj_idx)
            except Exception:
                warnings.append(f"skip action without valid object_index: {action_item!r}")
                continue

        if action == "move":
            dx = float(action_item.get("dx", 0.0))
            dy = float(action_item.get("dy", 0.0))
            dz = float(action_item.get("dz", 0.0))
            if abs(dx) <= _ACTION_EPS and abs(dy) <= _ACTION_EPS and abs(dz) <= _ACTION_EPS:
                continue
            edits.append(
                ObjectEdit(
                    object_index=obj_idx,
                    description=f"obj_{obj_idx}",
                    dx=dx,
                    dy=dy,
                    dz=dz,
                    raw_line=json.dumps(action_item, ensure_ascii=False),
                )
            )
        elif action == "rotate":
            yaw_deg = float(action_item.get("yaw_deg", 0.0))
            if abs(yaw_deg) <= _ACTION_EPS:
                continue
            edits.append(
                ObjectEdit(
                    object_index=obj_idx,
                    description=f"obj_{obj_idx}",
                    relative_yaw_deg=yaw_deg,
                    raw_line=json.dumps(action_item, ensure_ascii=False),
                )
            )
        elif action == "scale":
            sx = float(action_item.get("sx", 1.0))
            sy = float(action_item.get("sy", 1.0))
            sz = float(action_item.get("sz", 1.0))
            if abs(sx - 1.0) <= _ACTION_EPS and abs(sy - 1.0) <= _ACTION_EPS and abs(sz - 1.0) <= _ACTION_EPS:
                continue
            edits.append(
                ObjectEdit(
                    object_index=obj_idx,
                    description=f"obj_{obj_idx}",
                    scale_delta=[sx, sy, sz],
                    raw_line=json.dumps(action_item, ensure_ascii=False),
                )
            )
        else:
            warnings.append(f"unknown action type: {action}")
    return edits, warnings



def parse_move_prompt(text: str) -> MovePromptParseResult:
    result = MovePromptParseResult()

    try:
        json_text = _extract_json_text(text)
        payload = _normalize_actions_payload(json.loads(json_text))
        edits, warnings = _payload_to_edits(payload)
        result.header_line = "json_actions"
        result.room_name = ""
        result.edits = edits
        result.parse_warnings.extend(warnings)
        return result
    except Exception:
        pass

    lines = text.strip().splitlines()
    object_lines: List[str] = []
    found_header = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not found_header:
            lowered = stripped.lower()
            if "coordinate convention" in lowered or "precise pos + rot edits" in lowered:
                result.header_line = stripped
                found_header = True
                continue
            if "adjust" in lowered and "edits" in lowered:
                result.header_line = stripped
                found_header = True
                continue
            continue

        if not result.room_name and not _RE_JID.search(stripped):
            result.room_name = stripped
            continue

        if _RE_JID.search(stripped):
            object_lines.append(stripped)
        elif object_lines:
            object_lines[-1] += " " + stripped
        else:
            result.parse_warnings.append(f"unrecognized line before any object: {stripped}")

    for line in object_lines:
        jid_match = _RE_JID.search(line)
        if not jid_match:
            result.parse_warnings.append(f"no jid found: {line}")
            continue

        edit = ObjectEdit(
            jid_prefix=jid_match.group(1).lower(),
            description=line[: jid_match.start()].strip().rstrip("(").strip(),
            raw_line=line,
        )

        at_match = _RE_AT_POS.search(line)
        if at_match:
            x = _parse_sign_number(at_match.group(1))
            y_or_z = _parse_sign_number(at_match.group(2))
            if at_match.group(3) is not None:
                z = _parse_sign_number(at_match.group(3))
                edit.hint_pos = [x, y_or_z, z]
            else:
                edit.hint_pos = [x, 0.0, y_or_z]

        if _RE_NO_MOVEMENT.search(line):
            edit.no_movement = True
        else:
            for match in _RE_MOVE_SEGMENT.finditer(line):
                amount = _parse_sign_number(match.group(1))
                axis, sign = _parse_axis_sign(match.group(2))
                delta = amount * sign
                if axis == "X":
                    edit.dx += delta
                elif axis == "Y":
                    edit.dy += delta
                elif axis == "Z":
                    edit.dz += delta

        if _RE_NO_ROTATION.search(line):
            edit.no_rotation = True
        else:
            abs_match = _RE_ROT_ABSOLUTE.search(line)
            if abs_match:
                edit.target_yaw_deg = _parse_sign_number(abs_match.group(1))
            else:
                rel_match = _RE_ROT_RELATIVE.search(line)
                if rel_match:
                    deg = _parse_sign_number(rel_match.group(1))
                    direction = (rel_match.group(2) or "").lower().replace(" ", "").replace("-", "")
                    if "counter" in direction or "ccw" in direction:
                        deg = -abs(deg)
                    elif "clockwise" in direction or "cw" in direction:
                        deg = abs(deg)
                    edit.relative_yaw_deg = deg

        if _RE_NO_SCALE.search(line):
            edit.no_scale = True
        else:
            m = _RE_SCALE_TO_VEC.search(line)
            if m:
                edit.scale = _parse_scale_triplet(m.group(1), m.group(2), m.group(3))
            else:
                m = _RE_SCALE_BY_VEC.search(line)
                if m:
                    edit.scale_delta = _parse_scale_triplet(m.group(1), m.group(2), m.group(3))
                else:
                    m = _RE_SCALE_TO_UNI.search(line)
                    if m:
                        edit.scale = _parse_sign_number(m.group(1))
                    else:
                        m = _RE_SCALE_BY_UNI.search(line)
                        if m:
                            edit.scale_delta = _parse_sign_number(m.group(1))

        result.edits.append(edit)

    return result



def _pos_distance_xz(a: List[float], b: List[float]) -> float:
    return math.hypot(a[0] - b[0], (a[2] if len(a) > 2 else 0.0) - (b[2] if len(b) > 2 else 0.0))



def _normalize_scale_value(value: Optional[Union[float, List[float]]]) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        s = float(value)
        return [s, s, s]
    if isinstance(value, list) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    return None



def _parse_scale_from_jid(jid: Any) -> List[float]:
    if not isinstance(jid, str) or not jid:
        return [1.0, 1.0, 1.0]
    m = re.search(r"-\(([-+]?\d*\.?\d+)\)-\(([-+]?\d*\.?\d+)\)-\(([-+]?\d*\.?\d+)\)$", jid)
    if not m:
        return [1.0, 1.0, 1.0]
    return [float(m.group(1)), float(m.group(2)), float(m.group(3))]



def _recover_base_size(obj: Dict[str, Any], current_scale: List[float]) -> Optional[List[float]]:
    for key in ("base_size", "canonical_size", "asset_base_size"):
        value = obj.get(key)
        if isinstance(value, list) and len(value) >= 3:
            return [float(value[0]), float(value[1]), float(value[2])]

    size = obj.get("size")
    if isinstance(size, list) and len(size) >= 3:
        out: List[float] = []
        for s, c in zip(size[:3], current_scale):
            c = float(c)
            out.append(float(s) / c if abs(c) > 1e-8 else float(s))
        return out

    sampled = obj.get("sampled_asset_size")
    if isinstance(sampled, list) and len(sampled) >= 3:
        return [float(sampled[0]), float(sampled[1]), float(sampled[2])]

    return None



def _apply_scale_edit(obj: Dict[str, Any], edit: ObjectEdit) -> Optional[Dict[str, Any]]:
    if edit.no_scale:
        return None

    current_scale = _normalize_scale_value(obj.get("scale"))
    if current_scale is None:
        jid = obj.get("sampled_asset_jid") or obj.get("jid") or obj.get("sampled_jid")
        current_scale = _parse_scale_from_jid(jid)

    base_size = _recover_base_size(obj, current_scale)
    if base_size is None:
        return None

    scale_abs = _normalize_scale_value(edit.scale)
    scale_delta = _normalize_scale_value(edit.scale_delta)

    if scale_abs is not None:
        new_scale = scale_abs
    elif scale_delta is not None:
        new_scale = [
            current_scale[0] * scale_delta[0],
            current_scale[1] * scale_delta[1],
            current_scale[2] * scale_delta[2],
        ]
    else:
        return None

    new_scale = [max(0.05, float(v)) for v in new_scale]
    new_size = [
        base_size[0] * new_scale[0],
        base_size[1] * new_scale[1],
        base_size[2] * new_scale[2],
    ]

    before_scale = list(current_scale)
    before_size = list(obj.get("size", base_size))

    obj["base_size"] = list(base_size)
    obj["scale"] = list(new_scale)
    obj["size"] = list(new_size)

    return {
        "field": "scale_and_size",
        "before_scale": before_scale,
        "after_scale": obj["scale"],
        "before_size": before_size,
        "after_size": obj["size"],
        "base_size": base_size,
    }



def apply_edits_to_scene(scene: Dict[str, Any], edits: List[ObjectEdit]) -> Tuple[Dict[str, Any], int, List[Dict[str, Any]]]:
    objects = scene.get("objects", [])
    applied = 0
    changes: List[Dict[str, Any]] = []

    direct_pairs: List[Tuple[ObjectEdit, int]] = []
    legacy_edits: List[ObjectEdit] = []
    for edit in edits:
        if edit.object_index is not None:
            if 0 <= int(edit.object_index) < len(objects):
                direct_pairs.append((edit, int(edit.object_index)))
            continue
        legacy_edits.append(edit)

    edits_by_prefix: Dict[str, List[ObjectEdit]] = {}
    for edit in legacy_edits:
        edits_by_prefix.setdefault(edit.jid_prefix, []).append(edit)

    objs_by_prefix: Dict[str, List[int]] = {}
    for i, obj in enumerate(objects):
        jid = obj.get("sampled_asset_jid") or obj.get("jid") or obj.get("sampled_jid")
        if isinstance(jid, str) and len(jid) >= 6:
            objs_by_prefix.setdefault(jid[:6].lower(), []).append(i)

    legacy_pairs: List[Tuple[ObjectEdit, int]] = []
    for prefix, edit_list in edits_by_prefix.items():
        obj_indices = list(objs_by_prefix.get(prefix, []))
        if not obj_indices:
            continue

        available = set(obj_indices)
        with_pos = [e for e in edit_list if e.hint_pos is not None]
        without_pos = [e for e in edit_list if e.hint_pos is None]

        for edit in with_pos:
            if not available:
                break
            best_idx = min(
                available,
                key=lambda oi: _pos_distance_xz(edit.hint_pos or [0.0, 0.0, 0.0], objects[oi].get("pos", [0.0, 0.0, 0.0])),
            )
            legacy_pairs.append((edit, best_idx))
            available.discard(best_idx)

        for edit, oi in zip(without_pos, sorted(available)):
            legacy_pairs.append((edit, oi))

    for edit, obj_idx in direct_pairs + legacy_pairs:
        obj = objects[obj_idx]
        before_pos = list(obj.get("pos", [0.0, 0.0, 0.0]))
        before_rot = list(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
        changed = False

        if not edit.no_movement and (abs(edit.dx) > _ACTION_EPS or abs(edit.dy) > _ACTION_EPS or abs(edit.dz) > _ACTION_EPS):
            obj["pos"] = [before_pos[0] + edit.dx, before_pos[1] + edit.dy, before_pos[2] + edit.dz]
            changes.append(
                {
                    "obj_index": obj_idx,
                    "field": "pos",
                    "delta": [edit.dx, edit.dy, edit.dz],
                    "before": before_pos,
                    "after": obj["pos"],
                }
            )
            changed = True

        if not edit.no_rotation:
            target_yaw = None
            if edit.target_yaw_deg is not None:
                target_yaw = edit.target_yaw_deg
            elif edit.relative_yaw_deg is not None:
                target_yaw = yaw_from_quaternion(before_rot) + edit.relative_yaw_deg
            if target_yaw is not None:
                obj["rot"] = quaternion_from_yaw(target_yaw)
                changes.append(
                    {
                        "obj_index": obj_idx,
                        "field": "rot",
                        "before": before_rot,
                        "after": obj["rot"],
                        "target_yaw_deg": target_yaw,
                    }
                )
                changed = True

        scale_change = _apply_scale_edit(obj, edit)
        if scale_change is not None:
            changes.append({"obj_index": obj_idx, **scale_change})
            changed = True

        if changed:
            applied += 1

    return scene, applied, changes


try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None

def _load_local_vlm_cls():
    """
    尽量兼容你当前环境里的 transformers。
    优先尝试 Qwen3VLForConditionalGeneration，
    不行再退回通用 Auto 类。
    """
    try:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    except Exception:
        pass

    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    except Exception:
        pass

    try:
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq
    except Exception:
        pass

    raise ImportError(
        "Cannot import a local multimodal generation model class. "
        "Please check your transformers installation."
    )


class GPTVLMovePromptGeneratorV5:
    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 120.0,
        relation_vocab_mode: str = "canonical_v1",
        allow_legacy_relation_types: bool = False,
        backend: Optional[str] = None,                  # 新增
        device: Optional[str] = None,                  # 新增
        torch_dtype: str = "bfloat16",                 # 新增
        max_new_tokens: int = 1200,                    # 新增
        local_base_model: Optional[str] = None,        # 新增：adapter 时可指定 base
    ) -> None:
        self.model = model
        self.relation_vocab_mode = relation_vocab_mode
        self.allow_legacy_relation_types = allow_legacy_relation_types
        self.timeout_s = timeout_s

        self._prompt_cache: Dict[str, str] = {}
        self._relation_prompt_cache: Dict[str, str] = {}

        # backend 自动判断：如果 model 是本地路径，就默认 local
        if backend is None:
            backend = "local" if os.path.exists(model) else "api"
        self.backend = backend

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = int(max_new_tokens)
        self.local_base_model = local_base_model or os.getenv("LOCAL_VLM_BASE_MODEL")

        self.session = None
        self.api_base = None
        self.api_key = None
        self.processor = None
        self.local_model = None

        if self.backend == "api":
            self.api_base = (api_base or os.getenv("YUNWU_AI_API_BASE") or "").rstrip("/")
            self.api_key = api_key or os.getenv("YUNWU_AI_API_KEY")

            self.session = requests.Session()
            adapter = HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=0)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)

            if not self.api_key:
                raise RuntimeError("Missing YUNWU_AI_API_KEY.")
            if not self.api_base:
                raise RuntimeError("Missing YUNWU_AI_API_BASE.")

        elif self.backend == "local":
            if AutoProcessor is None:
                raise ImportError("transformers.AutoProcessor is not available.")

            model_cls = _load_local_vlm_cls()

            # 判断当前目录是不是 adapter/LoRA
            adapter_cfg = os.path.join(model, "adapter_config.json")
            is_adapter = os.path.exists(adapter_cfg)

            if is_adapter and not self.local_base_model:
                raise RuntimeError(
                    f"{model} looks like an adapter checkpoint "
                    f"(adapter_config.json exists), but LOCAL_VLM_BASE_MODEL/local_base_model is not set."
                )

            base_for_processor = self.local_base_model or os.getenv(
                "LOCAL_VLM_BASE_MODEL",
                "/home2/zhangjiawei/respace/model/qwen3-vl-8B-instruct",
            )

            # processor 优先从 base model 读
            processor_source = base_for_processor

            # model:
            # 1) adapter -> 先读 base，再挂 adapter
            # 2) merged/full -> 直接读当前 model 目录
            model_source = self.local_base_model if is_adapter else model

            self.processor = AutoProcessor.from_pretrained(
                processor_source,
                trust_remote_code=True,
            )

            dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16

            base_model = model_cls.from_pretrained(
                model_source,
                torch_dtype=dtype,
                device_map="auto" if self.device.startswith("cuda") else None,
                trust_remote_code=True,
            )

            if is_adapter:
                from peft import PeftModel
                base_model = PeftModel.from_pretrained(base_model, model)

            self.local_model = base_model.eval()

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _build_relation_type_block(self) -> str:
        canonical_lines = [
            "- distance_band",
            "- facing",
            "- centered_with",
            "- against_wall",
            "- parallel",
        ]
        if self.allow_legacy_relation_types:
            canonical_lines.extend(["- near", "- facing_pair", "- in_front_of", "- side_of"])
        elif self.relation_vocab_mode == "canonical_v1":
            canonical_lines.append("- side_of  # optional; only use for strongly typed side-support relations")
        return "Allowed relation types only:\n" + "\n".join(canonical_lines)

    def _build_prompt(self, scene: Dict[str, Any], extra_context: str) -> str:
        object_summary, _ = build_labeled_scene_summary(scene)
        scene_str = json.dumps(_compact_scene_for_prompt(scene), ensure_ascii=False, separators=(",", ":"))
        room_type = scene.get("room_type") or "room"
        return f"""You are a professional 3D indoor scene repair agent.

You will be given:
- Image 1: diagonal view render
- Image 2: top-down annotated render
- A compact scene JSON with exact pos/rot/scale values
- A labeled object list with object_index

Task:
Repair the current {room_type} scene. Improve physical plausibility and functionality with minimal edits.

Priority order:
1. Fix out-of-bounds.
2. Fix collisions / overlaps.
3. Improve reachability and circulation.
4. Improve functional grouping and orientation.
5. Use scale only if clearly necessary.

Return ONLY one JSON object with key `actions`.
Do not output scene JSON.
Do not copy or rewrite the input scene.
Do not output explanations.
Do not use markdown fences.
Use `object_index` to identify objects. `object_index` is the index of the object in Current scene JSON["objects"].
If no edit is needed, output exactly: {{"actions":[]}}

Allowed actions:
- move: {{"action":"move","object_index":0,"dx":0.0,"dy":0.0,"dz":0.0}}
- rotate: {{"action":"rotate","object_index":0,"yaw_deg":0.0}}
- scale: {{"action":"scale","object_index":0,"sx":1.0,"sy":1.0,"sz":1.0}}

Action semantics:
- move uses relative translation in meters.
- rotate uses relative yaw in degrees. Positive values mean clockwise rotation in the top view.
- scale uses multiplicative factors. (1.0, 1.0, 1.0) means no scale change.

Editing rules:
- Prefer minimal but effective edits.
- Usually return at most 1-3 actions for one step.
- Keep dominant anchors stable when possible; adjust accessories first if that solves the issue.
- Do not invent, delete, or replace objects.
- Do not edit objects that are already reasonable.
- Round dx/dy/dz to 0.05 m when possible.
- Round yaw_deg to 5 degrees when possible.
- Only use scale when the object is clearly too large or too small for its local context.
- Avoid simultaneous unnecessary move+rotate+scale on the same object unless clearly needed.

Room type: {room_type}

ADDITIONAL CONTEXT:
{extra_context.strip() if extra_context else '(none)'}

OBJECTS:
{object_summary}

Current scene JSON:
{scene_str}
"""

    def _build_relation_priors_prompt(self, scene: Dict[str, Any], extra_context: str) -> str:
        object_summary, _ = build_labeled_scene_summary(scene)
        scene_str = json.dumps(_compact_scene_for_prompt(scene), ensure_ascii=False, separators=(",", ":"))
        return f"""You are a professional 3D indoor scene layout analyst.

You will be given:
- Image 1: diagonal view render
- Image 2: top-down annotated render
- A compact scene JSON with exact pos/rot values
- A labeled object list

Goal:
Infer a small set of high-confidence relation priors that describe the intended functional layout.

{self._build_relation_type_block()}

Output must be strict JSON, with no markdown fence and no commentary:
{{
  "relations": [
    {{"src_idx": 0, "tgt_idx": 1, "type": "near", "confidence": 0.82, "weight": 1.0, "reason": "supporting object near dominant anchor"}},
    {{"src_idx": 2, "type": "against_wall", "confidence": 0.91, "weight": 1.1, "reason": "wall-affine dominant anchor"}}
  ]
}}

Rules:
- Use object indices from SCENE_JSON.
- Use at most 3 relations per source object.
- Prefer high-confidence, functionally meaningful relations only.
- Do not invent missing objects.
- For against_wall and parallel, omit tgt_idx.
- confidence should be in [0.0, 1.0].
- weight should usually be in [0.5, 1.5].
- If uncertain, return fewer relations, even an empty list.

Heuristics:
- Prefer canonical optimization-friendly relations over linguistically vague ones.
- Use distance_band instead of near unless legacy mode explicitly allows near.
- Use facing instead of facing_pair unless legacy mode explicitly allows facing_pair.
- Avoid in_front_of unless legacy mode explicitly allows it and the front direction is visually obvious.
- Use side_of only for strongly typed side-support relations such as nightstand-bed or side-table-sofa.
- Wall-affine anchors often include beds, wardrobes, cabinets, shelves, TV stands, consoles, sinks, vanities, toilets, bathtubs, counters, and appliances.
- Seating objects often relate to desks, tables, counters, vanities, or sofas.
- Small support accessories often stay near a dominant anchor.
- Scene organization matters more than isolated pairwise guesses.
- Prefer generalized functional relations that can transfer across bedrooms, kitchens, bathrooms, and living rooms.

OBJECTS:
{object_summary}

ADDITIONAL CONTEXT:
{extra_context.strip() if extra_context else '(none)'}

SCENE_JSON:
{scene_str}
"""

    def _chat_api(
        self,
        *,
        diag_image_path: Path,
        top_image_path: Path,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if not diag_image_path.exists():
            raise FileNotFoundError(diag_image_path)
        if not top_image_path.exists():
            raise FileNotFoundError(top_image_path)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _img_to_data_url(diag_image_path)}},
                        {"type": "image_url", "image_url": {"url": _img_to_data_url(top_image_path)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        data = _post_chat_completions(
            self.api_base,
            self.api_key,
            payload,
            self.timeout_s,
            session=self.session,
        )
        return data["choices"][0]["message"]["content"]

    def _chat_local(
        self,
        *,
        diag_image_path: Path,
        top_image_path: Path,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if not diag_image_path.exists():
            raise FileNotFoundError(diag_image_path)
        if not top_image_path.exists():
            raise FileNotFoundError(top_image_path)

        diag_img = Image.open(diag_image_path).convert("RGB")
        top_img = Image.open(top_image_path).convert("RGB")

        # 本地 Qwen-VL 常见输入格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": diag_img},
                    {"type": "image", "image": top_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[[diag_img, top_img]],
            padding=True,
            return_tensors="pt",
        )

        target_device = None
        if hasattr(self.local_model, "device"):
            target_device = self.local_model.device
        elif torch.cuda.is_available():
            target_device = torch.device("cuda")
        else:
            target_device = torch.device("cpu")

        for k, v in inputs.items():
            if hasattr(v, "to"):
                inputs[k] = v.to(target_device)

        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "do_sample": float(temperature) > 0,
        }
        if float(temperature) > 0:
            gen_kwargs["temperature"] = float(temperature)

        with torch.no_grad():
            output_ids = self.local_model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[:, prompt_len:]
        output_text = self.processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()

    def _chat(
        self,
        *,
        diag_image_path: Path,
        top_image_path: Path,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if self.backend == "local":
            return self._chat_local(
                diag_image_path=diag_image_path,
                top_image_path=top_image_path,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self._chat_api(
            diag_image_path=diag_image_path,
            top_image_path=top_image_path,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate(
        self,
        *,
        diag_image_path: Path,
        top_image_path: Path,
        scene: Dict[str, Any],
        extra_context: str = "",
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> GPTMovePromptV5Result:
        cache_key = _scene_signature(scene) + "\n<context>\n" + extra_context
        prompt = self._prompt_cache.get(cache_key)
        if prompt is None:
            prompt = self._build_prompt(scene, extra_context)
            self._prompt_cache[cache_key] = prompt

        raw_text = self._chat(
            diag_image_path=diag_image_path,
            top_image_path=top_image_path,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        json_text = _extract_json_text(raw_text)
        payload = _normalize_actions_payload(json.loads(json_text))
        normalized_json_text = json.dumps(payload, ensure_ascii=False)
        return GPTMovePromptV5Result(
            raw_text=raw_text,
            move_prompt=normalized_json_text,
            json_text=normalized_json_text,
            payload=payload,
        )

    def generate_relation_priors(
        self,
        *,
        diag_image_path: Path,
        top_image_path: Path,
        scene: Dict[str, Any],
        extra_context: str = "",
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> GPTRelationPriorsV5Result:
        cache_key = _scene_signature(scene) + "\n<relation_context>\n" + extra_context
        prompt = self._relation_prompt_cache.get(cache_key)
        if prompt is None:
            prompt = self._build_relation_priors_prompt(scene, extra_context)
            self._relation_prompt_cache[cache_key] = prompt

        raw_text = self._chat(
            diag_image_path=diag_image_path,
            top_image_path=top_image_path,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        json_text = _extract_json_text(raw_text)
        payload = _normalize_relation_priors_payload(json.loads(json_text))
        normalized_json_text = json.dumps(payload, ensure_ascii=False)
        return GPTRelationPriorsV5Result(
            raw_text=raw_text,
            json_text=normalized_json_text,
        )