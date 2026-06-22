from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from src.eval import create_floor_plan_polygon, compute_oob, eval_scene, get_xz_bbox_from_obj
from src.group_repair_in_loop import attach_group_repair_in_loop
from src.respace import ReSpace
from src.viz import render_annotated_top_view
from scorer.scorer_v15 import (
    GPTVLMovePromptGeneratorV5,
    ObjectEdit,
    apply_edits_to_scene,
    parse_move_prompt,
    quaternion_from_yaw,
    yaw_from_quaternion,
)
from scorer.scene_role_layout import (
    REL_TYPES,
    RoleGraph,
    build_role_based_relation_priors,
    compute_functional_loss,
    distance_to_nearest_wall_xz,
    infer_role_graph,
    obj_diag_size_xz,
    optimization_stage_order,
    post_refine_role_layout,
    room_center_xz,
    target_pose_for_attachment,
    xz_dist,
)


REL_CANONICAL_TYPES: Set[str] = {"distance_band", "facing", "centered_with", "against_wall", "parallel", "side_of"}
REL_LEGACY_ALIAS_TYPES: Set[str] = {"near", "facing_pair", "in_front_of"}
_V15_SCHEMA = "scene_actions_v1"
_V15_ALLOWED_ACTIONS = {"move", "rotate", "scale"}
_BOOL_FALSE = {"0", "false", "False", ""}

_W_PBL, _W_REL, _W_FUNC = 0.50, 0.20, 0.30


@dataclass
class TimingStats:
    render_sec: float = 0.0
    vlm_sec: float = 0.0
    optimize_sec: float = 0.0
    eval_sec: float = 0.0


@dataclass
class Config:
    max_steps: int = 3
    max_rounds: int = 2
    max_objects_per_round: int = 8
    proxy_topk: int = 3
    move_prompt_temperature: float = 0.2
    move_prompt_max_tokens: int = 1200
    move_prompt_retries: int = 2
    relation_prior_retries: int = 2
    relation_prior_temperature: float = 0.0
    relation_prior_max_tokens: int = 900
    relation_prior_confidence: float = 0.55
    use_vlm_relation_priors: bool = True
    refresh_vlm_relation_priors_every_step: bool = False
    merge_vlm_with_deterministic: bool = True
    vlm_prior_weight_scale: float = 0.35
    freeze_deterministic_priors: bool = False
    local_repair_passes: int = 1
    full_repair_after_refine_passes: int = 1
    stop_when_valid_pbl: bool = True
    cleanup_only_after_valid_pbl: bool = True
    max_steps_after_valid_pbl: int = 1
    max_objects_after_valid_pbl: int = 4
    max_rounds_after_valid_pbl: int = 1
    min_score_improve_after_valid_pbl: float = 0.02
    max_rel_increase_after_valid_pbl: float = 0.12
    max_func_increase_after_valid_pbl: float = 0.12
    max_score_increase_prevalid: float = 0.06
    max_rel_increase_prevalid: float = 0.55
    max_func_increase_prevalid: float = 0.35
    render_final: bool = True
    monotonic_eps: float = 1e-12
    step_xy: float = 0.22
    step_yaw: float = 15.0
    anchor_lock_pbl_threshold: float = 0.08
    role_refine_blend: float = 0.66
    valid_pbl_threshold: float = 0.10
    stop_score_threshold: float = 0.80
    skip_post_refine_when_valid_pbl: bool = True
    candidate_filter_reintroduced_pbl: bool = True
    use_structural_guard: bool = True
    min_open_space_ratio: float = 0.42
    max_zone_monopoly_ratio: float = 0.72
    corridor_width_ratio: float = 0.18
    max_open_space_drop_after_valid_pbl: float = 0.06
    max_monopoly_increase_after_valid_pbl: float = 0.08
    max_spread_increase_after_valid_pbl: float = 0.10
    max_flow_increase_after_valid_pbl: float = 0.10
    require_zone_count_preserve_after_valid_pbl: bool = True
    add_zone_release_candidates: bool = True
    zone_release_inset_ratio: float = 0.16
    enable_zone_layout_candidates: bool = True
    zone_layout_trigger_ratio: float = 0.76
    max_zone_layout_candidates: int = 4
    zone_layout_topk_objects: int = 3
    zone_layout_min_score_improve: float = 0.01
    zone_layout_anchor_inset_ratio: float = 0.14
    zone_layout_secondary_cluster_radius: float = 0.55
    zone_layout_pair_table_chair: bool = True
    use_iteration_history: bool = True
    max_history_steps: int = 4
    repeat_reject_patience: int = 2
    adaptive_relation_tradeoff: bool = True
    rewrite_priors_on_relayout: bool = True
    refresh_vlm_relation_priors_on_stagnation: bool = True
    force_history_replan_on_monopoly: bool = True
    relayout_accept_min_score_gain: float = 0.03
    relayout_accept_min_structure_gain: float = 0.01
    relayout_accept_min_monopoly_gain: float = 0.15
    relayout_accept_min_zone_gain: int = 1
    enable_dual_judge: bool = True
    stable_judge_weight: float = 0.75
    judge_min_score_improve_after_valid_pbl: float = 0.01
    judge_max_score_increase_after_valid_pbl: float = 0.02
    judge_max_score_increase_prevalid: float = 0.08
    proposal_branch_search: bool = True
    enable_stable_branch: bool = True
    enable_dynamic_det_branch: bool = True
    enable_dynamic_vlm_branch: bool = True
    mandatory_final_polish: bool = True
    final_polish_passes: int = 2
    use_structured_repair_plan: bool = True
    structured_plan_max_actions: int = 6
    structured_plan_apply_before_search: bool = True
    structured_plan_force_json: bool = True
    relation_refresh_mode: str = "once"
    relation_use_mode: str = "canonical_merged"
    relation_canonicalization_mode: str = "canonical_v1"
    relation_allow_side_of: bool = False
    relation_drop_in_front_of: bool = True
    relation_expand_facing_pair: bool = True
    relation_alias_near_to_band: bool = True
    relation_debug_write_raw: bool = True
    relation_debug_write_canonical: bool = True
    use_zero_shot_relation_plan: bool = True
    zero_shot_relation_use_mode: str = "canonical_merged"
    zero_shot_relation_weight_scale: float = 0.85


# ------------------------------
# basic utils
# ------------------------------

def _now() -> float:
    return time.perf_counter()


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _deepcopy_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(scene)


def _env_bool(v: str) -> bool:
    return str(v) not in _BOOL_FALSE


def _scene_state_hash(scene: Dict[str, Any]) -> str:
    payload = json.dumps(scene, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _get_float_metric(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    v = metrics.get(key)
    return float(v) if isinstance(v, (int, float)) else default


def _is_valid_pbl_value(pbl: float, cfg: Config) -> bool:
    return pbl <= cfg.valid_pbl_threshold


def _role_graph_dict(g: RoleGraph) -> Dict[str, Any]:
    return {
        "categories": g.categories,
        "role_by_idx": g.role_by_idx,
        "function_by_idx": g.function_by_idx,
        "zone_by_idx": g.zone_by_idx,
        "accessory_to_anchor": g.accessory_to_anchor,
        "notes": g.notes,
    }


def _config_from_env() -> Config:
    kwargs: Dict[str, Any] = {}
    for f in fields(Config):
        raw = os.getenv(f.name.upper())
        if raw is None:
            continue
        if isinstance(f.default, bool):
            kwargs[f.name] = _env_bool(raw)
        elif isinstance(f.default, int):
            kwargs[f.name] = int(raw)
        elif isinstance(f.default, float):
            kwargs[f.name] = float(raw)
        else:
            kwargs[f.name] = raw
    return Config(**kwargs)


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


def _is_nested_group_wrapper(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("objects"), list) and len(obj.get("objects") or []) > 0


def _flatten_object_entries(entries: Any) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    if not isinstance(entries, list):
        return flat

    WRAPPER_COPY_KEYS = (
        "sampled_asset_jid",
        "sampled_asset_desc",
        "sampled_asset_size",
        "uuid",
        "jid",
        "sampled_jid",
    )

    for item in entries:
        if not isinstance(item, dict):
            continue

        if _is_nested_group_wrapper(item):
            children = _flatten_object_entries(item.get("objects") or [])

            # 关键修复：
            # 如果 wrapper 里只有一个真实 leaf object，
            # 且 sampled_* / uuid 挂在 wrapper 上，就把这些字段转移到 leaf 上。
            if len(children) == 1 and isinstance(children[0], dict):
                child = children[0]

                for k in WRAPPER_COPY_KEYS:
                    if k in item and k not in child:
                        child[k] = copy.deepcopy(item[k])

                # 有些场景 wrapper 上还有 prompt / planning_prompt_raw，
                # 叶子没有时也补过去，但不要覆盖叶子已有值
                for k in ("prompt", "planning_prompt_raw"):
                    if k in item and (k not in child or not child.get(k)):
                        child[k] = copy.deepcopy(item[k])

            flat.extend(children)
            continue

        flat.append(item)

    return flat


def _normalize_scene_after_generation(scene: Dict[str, Any], keep_raw: bool = True) -> Dict[str, Any]:
    sc = _deepcopy_scene(scene)
    sc["objects"] = _flatten_object_entries(sc.get("objects", []))
    sc = _sanitize_prompt_fields_recursive(sc, keep_raw=keep_raw)
    return sc


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


def _safe_object_prompt_text(obj: Dict[str, Any]) -> str:
    p = obj.get("prompt")
    if not isinstance(p, str):
        return ""
    p = p.strip()
    if not p or _looks_like_planning_blob(p):
        return ""
    return p


def _short_object_prompt_from_obj(obj: Dict[str, Any]) -> str:
    raw_prompt = str(obj.get("prompt", "") or "").strip()
    extracted = _extract_requested_object_prompt(raw_prompt)
    if extracted:
        return extracted

    raw_saved = str(obj.get("planning_prompt_raw", "") or "").strip()
    extracted_saved = _extract_requested_object_prompt(raw_saved)
    if extracted_saved:
        return extracted_saved

    for k in ("category", "type", "super_category"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()

    for k in ("desc", "sampled_asset_desc"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            s = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff \-_]+", " ", v).strip().lower()
            words = s.split()
            if words:
                return " ".join(words[:4])

    return "object"


def _is_object_like_dict(d: Dict[str, Any]) -> bool:
    if not isinstance(d, dict):
        return False
    object_keys = {"desc", "prompt", "sampled_asset_desc", "sampled_asset_jid", "jid", "uuid", "pos", "rot", "size"}
    return len(object_keys.intersection(d.keys())) >= 3


def _sanitize_prompt_fields_recursive(node: Any, keep_raw: bool = True) -> Any:
    if isinstance(node, dict):
        if _is_object_like_dict(node):
            raw_prompt = str(node.get("prompt", "") or "").strip()
            if _looks_like_planning_blob(raw_prompt):
                if keep_raw and raw_prompt:
                    node["planning_prompt_raw"] = raw_prompt
                node["prompt"] = _short_object_prompt_from_obj(node)
        for k, v in list(node.items()):
            node[k] = _sanitize_prompt_fields_recursive(v, keep_raw=keep_raw)
        return node
    if isinstance(node, list):
        return [_sanitize_prompt_fields_recursive(x, keep_raw=keep_raw) for x in node]
    return node


def _sanitize_scene_object_prompts(scene: Dict[str, Any], keep_raw: bool = True) -> Dict[str, Any]:
    return _normalize_scene_after_generation(scene, keep_raw=keep_raw)



def _looks_like_scene_dict(x: Any) -> bool:
    return isinstance(x, dict) and isinstance(x.get("objects"), list)


def _sanitize_scene_like_value(value: Any, keep_raw: bool = True) -> Any:
    if _looks_like_scene_dict(value):
        return _sanitize_scene_object_prompts(value, keep_raw=keep_raw)
    if isinstance(value, tuple):
        if value and _looks_like_scene_dict(value[0]):
            value = list(value)
            value[0] = _sanitize_scene_object_prompts(value[0], keep_raw=keep_raw)
            return tuple(value)
        return value
    if isinstance(value, list):
        if value and _looks_like_scene_dict(value[0]):
            value = list(value)
            value[0] = _sanitize_scene_object_prompts(value[0], keep_raw=keep_raw)
            return value
        return value
    return value


def _install_respace_sanitize_wrappers(respace: ReSpace) -> None:
    for method_name in ("handle_prompt", "handle_prompt_group_repair_in_loop"):
        original = getattr(respace, method_name, None)
        if original is None or not callable(original):
            continue
        if getattr(original, "_infer_v15_scene_sanitize_wrapped", False):
            continue

        def _wrapped(*args: Any, __original=original, __method_name=method_name, **kwargs: Any) -> Any:
            sanitized_args = tuple(
                _sanitize_scene_object_prompts(arg, keep_raw=True) if _looks_like_scene_dict(arg) else arg
                for arg in args
            )
            sanitized_kwargs = {
                k: (_sanitize_scene_object_prompts(v, keep_raw=True) if _looks_like_scene_dict(v) else v)
                for k, v in kwargs.items()
            }
            result = __original(*sanitized_args, **sanitized_kwargs)
            result = _sanitize_scene_like_value(result, keep_raw=True)
            if os.getenv("INFER_V15_SANITIZE_DEBUG", "0") not in _BOOL_FALSE:
                _log(f"[sanitize wrapper] {__method_name} scene cleaned before/after call")
            return result

        setattr(_wrapped, "_infer_v15_scene_sanitize_wrapped", True)
        setattr(respace, method_name, _wrapped)




# ------------------------------
# planning RAG helpers
# ------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Failed to parse JSONL line {line_no} in {path}: {e}") from e
    return rows


# ============================================================
# small utils
# ============================================================

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def norm_text(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    return str(x).strip().lower()


def coarse_anchor(anchor: Optional[str]) -> Optional[str]:
    if anchor is None:
        return None
    a = norm_text(anchor)
    if a is None:
        return None

    if "bed" in a:
        return "bed"
    if "sofa" in a or "couch" in a:
        return "sofa"
    if "dining table" in a:
        return "dining table"
    if "coffee table" in a:
        return "coffee table"
    if "side table" in a or "end table" in a:
        return "side table"
    if "nightstand" in a or "bedside" in a:
        return "nightstand"
    if "desk" in a:
        return "desk"
    if "tv stand" in a or "media console" in a:
        return "tv stand"
    if "bookshelf" in a or "bookcase" in a:
        return "bookshelf"
    if "cabinet" in a or "sideboard" in a:
        return "cabinet"
    if "wardrobe" in a:
        return "wardrobe"
    if "dresser" in a or "drawer chest" in a:
        return "dresser"
    if "washing machine" in a or "washer" in a:
        return "washing machine"
    return a


def doc_anchor_matches(doc_anchor: Optional[str], query_anchor: Optional[str], allow_coarse: bool = True) -> bool:
    if query_anchor is None:
        return True
    da = norm_text(doc_anchor)
    qa = norm_text(query_anchor)
    if da == qa:
        return True
    if allow_coarse and coarse_anchor(da) == coarse_anchor(qa):
        return True
    return False


def build_query_text(user_prompt: str, room_type: Optional[str], anchor: Optional[str]) -> str:
    parts = []
    if room_type:
        parts.append(room_type)
    if anchor:
        parts.append(anchor)
    parts.append(user_prompt)
    return " ".join(parts).strip()


# ============================================================
# dataclasses
# ============================================================

@dataclass
class RetrievalDoc:
    doc_id: str
    title: str
    scope: Optional[str]
    room_type: Optional[str]
    anchor: Optional[str]
    text: str
    score: float
    top_members: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    source_fine_anchors: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetrievalDoc":
        return cls(
            doc_id=str(d.get("doc_id", "")),
            title=str(d.get("title", "")),
            scope=d.get("scope"),
            room_type=d.get("room_type"),
            anchor=d.get("anchor"),
            text=str(d.get("text", "")),
            score=float(d.get("score", 0.0)),
            top_members=d.get("top_members"),
            keywords=d.get("keywords"),
            source_fine_anchors=d.get("source_fine_anchors"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlanningRAGConfig:
    top_k_per_anchor: int = 2
    top_k_room_level: int = 3
    max_docs_per_anchor_in_prompt: int = 2
    allow_coarse_anchor_match: bool = True
    batch_size: int = 16
    device: Optional[str] = None


# ============================================================
# embedding backend
# ============================================================

class EmbeddingBackend:
    def __init__(self, model_name_or_path: str, device: Optional[str] = None, batch_size: int = 16):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self.backend_type = None
        self.model = None
        self.tokenizer = None

        # sentence-transformers first
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self.model = SentenceTransformer(model_name_or_path, device=device)
            self.backend_type = "sentence_transformers"
            return
        except Exception:
            pass

        # transformers fallback
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )

            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            self.model.to(self.device)
            self.model.eval()
            self.backend_type = "transformers"
            return
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model from {model_name_or_path}") from e

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        if self.backend_type == "sentence_transformers":
            embs = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embs.astype(np.float32)

        if self.backend_type == "transformers":
            return self._encode_transformers(texts)

        raise RuntimeError("Embedding backend is not initialized.")

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])

    def _encode_transformers(self, texts: List[str]) -> np.ndarray:
        import torch

        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(**batch)

            if not hasattr(outputs, "last_hidden_state"):
                raise RuntimeError("Transformers model output does not contain last_hidden_state.")

            token_embeddings = outputs.last_hidden_state.float()
            attention_mask = batch["attention_mask"].unsqueeze(-1).float()

            masked = token_embeddings * attention_mask
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1.0)
            emb = summed / counts

            emb = emb.detach().cpu().numpy().astype(np.float32)
            emb = l2_normalize(emb)
            all_embeddings.append(emb)

        return np.concatenate(all_embeddings, axis=0)


# ============================================================
# candidate anchor inference
# ============================================================

ANCHOR_KEYWORDS = [
    "King-Size Bed",
    "Queen-Size Bed",
    "Double Bed",
    "Single Bed",
    "Kids Bed",
    "Bed",
    "Loveseat Sofa",
    "Sectional Sofa",
    "L-Shaped Sofa",
    "Sofa",
    "Dining Table",
    "Coffee Table",
    "Side Table",
    "Nightstand",
    "Desk",
    "TV Stand",
    "Bookshelf",
    "Cabinet",
    "Wardrobe",
    "Dresser",
    "Washing Machine",
]


def _extract_scene_text(obj: Dict[str, Any]) -> str:
    parts = []
    for k in ("category", "super_category", "desc", "sampled_asset_desc"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    p = _safe_object_prompt_text(obj)
    if p:
        parts.append(p)
    return " | ".join(parts).lower()


def infer_anchor_candidates(
    scene: Optional[Dict[str, Any]] = None,
    role_categories: Optional[Sequence[str]] = None,
    user_prompt: str = "",
    max_anchors: int = 4,
) -> List[str]:
    """
    Heuristic anchor proposal.
    Priority:
    1) role_categories if provided
    2) scene object text
    3) user prompt text
    """
    votes: List[str] = []

    if role_categories:
        for cat in role_categories:
            if not isinstance(cat, str):
                continue
            votes.append(cat)

    if scene is not None:
        for obj in scene.get("objects", []):
            text = _extract_scene_text(obj)
            for kw in ANCHOR_KEYWORDS:
                if kw.lower() in text:
                    votes.append(kw)

    prompt_low = user_prompt.lower()
    for kw in ANCHOR_KEYWORDS:
        if kw.lower() in prompt_low:
            votes.append(kw)

    # dedupe but preserve a soft preference for more specific anchors first
    # sort by specificity (longer first), then frequency
    counter = {}
    for v in votes:
        counter[v] = counter.get(v, 0) + 1

    ranked = sorted(counter.items(), key=lambda kv: (-len(kv[0]), -kv[1], kv[0]))
    return [k for k, _ in ranked[:max_anchors]]


# ============================================================
# retriever
# ============================================================

class PlanningRAGRetriever:
    """
    Reusable retriever object.
    Load once, call many times.
    """

    def __init__(
        self,
        index_dir: str,
        model_name_or_path: str,
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        self.index_dir = Path(index_dir)
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size

        index_path = self.index_dir / "index.faiss"
        metadata_path = self.index_dir / "metadata.jsonl"
        config_path = self.index_dir / "config.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file: {config_path}")

        self.index = faiss.read_index(str(index_path))
        self.metadata = load_jsonl(metadata_path)
        self.config = load_json(config_path)
        self.embedder = EmbeddingBackend(
            model_name_or_path=model_name_or_path,
            device=device,
            batch_size=batch_size,
        )

    def _search(
        self,
        query_text: str,
        top_k: int,
        room_type: Optional[str] = None,
        anchor: Optional[str] = None,
        scope: Optional[str] = "room",
        allow_coarse_anchor_match: bool = True,
    ) -> List[RetrievalDoc]:
        query_vec = l2_normalize(self.embedder.encode_query(query_text).astype(np.float32))
        room_type_n = norm_text(room_type)
        scope_n = norm_text(scope)

        candidate_ids = []
        for i, doc in enumerate(self.metadata):
            doc_room = norm_text(doc.get("room_type"))
            doc_scope = norm_text(doc.get("scope"))
            doc_anchor = doc.get("anchor")

            if room_type_n is not None and doc_room != room_type_n:
                continue
            if scope_n is not None and doc_scope != scope_n:
                continue
            if not doc_anchor_matches(doc_anchor, anchor, allow_coarse=allow_coarse_anchor_match):
                continue
            candidate_ids.append(i)

        if not candidate_ids:
            return []

        vecs = np.asarray([self.index.reconstruct(int(i)) for i in candidate_ids], dtype=np.float32)
        scores = np.matmul(vecs, query_vec[0])
        order = np.argsort(-scores)

        results: List[RetrievalDoc] = []
        used_doc_ids = set()
        for rank in order:
            idx = candidate_ids[int(rank)]
            doc = dict(self.metadata[idx])
            doc["score"] = float(scores[int(rank)])
            doc_id = doc.get("doc_id", idx)
            if doc_id in used_doc_ids:
                continue
            used_doc_ids.add(doc_id)
            results.append(RetrievalDoc.from_dict(doc))
            if len(results) >= top_k:
                break

        return results

    def retrieve(
        self,
        room_type: Optional[str],
        user_prompt: str,
        anchors: Optional[List[str]] = None,
        top_k_per_anchor: int = 2,
        top_k_room_level: int = 3,
        allow_coarse_anchor_match: bool = True,
    ) -> Dict[str, Any]:
        """
        Returns a structured retrieval bundle.
        """
        results_by_anchor: Dict[str, List[Dict[str, Any]]] = {}

        # room-level retrieval
        room_query = build_query_text(user_prompt=user_prompt, room_type=room_type, anchor=None)
        room_docs = self._search(
            query_text=room_query,
            top_k=top_k_room_level,
            room_type=room_type,
            anchor=None,
            scope="room",
            allow_coarse_anchor_match=allow_coarse_anchor_match,
        )
        if room_docs:
            results_by_anchor["__room_level__"] = [d.to_dict() for d in room_docs]

        # anchor-level retrieval
        anchors = anchors or []
        for anchor in anchors:
            q = build_query_text(user_prompt=user_prompt, room_type=room_type, anchor=anchor)
            docs = self._search(
                query_text=q,
                top_k=top_k_per_anchor,
                room_type=room_type,
                anchor=anchor,
                scope="room",
                allow_coarse_anchor_match=allow_coarse_anchor_match,
            )
            if docs:
                results_by_anchor[anchor] = [d.to_dict() for d in docs]

        return {
            "room_type": room_type,
            "user_prompt": user_prompt,
            "anchors": anchors,
            "results_by_anchor": results_by_anchor,
        }

    def summarize_for_prompt(
        self,
        retrieval_bundle: Dict[str, Any],
        max_docs_per_anchor: int = 1,
        max_chars_per_doc: int = 180,
    ) -> str:
        room_type = retrieval_bundle.get("room_type")
        user_prompt = retrieval_bundle.get("user_prompt")
        results_by_anchor = retrieval_bundle.get("results_by_anchor", {})

        def _short(x: str) -> str:
            x = str(x or "").strip().replace("\n", " ")
            return x[:max_chars_per_doc] + ("..." if len(x) > max_chars_per_doc else "")

        lines: List[str] = []
        lines.append("RETRIEVED GROUP PRIORS FOR PLANNING:")
        if room_type:
            lines.append(f"- room_type: {room_type}")
        lines.append(f"- original prompt: {_short(user_prompt)}")

        room_docs = results_by_anchor.get("__room_level__", [])
        if room_docs:
            lines.append("Room-level priors:")
            for doc in room_docs[:max_docs_per_anchor]:
                lines.append(f"- {doc.get('title', '')}: {_short(doc.get('text', ''))}")

        for anchor, docs in results_by_anchor.items():
            if anchor == "__room_level__" or not docs:
                continue
            for doc in docs[:max_docs_per_anchor]:
                lines.append(f"- {anchor}: {_short(doc.get('text', ''))}")

        lines.append("Use priors as soft hints only.")
        return "\n".join(lines)

    def build_augmented_prompt(
        self,
        base_prompt: str,
        retrieval_bundle: Dict[str, Any],
        max_docs_per_anchor: int = 2,
    ) -> str:
        prior_hint = self.summarize_for_prompt(retrieval_bundle, max_docs_per_anchor=max_docs_per_anchor)
        return f"""{prior_hint}

Now perform scene planning for the following request:
{base_prompt}
"""


# ============================================================
# high-level helpers
# ============================================================

def build_planning_rag_hint(
    retriever: PlanningRAGRetriever,
    *,
    user_prompt: str,
    room_type: Optional[str],
    scene: Optional[Dict[str, Any]] = None,
    role_categories: Optional[Sequence[str]] = None,
    anchors: Optional[List[str]] = None,
    top_k_per_anchor: int = 2,
    top_k_room_level: int = 3,
    max_docs_per_anchor_in_prompt: int = 2,
    allow_coarse_anchor_match: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    One-shot helper:
    - infer anchors if not provided
    - retrieve docs
    - summarize into a prompt-ready hint
    """
    if anchors is None or len(anchors) == 0:
        anchors = infer_anchor_candidates(
            scene=scene,
            role_categories=role_categories,
            user_prompt=user_prompt,
            max_anchors=4,
        )

    bundle = retriever.retrieve(
        room_type=room_type,
        user_prompt=user_prompt,
        anchors=anchors,
        top_k_per_anchor=top_k_per_anchor,
        top_k_room_level=top_k_room_level,
        allow_coarse_anchor_match=allow_coarse_anchor_match,
    )
    hint = retriever.summarize_for_prompt(bundle, max_docs_per_anchor=max_docs_per_anchor_in_prompt)
    return hint, bundle


def build_augmented_extra_hints(
    retriever: PlanningRAGRetriever,
    *,
    base_extra_hints_text: str,
    user_prompt: str,
    room_type: Optional[str],
    scene: Optional[Dict[str, Any]] = None,
    role_categories: Optional[Sequence[str]] = None,
    anchors: Optional[List[str]] = None,
    top_k_per_anchor: int = 2,
    top_k_room_level: int = 3,
    max_docs_per_anchor_in_prompt: int = 2,
    allow_coarse_anchor_match: bool = True,
    separator: str = "\n\n",
) -> str:
    """
    Appends RAG hint to your existing extra_hints_text.
    This is the easiest way to integrate with the user's current architecture.
    """
    hint, _ = build_planning_rag_hint(
        retriever=retriever,
        user_prompt=user_prompt,
        room_type=room_type,
        scene=scene,
        role_categories=role_categories,
        anchors=anchors,
        top_k_per_anchor=top_k_per_anchor,
        top_k_room_level=top_k_room_level,
        max_docs_per_anchor_in_prompt=max_docs_per_anchor_in_prompt,
        allow_coarse_anchor_match=allow_coarse_anchor_match,
    )

    base = (base_extra_hints_text or "").strip()
    if not base:
        return hint
    return f"{base}{separator}{hint}"


def dump_retrieval_bundle(path: str | Path, retrieval_bundle: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(retrieval_bundle, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# optional self-test
# ============================================================


_PLANNING_RAG_RETRIEVER_CACHE: Dict[Tuple[str, str, Optional[str], int], PlanningRAGRetriever] = {}


def _get_planning_rag_retriever(index_dir: str, model_name_or_path: str, device: Optional[str], batch_size: int) -> PlanningRAGRetriever:
    key = (str(index_dir), str(model_name_or_path), device, int(batch_size))
    retriever = _PLANNING_RAG_RETRIEVER_CACHE.get(key)
    if retriever is None:
        if faiss is None:
            raise RuntimeError("faiss is not available but ENABLE_PLANNING_RAG=1.")
        retriever = PlanningRAGRetriever(
            index_dir=index_dir,
            model_name_or_path=model_name_or_path,
            device=device,
            batch_size=batch_size,
        )
        _PLANNING_RAG_RETRIEVER_CACHE[key] = retriever
    return retriever


def _infer_room_type_from_scene_for_rag(scene: Dict[str, Any]) -> Optional[str]:
    room_id = scene.get("room_id", None)
    if isinstance(room_id, str) and room_id.strip():
        token = room_id.split("-", 1)[0].strip().lower().replace("_", "").replace(" ", "")
        if "livingdiningroom" in token:
            return "livingdiningroom"
        if "bedroom" in token:
            return "bedroom"
        if "livingroom" in token:
            return "livingroom"
        if "diningroom" in token:
            return "diningroom"
        if "library" in token or "study" in token:
            return "library"
        if "laundry" in token:
            return "laundry"
    room_type = scene.get("room_type", None)
    if isinstance(room_type, str) and room_type.strip():
        token = room_type.strip().lower().replace("_", "").replace(" ", "")
        if "livingdiningroom" in token:
            return "livingdiningroom"
        if "bedroom" in token:
            return "bedroom"
        if "livingroom" in token:
            return "livingroom"
        if "diningroom" in token:
            return "diningroom"
        if "library" in token or "study" in token:
            return "library"
        if "laundry" in token:
            return "laundry"
    text_chunks = []
    for obj in scene.get("objects", []):
        for k in ("category", "super_category", "desc", "sampled_asset_desc"):
            v = obj.get(k, None)
            if isinstance(v, str) and v.strip():
                text_chunks.append(v.lower())
        p = _safe_object_prompt_text(obj)
        if p:
            text_chunks.append(p.lower())
    full_text = " | ".join(text_chunks)
    if "washing machine" in full_text or "washer" in full_text:
        return "laundry"
    return None


def _maybe_apply_planning_rag(
    *,
    scene: Dict[str, Any],
    room_prompt: str,
    extra_hints_text: str,
    out_root: Path,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    enable_planning_rag = _env_bool(os.getenv("ENABLE_PLANNING_RAG", "0"))
    if not enable_planning_rag:
        return room_prompt, extra_hints_text, None

    index_dir = os.getenv("PLANNING_RAG_INDEX_DIR", "").strip()
    model_name_or_path = os.getenv("PLANNING_RAG_MODEL", "").strip()
    require_index = _env_bool(os.getenv("PLANNING_RAG_REQUIRE_INDEX", "0"))
    if not index_dir or not model_name_or_path:
        if require_index:
            raise RuntimeError("ENABLE_PLANNING_RAG=1 but PLANNING_RAG_INDEX_DIR / PLANNING_RAG_MODEL is missing.")
        _log("[planning RAG] skipped: missing PLANNING_RAG_INDEX_DIR or PLANNING_RAG_MODEL")
        return room_prompt, extra_hints_text, None

    try:
        room_type = _infer_room_type_from_scene_for_rag(scene)
        role_graph = infer_role_graph(scene)
        anchors_override = [x.strip() for x in os.getenv("PLANNING_RAG_ANCHORS", "").split(",") if x.strip()]
        max_anchors = int(os.getenv("PLANNING_RAG_MAX_ANCHORS", "4"))
        top_k_per_anchor = int(os.getenv("PLANNING_RAG_TOP_K_PER_ANCHOR", "2"))
        top_k_room_level = int(os.getenv("PLANNING_RAG_TOP_K_ROOM_LEVEL", "3"))
        max_docs_per_anchor = int(os.getenv("PLANNING_RAG_MAX_DOCS_PER_ANCHOR", "2"))
        batch_size = int(os.getenv("PLANNING_RAG_BATCH_SIZE", "16"))
        device = os.getenv("PLANNING_RAG_DEVICE", "").strip() or None

        retriever = _get_planning_rag_retriever(index_dir=index_dir, model_name_or_path=model_name_or_path, device=device, batch_size=batch_size)
        anchors = anchors_override if anchors_override else infer_anchor_candidates(
            scene=scene,
            role_categories=role_graph.categories,
            user_prompt=room_prompt,
            max_anchors=max_anchors,
        )
        bundle = retriever.retrieve(
            room_type=room_type,
            user_prompt=room_prompt,
            anchors=anchors,
            top_k_per_anchor=top_k_per_anchor,
            top_k_room_level=top_k_room_level,
            allow_coarse_anchor_match=True,
        )
        results_by_anchor = bundle.get("results_by_anchor", {})
        if not results_by_anchor:
            _log("[planning RAG] no retrieval results, using original prompt")
            return room_prompt, extra_hints_text, bundle

        hint_text = retriever.summarize_for_prompt(
            bundle,
            max_docs_per_anchor=max_docs_per_anchor,
        ).strip()

        planning_prompt = f"{hint_text}\n\nNow perform scene planning for the following request:\n{room_prompt}".strip()

        # 不再重复把完整 RAG hint 塞到 extra_hints_text
        extra_hints_text = extra_hints_text.strip()

        if _env_bool(os.getenv("PLANNING_RAG_WRITE_DEBUG", "1")):
            _write_text(out_root / "planning_rag_hint.txt", hint_text)
            _write_json(out_root / "planning_rag_bundle.json", bundle)
            _write_json(out_root / "planning_rag_meta.json", {
                "room_type": room_type,
                "anchors": anchors,
                "top_k_per_anchor": top_k_per_anchor,
                "top_k_room_level": top_k_room_level,
                "max_docs_per_anchor": max_docs_per_anchor,
                "index_dir": index_dir,
                "model_name_or_path": model_name_or_path,
            })
            _write_text(out_root / "planning_prompt_with_rag.txt", planning_prompt)

        _log(f"[planning RAG] enabled room_type={room_type} anchors={anchors} room_docs={len(results_by_anchor.get('__room_level__', []))}")
        return planning_prompt, extra_hints_text, bundle
    except Exception:
        if _env_bool(os.getenv("PLANNING_RAG_WRITE_DEBUG", "1")):
            _write_text(out_root / "planning_rag.error.txt", traceback.format_exc())
        if require_index:
            raise
        _log("[planning RAG] failed, fallback to original prompt")
        return room_prompt, extra_hints_text, None


def _summarize_scene_without_vlm_optimization(*, scene: Dict[str, Any], out_root: Path, respace: ReSpace, cfg: Config, zero_shot_relation_plan: Optional[Dict[str, Any]] = None, planning_rag_bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    scene = _sanitize_scene_object_prompts(scene, keep_raw=True)
    out_root.mkdir(parents=True, exist_ok=True)
    timing = TimingStats()
    if cfg.render_final:
        final_dir = out_root / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        t0 = _now()
        respace.render_scene_frame(scene, filename="final", pth_viz_output=final_dir)
        render_annotated_top_view(scene, "final", final_dir, resolution=(1024, 1024), show_assets=True, font_size=14)
        timing.render_sec += _now() - t0
        _write_json(final_dir / "scene.json", scene)

    role_graph = infer_role_graph(scene)
    priors = build_role_based_relation_priors(scene, role_graph)
    final_score, final_metrics, final_rel, final_func = _score_scene_full(scene, role_graph, priors, timing, cfg)
    final_judge_score, final_judge_metrics, final_judge_rel, final_judge_func = _score_scene_stable_judge(scene, role_graph, role_graph, priors, timing, cfg)
    summary = {
        "mode": "no_vlm_optimization",
        "planning_rag_enabled": _env_bool(os.getenv("ENABLE_PLANNING_RAG", "0")),
        "vlm_optimization_enabled": False,
        "num_relation_priors": len(priors),
        "total_runtime_sec": 0.0,
        "render_sec": round(timing.render_sec, 4),
        "vlm_sec": 0.0,
        "optimize_sec": 0.0,
        "eval_sec": round(timing.eval_sec, 4),
        "step_runtime_records": [],
        "history_records": [],
        "initial_metrics": final_metrics,
        "initial_rel_loss": round(final_rel, 4),
        "initial_func_loss": round(final_func, 4),
        "initial_structure_stats": final_metrics.get("structure_stats") if isinstance(final_metrics, dict) else None,
        "initial_score": round(final_score, 6),
        "initial_judge_score": round(final_judge_score, 6),
        "final_metrics": final_metrics,
        "final_judge_metrics": final_judge_metrics,
        "final_rel_loss": round(final_rel, 4),
        "final_func_loss": round(final_func, 4),
        "final_judge_rel_loss": round(final_judge_rel, 4),
        "final_judge_func_loss": round(final_judge_func, 4),
        "final_structure_stats": final_metrics.get("structure_stats") if isinstance(final_metrics, dict) else None,
        "final_score": round(final_score, 6),
        "final_judge_score": round(final_judge_score, 6),
        "zero_shot_relation_plan_used": bool(zero_shot_relation_plan),
        "planning_rag_bundle_present": planning_rag_bundle is not None,
    }
    _write_json(out_root / "summary.json", summary)
    return summary
# ------------------------------
# geometry helpers
# ------------------------------

def _normalize_angle(deg: float) -> float:
    return deg % 360.0


def _angle_diff(a: float, b: float) -> float:
    d = abs(_normalize_angle(a) - _normalize_angle(b))
    return min(d, 360.0 - d)


def _axis_angle_diff(a: float, b: float) -> float:
    return min(_angle_diff(a, b), _angle_diff(a, _normalize_angle(b + 180.0)))


def _forward_vec_from_yaw(yaw_deg: float) -> Tuple[float, float]:
    rad = math.radians(yaw_deg)
    return math.sin(rad), math.cos(rad)


def _signed_proj(ax: float, az: float, bx: float, bz: float, vx: float, vz: float) -> float:
    return (bx - ax) * vx + (bz - az) * vz


def _clone_scene_with_updated_object(scene: Dict[str, Any], idx: int, new_obj: Dict[str, Any]) -> Dict[str, Any]:
    sc = dict(scene)
    objs = list(scene.get("objects", []))
    objs[idx] = new_obj
    sc["objects"] = objs
    return sc


def _get_floor_polygon(scene: Dict[str, Any]) -> Optional[ShapelyPolygon]:
    bb = scene.get("bounds_bottom")
    if not isinstance(bb, list) or len(bb) < 3:
        return None
    try:
        return create_floor_plan_polygon(bb)
    except Exception:
        return None


def _room_extents_xz(scene: Dict[str, Any]) -> Tuple[float, float, float, float]:
    pts = [(float(p[0]), float(p[2])) for p in scene.get("bounds_bottom", []) if isinstance(p, list) and len(p) >= 3]
    if not pts:
        return -1.0, 1.0, -1.0, 1.0
    xs, zs = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), max(xs), min(zs), max(zs)


def _find_nearest_wall_yaw(scene: Dict[str, Any], pos: Sequence[float]) -> Optional[float]:
    pts = [(float(p[0]), float(p[2])) for p in scene.get("bounds_bottom", []) if isinstance(p, list) and len(p) >= 3]
    if len(pts) < 3:
        return None
    cx, cz = room_center_xz(scene)
    best_d, best_n = float("inf"), None
    for i, (ax, az) in enumerate(pts):
        bx, bz = pts[(i + 1) % len(pts)]
        abx, abz = bx - ax, bz - az
        apx, apz = pos[0] - ax, pos[2] - az
        den = abx * abx + abz * abz
        if den < 1e-12:
            continue
        t = max(0.0, min(1.0, (apx * abx + apz * abz) / den))
        px, pz = ax + t * abx, az + t * abz
        d = math.hypot(pos[0] - px, pos[2] - pz)
        if d >= best_d:
            continue
        nx, nz = -abz, abx
        nl = math.hypot(nx, nz)
        if nl < 1e-9:
            continue
        nx, nz = nx / nl, nz / nl
        if nx * (cx - px) + nz * (cz - pz) < 0:
            nx, nz = -nx, -nz
        best_d, best_n = d, (nx, nz)
    if best_n is None:
        return None
    return _normalize_angle(math.degrees(math.atan2(best_n[0], best_n[1])))


def _nearest_parallel_wall_yaw(scene: Dict[str, Any], pos: Sequence[float], current_yaw: Optional[float] = None) -> Optional[float]:
    wy = _find_nearest_wall_yaw(scene, pos)
    if wy is None:
        return None
    opts = [_normalize_angle(wy + 90.0), _normalize_angle(wy + 270.0)]
    return opts[0] if current_yaw is None else min(opts, key=lambda y: _angle_diff(y, current_yaw))


def _nearest_normal_axis_yaw(scene: Dict[str, Any], pos: Sequence[float], current_yaw: Optional[float] = None) -> Optional[float]:
    wy = _find_nearest_wall_yaw(scene, pos)
    if wy is None:
        return None
    opts = [_normalize_angle(wy), _normalize_angle(wy + 180.0)]
    return opts[0] if current_yaw is None else min(opts, key=lambda y: _angle_diff(y, current_yaw))


def _object_xz_polygon(obj: Dict[str, Any]) -> ShapelyPolygon:
    try:
        poly, _, _, _ = get_xz_bbox_from_obj(obj)
        return poly
    except Exception:
        pos = obj.get("pos", [0.0, 0.0, 0.0])
        return ShapelyPoint(float(pos[0]), float(pos[2])).buffer(0.05)


def _normalize_hint_pos(hint_pos: Any):
    if hint_pos is None:
        return None
    if isinstance(hint_pos, dict) and "x" in hint_pos and "z" in hint_pos:
        return [float(hint_pos["x"]), float(hint_pos.get("y", 0.0)), float(hint_pos["z"])]
    if isinstance(hint_pos, (list, tuple)):
        if len(hint_pos) == 2:
            return [float(hint_pos[0]), 0.0, float(hint_pos[1])]
        if len(hint_pos) >= 3:
            return [float(hint_pos[0]), float(hint_pos[1]), float(hint_pos[2])]
    return None


# ------------------------------
# local repair
# ------------------------------

def _compute_oob_push_direction(scene: Dict[str, Any], obj: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    floor = _get_floor_polygon(scene)
    if floor is None:
        return None
    oob = compute_oob(obj, floor, scene.get("bounds_bottom", []), scene.get("bounds_top", []), is_debug=False)
    if oob <= 1e-8:
        return None
    pos = obj.get("pos", [0.0, 0.0, 0.0])
    p = ShapelyPoint(pos[0], pos[2])
    if floor.contains(p):
        c = floor.centroid
        dx, dz = c.x - pos[0], c.y - pos[2]
    else:
        n = floor.exterior.interpolate(floor.exterior.project(p))
        dx, dz = n.x - pos[0], n.y - pos[2]
    nrm = math.hypot(dx, dz)
    return None if nrm < 1e-9 else (dx / nrm, dz / nrm, oob)


def _collision_neighbors(scene: Dict[str, Any], idx: int) -> List[Tuple[int, float, float, float]]:
    objs = scene.get("objects", [])
    if not (0 <= idx < len(objs)):
        return []
    try:
        bbox_a, _, ya0, ya1 = get_xz_bbox_from_obj(objs[idx])
    except Exception:
        return []
    pa = objs[idx].get("pos", [0.0, 0.0, 0.0])
    out = []
    for j, other in enumerate(objs):
        if j == idx:
            continue
        try:
            bbox_b, _, yb0, yb1 = get_xz_bbox_from_obj(other)
        except Exception:
            continue
        if max(0.0, min(ya1, yb1) - max(ya0, yb0)) <= 0:
            continue
        inter = bbox_a.intersection(bbox_b)
        if inter.is_empty or inter.area < 1e-8:
            continue
        pb = other.get("pos", [0.0, 0.0, 0.0])
        dx, dz = pa[0] - pb[0], pa[2] - pb[2]
        n = math.hypot(dx, dz) or 1.0
        out.append((j, dx / n, dz / n, float(inter.area)))
    return out


def _project_inside_room(scene: Dict[str, Any], idx: int) -> None:
    res = _compute_oob_push_direction(scene, scene.get("objects", [])[idx])
    if res is None:
        return
    dx, dz, oob = res
    step = min(0.35, max(0.04, math.sqrt(oob + 1e-8)))
    obj = scene["objects"][idx]
    pos = list(obj.get("pos", [0.0, 0.0, 0.0]))
    obj["pos"] = [pos[0] + dx * step, pos[1], pos[2] + dz * step]


def _separate_local_collisions(scene: Dict[str, Any], idx: int) -> None:
    cols = _collision_neighbors(scene, idx)
    if not cols:
        return
    obj = scene["objects"][idx]
    dx = dz = 0.0
    for _, ux, uz, area in cols:
        mag = min(0.28, max(0.02, math.sqrt(area + 1e-8)))
        dx += ux * mag
        dz += uz * mag
    pos = list(obj.get("pos", [0.0, 0.0, 0.0]))
    obj["pos"] = [pos[0] + dx, pos[1], pos[2] + dz]
    _project_inside_room(scene, idx)


def _repair_object_local(scene: Dict[str, Any], idx: int, passes: int) -> None:
    if not (0 <= idx < len(scene.get("objects", []))):
        return
    for _ in range(max(1, passes)):
        _project_inside_room(scene, idx)
        _separate_local_collisions(scene, idx)


# ------------------------------
# relation / structure / scoring
# ------------------------------

def _pair_target_dist(a: Dict[str, Any], b: Dict[str, Any], alpha: float = 0.35, bias: float = 0.15) -> float:
    return alpha * (obj_diag_size_xz(a) + obj_diag_size_xz(b)) + bias


def _loss_distance_band(d: float, lo: float, hi: float, w: float) -> float:
    return (lo - d) * w if d < lo else (d - hi) * w if d > hi else 0.0


def _loss_facing(src: Dict[str, Any], tgt: Dict[str, Any], w: float) -> float:
    s, t = src.get("pos", [0.0, 0.0, 0.0]), tgt.get("pos", [0.0, 0.0, 0.0])
    fx, fz = _forward_vec_from_yaw(yaw_from_quaternion(src.get("rot", [0.0, 0.0, 0.0, 1.0])))
    tx, tz = t[0] - s[0], t[2] - s[2]
    n = math.hypot(tx, tz)
    if n < 1e-9:
        return 0.0
    tx, tz = tx / n, tz / n
    return max(0.0, 1.0 - (fx * tx + fz * tz)) * w


def _loss_centered_lateral(src: Dict[str, Any], anchor: Dict[str, Any], w: float) -> float:
    ap, sp = anchor.get("pos", [0.0, 0.0, 0.0]), src.get("pos", [0.0, 0.0, 0.0])
    fx, fz = _forward_vec_from_yaw(yaw_from_quaternion(anchor.get("rot", [0.0, 0.0, 0.0, 1.0])))
    return abs(_signed_proj(ap[0], ap[2], sp[0], sp[2], fz, -fx)) * w


def _loss_in_front_of(src: Dict[str, Any], anchor: Dict[str, Any], w: float) -> float:
    ap, sp = anchor.get("pos", [0.0, 0.0, 0.0]), src.get("pos", [0.0, 0.0, 0.0])
    fx, fz = _forward_vec_from_yaw(yaw_from_quaternion(anchor.get("rot", [0.0, 0.0, 0.0, 1.0])))
    proj = _signed_proj(ap[0], ap[2], sp[0], sp[2], fx, fz)
    return 0.0 if proj >= 0 else abs(proj) * w


def _loss_side_of(src: Dict[str, Any], anchor: Dict[str, Any], w: float) -> float:
    ap, sp = anchor.get("pos", [0.0, 0.0, 0.0]), src.get("pos", [0.0, 0.0, 0.0])
    fx, fz = _forward_vec_from_yaw(yaw_from_quaternion(anchor.get("rot", [0.0, 0.0, 0.0, 1.0])))
    lx, lz = fz, -fx
    fwd = abs(_signed_proj(ap[0], ap[2], sp[0], sp[2], fx, fz))
    lat = abs(_signed_proj(ap[0], ap[2], sp[0], sp[2], lx, lz))
    return (max(0.0, 0.2 - lat) + max(0.0, fwd - 0.7)) * w


def _loss_against_wall(scene: Dict[str, Any], obj: Dict[str, Any], w: float, category: Optional[str] = None) -> float:
    pos = obj.get("pos", [0.0, 0.0, 0.0])
    wy = _find_nearest_wall_yaw(scene, pos)
    if wy is None:
        return 0.0
    dist = max(0.0, distance_to_nearest_wall_xz(scene, pos) - (0.35 if category == "bed" else 0.3))
    yaw = yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
    if category == "bed":
        yd = _axis_angle_diff(yaw, wy) / 180.0
    elif category in {"table", "desk", "counter", "vanity", "cabinet", "console table", "coffee table"}:
        py = _nearest_parallel_wall_yaw(scene, pos, yaw)
        yd = 0.0 if py is None else _angle_diff(yaw, py) / 180.0
    else:
        yd = _angle_diff(yaw, _normalize_angle(wy + 180.0)) / 180.0
    return (dist + 0.5 * yd) * w


def _loss_parallel(scene: Dict[str, Any], obj: Dict[str, Any], w: float) -> float:
    pos = obj.get("pos", [0.0, 0.0, 0.0])
    yaw = yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
    py = _nearest_parallel_wall_yaw(scene, pos, yaw)
    return 0.0 if py is None else (_axis_angle_diff(yaw, py) / 180.0) * w


def _compute_relation_loss(scene: Dict[str, Any], role_graph: RoleGraph, priors: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]]]:
    objs, total, violations = scene.get("objects", []), 0.0, []
    for item in priors:
        t = str(item.get("type", "")).strip()
        if t not in REL_TYPES:
            continue
        si, ti = item.get("src_idx"), item.get("tgt_idx")
        if not isinstance(si, int) or not (0 <= si < len(objs)):
            continue
        w = float(item.get("weight", 1.0)) * max(0.0, min(1.0, float(item.get("confidence", 1.0))))
        src, loss, dist = objs[si], 0.0, None
        if t == "against_wall":
            loss = _loss_against_wall(scene, src, w, role_graph.categories[si])
        elif t == "parallel":
            loss = _loss_parallel(scene, src, w)
        else:
            if not isinstance(ti, int) or not (0 <= ti < len(objs)) or ti == si:
                continue
            tgt, dist = objs[ti], xz_dist(src, objs[ti])
            if t == "near":
                loss = max(0.0, dist - (_pair_target_dist(src, tgt) + 0.25)) * w
            elif t == "distance_band":
                lo = float(item.get("lo", max(0.3, _pair_target_dist(src, tgt, 0.5, 0.4) - 0.5)))
                hi = float(item.get("hi", _pair_target_dist(src, tgt, 0.5, 0.4) + 0.8))
                loss = _loss_distance_band(dist, lo, hi, w)
            elif t == "facing":
                loss = _loss_facing(src, tgt, w)
            elif t == "facing_pair":
                loss = 0.5 * _loss_facing(src, tgt, w) + 0.5 * _loss_facing(tgt, src, w)
            elif t == "centered_with":
                loss = _loss_centered_lateral(src, tgt, w)
            elif t == "in_front_of":
                loss = _loss_in_front_of(src, tgt, w)
            elif t == "side_of":
                loss = _loss_side_of(src, tgt, w)
        if loss > 1e-8:
            total += loss
            violations.append({"src_idx": si, "tgt_idx": ti, "type": t, "dist": None if dist is None else round(dist, 3), "penalty": round(loss, 4)})
    return total, violations


def _collect_zone_groups(role_graph: RoleGraph, n_obj: int) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i in range(n_obj):
        label = role_graph.zone_by_idx.get(i) or role_graph.function_by_idx.get(i) or role_graph.role_by_idx.get(i) or f"zone_{i}"
        groups.setdefault(str(label), []).append(i)
    return groups


def _compute_structure_stats(scene: Dict[str, Any], role_graph: RoleGraph, cfg: Optional[Config] = None) -> Dict[str, float]:
    objs = scene.get("objects", [])
    floor = _get_floor_polygon(scene)
    minx, maxx, minz, maxz = _room_extents_xz(scene)
    room_area = float(floor.area) if floor is not None else max(1e-6, (maxx - minx) * (maxz - minz))
    room_w, room_h = max(1e-6, maxx - minx), max(1e-6, maxz - minz)
    polys = [_object_xz_polygon(o) for o in objs]
    union_poly = unary_union(polys) if polys else None
    occupied = 0.0 if union_poly is None else min(room_area, float(union_poly.area))
    open_ratio = max(0.0, 1.0 - occupied / max(room_area, 1e-6))
    groups = _collect_zone_groups(role_graph, len(objs))
    zone_count = len([g for g in groups.values() if g])
    max_zone_ratio = 1.0 if not objs else max((len(v) / len(objs) for v in groups.values()), default=1.0)
    spread = 0.0
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        xs = [float(objs[i].get("pos", [0.0, 0.0, 0.0])[0]) for i in idxs]
        zs = [float(objs[i].get("pos", [0.0, 0.0, 0.0])[2]) for i in idxs]
        cluster = max(1e-4, (max(xs) - min(xs) + 0.05) * (max(zs) - min(zs) + 0.05))
        footprint = sum(float(_object_xz_polygon(objs[i]).area) for i in idxs)
        desired = 0.35 * footprint + 0.08 * max(0, len(idxs) - 2)
        spread += max(0.0, desired - cluster) / max(room_area, 1e-6)
    flow = 0.0
    if floor is not None and union_poly is not None:
        cx, cz = room_center_xz(scene)
        ratio = cfg.corridor_width_ratio if cfg else 0.18
        if room_w >= room_h:
            half = ratio * room_h * 0.5
            corridor = ShapelyPolygon([(minx, cz - half), (maxx, cz - half), (maxx, cz + half), (minx, cz + half)])
        else:
            half = ratio * room_w * 0.5
            corridor = ShapelyPolygon([(cx - half, minz), (cx + half, minz), (cx + half, maxz), (cx - half, maxz)])
        corridor = corridor.intersection(floor)
        if corridor is not None and not corridor.is_empty and corridor.area > 1e-6:
            flow = max(0.0, float(union_poly.intersection(corridor).area) / float(corridor.area) - 0.22)
    mono_target = cfg.max_zone_monopoly_ratio if cfg else 0.72
    open_target = cfg.min_open_space_ratio if cfg else 0.42
    mono_penalty = max(0.0, max_zone_ratio - mono_target)
    open_penalty = max(0.0, open_target - open_ratio)
    structure_loss = 0.0
    return {
        "room_area": round(room_area, 6),
        "occupied_area": round(occupied, 6),
        "open_space_ratio": round(open_ratio, 6),
        "zone_count": float(zone_count),
        "max_zone_ratio": round(max_zone_ratio, 6),
        "spread_penalty": round(spread, 6),
        "flow_penalty": round(flow, 6),
        "monopoly_penalty": round(mono_penalty, 6),
        "open_penalty": round(open_penalty, 6),
        "structure_loss": round(structure_loss, 6),
    }


def _score_scene_full(scene: Dict[str, Any], role_graph: Optional[RoleGraph] = None, relation_priors: Optional[List[Dict[str, Any]]] = None, timing_stats: Optional[TimingStats] = None, cfg: Optional[Config] = None) -> Tuple[float, Dict[str, Any], float, float]:
    t0 = _now()
    metrics = dict(eval_scene(scene, is_debug=False))
    role_graph = role_graph or infer_role_graph(scene)
    rel_loss, _ = _compute_relation_loss(scene, role_graph, relation_priors or [])
    func_loss = compute_functional_loss(scene, role_graph, yaw_from_quaternion).total
    struct = _compute_structure_stats(scene, role_graph, cfg)
    metrics.update({
        "structure_stats": struct,
        "open_space_ratio": struct["open_space_ratio"],
        "zone_count": int(struct["zone_count"]),
        "max_zone_ratio": struct["max_zone_ratio"],
        "structure_loss": struct["structure_loss"],
    })
    score = _W_PBL * _get_float_metric(metrics, "total_pbl_loss") + _W_REL * rel_loss + _W_FUNC * func_loss
    if timing_stats is not None:
        timing_stats.eval_sec += _now() - t0
    return score, metrics, rel_loss, func_loss


def _score_scene_stable_judge(scene: Dict[str, Any], role_graph: RoleGraph, frozen_role_graph: RoleGraph, frozen_deterministic_priors: List[Dict[str, Any]], timing_stats: Optional[TimingStats], cfg: Config) -> Tuple[float, Dict[str, Any], float, float]:
    score, metrics, rel, func = _score_scene_full(scene, frozen_role_graph, frozen_deterministic_priors, timing_stats, cfg)
    metrics = dict(metrics)
    metrics["judge_stats"] = {
        "mode": "stable_dynamic_blend" if cfg.enable_dual_judge else "stable_only",
        "stable_judge_weight": float(cfg.stable_judge_weight),
        "frozen_priors_count": len(frozen_deterministic_priors),
        "dynamic_role_categories": getattr(role_graph, "categories", None),
    }
    return score, metrics, rel, func


# ------------------------------
# relation prior helpers
# ------------------------------

def _dedup_priors(priors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    uniq: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for item in priors:
        key = (item.get("src_idx"), item.get("tgt_idx"), item.get("type"), item.get("lo"), item.get("hi"))
        score = float(item.get("confidence", 0.0)) * float(item.get("weight", 1.0))
        old = uniq.get(key)
        old_score = -1.0 if old is None else float(old.get("confidence", 0.0)) * float(old.get("weight", 1.0))
        if old is None or score > old_score:
            uniq[key] = item
    return list(uniq.values())


def _normalize_text_key(x: Any) -> str:
    s = re.sub(r"\s+", " ", str(x or "").lower()).strip()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", s).strip()


def _relation_priority_to_weight(priority: Any) -> float:
    p = str(priority or "medium").strip().lower()
    return {"low": 0.8, "medium": 1.0, "high": 1.2, "critical": 1.4}.get(p, 1.0)


def _desc_bank(scene: Dict[str, Any], role_graph: RoleGraph, idx: int) -> str:
    obj = scene.get("objects", [])[idx]
    parts = [
        role_graph.categories[idx] if idx < len(role_graph.categories) else "",
        obj.get("category", ""),
        obj.get("type", ""),
        obj.get("desc", ""),
        obj.get("sampled_asset_desc", ""),
        _safe_object_prompt_text(obj),
        obj.get("sampled_asset_jid", ""),
        obj.get("jid", ""),
    ]
    return _normalize_text_key(" ".join(str(p) for p in parts if p))


def _find_best_obj_idx_by_desc(scene: Dict[str, Any], role_graph: RoleGraph, desc: Any, used_src: Optional[Set[int]] = None, preferred_idx: Optional[int] = None) -> Optional[int]:
    query = _normalize_text_key(desc)
    if not query:
        return preferred_idx
    q_tokens = set(query.split())
    best = None
    for i, _ in enumerate(scene.get("objects", [])):
        if used_src and i in used_src:
            continue
        bank = _desc_bank(scene, role_graph, i)
        if not bank:
            continue
        b_tokens = set(bank.split())
        overlap = len(q_tokens & b_tokens)
        substr = 1 if query in bank or bank in query else 0
        cat_boost = 1 if (i < len(role_graph.categories) and role_graph.categories[i] in q_tokens) else 0
        score = 10 * substr + 4 * overlap + cat_boost
        if preferred_idx is not None and i == preferred_idx:
            score += 0.5
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, i)
    return None if best is None else best[1]


def _allowed_side_pair(role_graph: RoleGraph, src_idx: int, tgt_idx: int) -> bool:
    cats = role_graph.categories
    a = cats[src_idx] if src_idx < len(cats) else ""
    b = cats[tgt_idx] if tgt_idx < len(cats) else ""
    valid_src = {"chair", "sofa", "bench", "nightstand", "plant", "lamp"}
    valid_tgt = {"table", "desk", "bed", "sofa", "coffee table", "tv stand"}
    return (a in valid_src and b in valid_tgt) or (b in valid_src and a in valid_tgt)


def _clean_relation_priors(priors: List[Dict[str, Any]], n_obj: int, min_confidence: float) -> List[Dict[str, Any]]:
    cleaned, per_src = [], {}
    for item in priors:
        if not isinstance(item, dict):
            continue
        rel_type = str(item.get("type", "")).strip()
        if rel_type not in REL_TYPES:
            continue
        si = item.get("src_idx")
        if not isinstance(si, int) or not (0 <= si < n_obj):
            continue
        ti = item.get("tgt_idx")
        if rel_type not in {"against_wall", "parallel"}:
            if not isinstance(ti, int) or not (0 <= ti < n_obj) or ti == si:
                continue
        else:
            ti = None
        conf = float(item.get("confidence", 1.0))
        if conf < min_confidence or per_src.get(si, 0) >= 3:
            continue
        payload = {"src_idx": si, "tgt_idx": ti, "type": rel_type, "confidence": conf, "weight": float(item.get("weight", 1.0)), "reason": str(item.get("reason", ""))}
        if rel_type == "distance_band":
            if item.get("lo") is not None:
                payload["lo"] = float(item["lo"])
            if item.get("hi") is not None:
                payload["hi"] = float(item["hi"])
        cleaned.append(payload)
        per_src[si] = per_src.get(si, 0) + 1
        if len(cleaned) >= 32:
            break
    return cleaned


def _safe_generate_relation_priors(generator: GPTVLMovePromptGeneratorV5, diag_path: Path, top_path: Path, scene: Dict[str, Any], extra_context: str, retries: int, temperature: float, max_tokens: int, min_confidence: float, out_dir: Path) -> Optional[List[Dict[str, Any]]]:
    if not hasattr(generator, "generate_relation_priors"):
        _write_text(out_dir / "vlm_relation_priors.error.txt", "generator does not implement generate_relation_priors(); fallback to deterministic relation priors.\n")
        return None
    last_exc = None
    for retry in range(retries):
        try:
            result = generator.generate_relation_priors(diag_image_path=diag_path, top_image_path=top_path, scene=scene, extra_context=extra_context, temperature=temperature, max_tokens=max_tokens)
            raw = json.loads(getattr(result, "json_text", "{}"))
            cleaned = _clean_relation_priors(raw.get("relations", []), len(scene.get("objects", [])), min_confidence)
            _write_json(out_dir / "vlm_relation_priors.json", {"raw_text": getattr(result, "raw_text", ""), "raw_json": raw, "cleaned_relations": cleaned})
            return cleaned
        except Exception as exc:
            last_exc = exc
            _write_text(out_dir / f"vlm_relation_priors_retry{retry + 1}.error.txt", traceback.format_exc())
    return None if last_exc is not None else None


def _canonicalize_single_relation(scene: Dict[str, Any], role_graph: RoleGraph, item: Dict[str, Any], cfg: Config) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    drops, out = [], []
    if not isinstance(item, dict):
        return out, [{"item": item, "reason": "not_dict"}]
    rel_type = str(item.get("type", "")).strip()
    si, ti = item.get("src_idx"), item.get("tgt_idx")
    n_obj = len(scene.get("objects", []))
    if not isinstance(si, int) or not (0 <= si < n_obj):
        return out, [{"item": item, "reason": "bad_src"}]
    if rel_type not in REL_TYPES | REL_LEGACY_ALIAS_TYPES:
        return out, [{"item": item, "reason": "bad_type"}]
    base = {
        "src_idx": si,
        "tgt_idx": ti,
        "type": rel_type,
        "confidence": float(item.get("confidence", 1.0)),
        "weight": float(item.get("weight", 1.0)),
        "reason": str(item.get("reason", "")).strip(),
    }
    if rel_type in {"against_wall", "parallel"}:
        base["tgt_idx"] = None
        base["type"] = rel_type
        out.append(base)
        return out, drops
    if not isinstance(ti, int) or not (0 <= ti < n_obj) or ti == si:
        return out, [{"item": item, "reason": "bad_tgt"}]
    if rel_type == "near":
        if not cfg.relation_alias_near_to_band:
            drops.append({"item": item, "reason": "near_dropped"})
            return out, drops
        base.update({"type": "distance_band", "lo": max(0.25, _pair_target_dist(scene["objects"][si], scene["objects"][ti], 0.5, 0.4) - 0.5), "hi": _pair_target_dist(scene["objects"][si], scene["objects"][ti], 0.5, 0.4) + 0.8})
        out.append(base)
        return out, drops
    if rel_type == "facing_pair":
        if not cfg.relation_expand_facing_pair:
            drops.append({"item": item, "reason": "facing_pair_dropped"})
            return out, drops
        a = dict(base, type="facing", weight=0.5 * base["weight"], reason=(base["reason"] + " | expand:facing_pair:a").strip(" |"))
        b = dict(base, src_idx=ti, tgt_idx=si, type="facing", weight=0.5 * base["weight"], reason=(base["reason"] + " | expand:facing_pair:b").strip(" |"))
        out.extend([a, b])
        return out, drops
    if rel_type == "side_of":
        if not cfg.relation_allow_side_of or not _allowed_side_pair(role_graph, si, ti):
            drops.append({"item": item, "reason": "side_of_filtered"})
            return out, drops
        base["type"] = "side_of"
        out.append(base)
        return out, drops
    if rel_type == "in_front_of":
        if cfg.relation_drop_in_front_of:
            drops.append({"item": item, "reason": "in_front_of_dropped"})
            return out, drops
        out.append(base)
        return out, drops
    if rel_type in REL_CANONICAL_TYPES:
        if rel_type == "distance_band":
            if item.get("lo") is not None:
                base["lo"] = float(item["lo"])
            if item.get("hi") is not None:
                base["hi"] = float(item["hi"])
        out.append(base)
        return out, drops
    drops.append({"item": item, "reason": "unsupported_after_canonicalization"})
    return out, drops


def _canonicalize_relation_priors(scene: Dict[str, Any], role_graph: RoleGraph, priors: List[Dict[str, Any]], cfg: Config) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    out, log = [], []
    for item in priors:
        a, b = _canonicalize_single_relation(scene, role_graph, item, cfg)
        out.extend(a)
        log.extend(b)
    return _dedup_priors(out), log


def _should_refresh_relation_priors(step: int, cache: Optional[List[Dict[str, Any]]], stagnation_count: int, cfg: Config) -> bool:
    mode = str(cfg.relation_refresh_mode).strip().lower()
    if mode == "never":
        return False
    if mode == "every_step":
        return True
    if mode == "on_stagnation":
        return cache is None or stagnation_count >= 1
    return cache is None and step == 0


def _select_relation_priors_for_optimizer(deterministic_priors: List[Dict[str, Any]], raw_vlm_priors: Optional[List[Dict[str, Any]]], canonical_vlm_priors: Optional[List[Dict[str, Any]]], cfg: Config) -> List[Dict[str, Any]]:
    mode = str(cfg.relation_use_mode).strip().lower()
    if mode == "deterministic_only":
        return list(deterministic_priors)
    if mode == "raw_vlm":
        return list(raw_vlm_priors or deterministic_priors)
    if mode == "canonical_vlm":
        return list(canonical_vlm_priors or deterministic_priors)
    return _dedup_priors(list(deterministic_priors) + list(canonical_vlm_priors or []))


def _canonicalize_zero_shot_relation_plan(plan: Optional[Dict[str, Any]], cfg: Config) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not isinstance(plan, dict):
        return [], [{"reason": "plan_not_dict"}]
    rels = plan.get("relation_plan", plan.get("relations"))
    if not isinstance(rels, list):
        return [], [{"reason": "missing_relation_plan_list"}]
    cleaned, log = [], []
    for item in rels:
        if not isinstance(item, dict):
            log.append({"item": item, "reason": "not_dict"})
            continue
        t = str(item.get("type", "")).strip()
        if t not in REL_TYPES | REL_LEGACY_ALIAS_TYPES:
            log.append({"item": item, "reason": "invalid_type"})
            continue
        cleaned.append({
            "src_desc": _normalize_text_key(item.get("src_desc", item.get("source", item.get("src", "")))),
            "tgt_desc": None if item.get("tgt_desc", item.get("target", item.get("tgt"))) is None else _normalize_text_key(item.get("tgt_desc", item.get("target", item.get("tgt", "")))),
            "type": t,
            "priority": str(item.get("priority", "medium")).strip().lower(),
            "reason": str(item.get("reason", "")).strip(),
            "confidence": float(item.get("confidence", 0.8)),
            "weight": float(item.get("weight", 1.0)),
        })
    return cleaned, log


def _ground_zero_shot_relation_plan(scene: Dict[str, Any], role_graph: RoleGraph, relation_plan: Sequence[Dict[str, Any]], cfg: Config) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grounded, drop_log, used_src = [], [], set()
    for item in relation_plan:
        si = _find_best_obj_idx_by_desc(scene, role_graph, item.get("src_desc"), used_src)
        if si is None:
            drop_log.append({"item": item, "reason": "src_not_grounded"})
            continue
        t = str(item.get("type", "")).strip()
        ti = None
        if t not in {"against_wall", "parallel"}:
            ti = _find_best_obj_idx_by_desc(scene, role_graph, item.get("tgt_desc"), preferred_idx=None)
            if ti is None or ti == si:
                drop_log.append({"item": item, "reason": "tgt_not_grounded"})
                continue
        used_src.add(si)
        grounded.append({
            "src_idx": si,
            "tgt_idx": ti,
            "type": t,
            "confidence": float(item.get("confidence", 0.8)),
            "weight": float(item.get("weight", 1.0)) * _relation_priority_to_weight(item.get("priority", "medium")) * float(cfg.zero_shot_relation_weight_scale),
            "reason": str(item.get("reason", "")).strip() or "zero_shot_relation_plan",
        })
    canon, canon_drop = _canonicalize_relation_priors(scene, role_graph, grounded, cfg)
    return canon, drop_log + canon_drop


def _select_zero_shot_base_priors(base_priors: List[Dict[str, Any]], grounded_zero_shot_priors: List[Dict[str, Any]], cfg: Config) -> List[Dict[str, Any]]:
    if not cfg.use_zero_shot_relation_plan or not grounded_zero_shot_priors:
        return list(base_priors)
    mode = str(cfg.zero_shot_relation_use_mode).strip().lower()
    if mode in {"raw_vlm", "canonical_vlm"}:
        return list(grounded_zero_shot_priors)
    if mode == "deterministic_only":
        return list(base_priors)
    return _dedup_priors(list(base_priors) + list(grounded_zero_shot_priors))


# ------------------------------
# prompt / structured plan helpers
# ------------------------------

def _compose_step_extra_context(base_context: str, history_records: Sequence[Dict[str, Any]], planner_mode: str, max_history_steps: int) -> str:
    parts = [base_context.strip()] if base_context.strip() else []
    parts += [
        "HISTORY-AWARE ITERATIVE PLANNING MODE:\nUse the step history below as additional context. Avoid repeating rejected edits. Diagnose why the previous proposal failed, then propose a different fix-as-you-go action plan. Prefer preserving global functional zones while repairing local relation issues.",
        f"CURRENT PLANNER MODE: {planner_mode}",
    ]
    if history_records:
        parts.append("RECENT STEP HISTORY:")
        for rec in list(history_records)[-max_history_steps:]:
            parts.append(
                f"- step {rec.get('step', -1):02d} mode={rec.get('planner_mode', 'unknown')} accepted={rec.get('accepted', False)} reject={rec.get('reject_reason', 'none')} "
                f"score {rec.get('score_before', 0.0):.4f}->{rec.get('score_after', 0.0):.4f}; rel {rec.get('rel_before', 0.0):.4f}->{rec.get('rel_after', 0.0):.4f}; "
                f"func {rec.get('func_before', 0.0):.4f}->{rec.get('func_after', 0.0):.4f}; struct {rec.get('struct_before', 0.0):.4f}->{rec.get('struct_after', 0.0):.4f}; "
                f"zones {rec.get('zone_before', 0)}->{rec.get('zone_after', 0)}; mono {rec.get('mono_before', 0.0):.3f}->{rec.get('mono_after', 0.0):.3f}"
            )
            if rec.get("diagnosis"):
                parts.append(f"  diagnosis: {rec['diagnosis']}")
    return "\n\n".join(p for p in parts if p).strip()


def _compose_v13_structured_repair_context(extra_context: str, scene: Dict[str, Any], metrics_before: Dict[str, Any], rel_before: float, func_before: float, struct_before: Dict[str, Any], current_priors: List[Dict[str, Any]], cfg: Config, allowed_target_indices: Optional[Sequence[int]] = None) -> str:
    allowed = [] if not allowed_target_indices else sorted({int(i) for i in allowed_target_indices if isinstance(i, int)})
    role_graph = infer_role_graph(scene)
    obj_lines = []
    for i, obj in enumerate(scene.get("objects", [])):
        pos = obj.get("pos", [0.0, 0.0, 0.0])
        scale = obj.get("scale") if isinstance(obj.get("scale"), list) else None
        obj_lines.append({
            "object_index": i,
            "category": role_graph.categories[i] if i < len(role_graph.categories) else "unknown",
            "desc": obj.get("desc", obj.get("description", "")),
            "pos": [round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)],
            "yaw_deg": round(float(yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))), 2),
            "scale": scale,
        })
    priors_preview = current_priors[: min(len(current_priors), 24)]
    constraints = [
        "Return ONLY one JSON object with key `actions`.",
        "Do not output explanations.",
        "Do not output markdown fences.",
        f"Return at most {cfg.structured_plan_max_actions} actions.",
        "Allowed actions only:",
        'move: {"action":"move","object_index":0,"dx":0.0,"dy":0.0,"dz":0.0}',
        'rotate: {"action":"rotate","object_index":0,"yaw_deg":0.0}',
        'scale: {"action":"scale","object_index":0,"sx":1.0,"sy":1.0,"sz":1.0}',
        "move uses relative translation in meters.",
        "rotate uses relative yaw in degrees.",
        "scale uses multiplicative factors.",
        "Prefer minimal but effective edits.",
        "Do not invent, delete, or replace objects.",
    ]
    if allowed:
        constraints.append(f"Only edit these object_index values: {allowed}.")
    return "\n\n".join([
        extra_context.strip(),
        "OUTPUT FORMAT:\n{" + '"actions":[{"action":"move","object_index":0,"dx":0.0,"dy":0.0,"dz":0.0}]' + "}",
        "CONSTRAINTS:\n- " + "\n- ".join(constraints),
        "CURRENT OBJECTS:\n" + json.dumps(obj_lines, ensure_ascii=False, indent=2),
        "CURRENT METRICS:\n" + json.dumps({
            "pbl": round(_get_float_metric(metrics_before, "total_pbl_loss"), 6),
            "oob": round(_get_float_metric(metrics_before, "total_oob_loss"), 6),
            "mbl": round(_get_float_metric(metrics_before, "total_mbl_loss"), 6),
            "rel": round(rel_before, 4),
            "func": round(func_before, 4),
            "structure": struct_before,
        }, ensure_ascii=False, indent=2),
        "CURRENT RELATION PRIORS (soft hints only):\n" + json.dumps(priors_preview, ensure_ascii=False, indent=2),
    ]).strip()
def _safe_generate_move_prompt(generator: GPTVLMovePromptGeneratorV5, diag_path: Path, top_path: Path, scene: Dict[str, Any], extra_context: str, retries: int, temperature: float, max_tokens: int, trial_dir: Path):
    last_exc = None
    for retry in range(retries):
        try:
            return generator.generate(diag_image_path=diag_path, top_image_path=top_path, scene=scene, extra_context=extra_context, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            last_exc = exc
            _write_text(trial_dir / f"move_prompt_retry{retry + 1}.error.txt", traceback.format_exc())
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("move prompt generation failed")


def _extract_text_from_prompt_result(prompt_result: Any) -> str:
    for key in ("move_prompt", "json_text", "raw_text", "text", "response"):
        if hasattr(prompt_result, key):
            val = getattr(prompt_result, key)
            if isinstance(val, str) and val.strip():
                return val
    return str(prompt_result)


def _objectedit_kwargs() -> Set[str]:
    if is_dataclass(ObjectEdit):
        return {f.name for f in fields(ObjectEdit)}
    try:
        return {k for k in inspect.signature(ObjectEdit).parameters if k != "self"}
    except Exception:
        return {"jid_prefix", "hint_pos", "dx", "dz", "relative_yaw_deg", "target_yaw_deg", "no_rotation"}


_OBJECTEDIT_FIELDS = _objectedit_kwargs()


def _make_object_edit(**kwargs) -> ObjectEdit:
    payload = {k: v for k, v in kwargs.items() if k in _OBJECTEDIT_FIELDS}
    return ObjectEdit(**payload)


def _find_obj_idx_for_edit(scene: Dict[str, Any], edit: ObjectEdit) -> Optional[int]:
    obj_idx = getattr(edit, "object_index", None)
    if isinstance(obj_idx, int) and 0 <= obj_idx < len(scene.get("objects", [])):
        return int(obj_idx)
    best_idx, best_dist = None, float("inf")
    hint = _normalize_hint_pos(getattr(edit, "hint_pos", None))
    prefix = str(getattr(edit, "jid_prefix", "") or "").lower()
    for i, obj in enumerate(scene.get("objects", [])):
        jid = str(obj.get("sampled_asset_jid") or obj.get("jid") or obj.get("sampled_jid") or "")
        if prefix and jid[: len(prefix)].lower() != prefix:
            continue
        if hint is None:
            return i
        pos = obj.get("pos", [0.0, 0.0, 0.0])
        try:
            d = math.hypot(float(pos[0]) - hint[0], float(pos[2]) - hint[2])
        except Exception:
            continue
        if d < best_dist:
            best_idx, best_dist = i, d
    return best_idx

def _resolve_action_target_idx(scene: Dict[str, Any], action: Dict[str, Any]) -> Optional[int]:
    obj_idx = action.get("object_index", action.get("target_idx"))
    if isinstance(obj_idx, int) and 0 <= obj_idx < len(scene.get("objects", [])):
        return obj_idx
    try:
        obj_idx = int(obj_idx)
        if 0 <= obj_idx < len(scene.get("objects", [])):
            return obj_idx
    except Exception:
        pass
    role_graph = infer_role_graph(scene)
    desc = action.get("target_desc", action.get("desc", action.get("target")))
    if desc:
        return _find_best_obj_idx_by_desc(scene, role_graph, desc)
    prefix = str(action.get("jid_prefix", "") or "")
    hint = _normalize_hint_pos(action.get("hint_pos"))
    if prefix:
        dummy = _make_object_edit(jid_prefix=prefix, hint_pos=hint, dx=0.0, dz=0.0, no_rotation=True)
        return _find_obj_idx_for_edit(scene, dummy)
    return None

def _convert_v15_action_to_edit(scene: Dict[str, Any], action: Dict[str, Any]) -> Optional[ObjectEdit]:
    idx = action.get("target_idx")
    if not isinstance(idx, int) or not (0 <= idx < len(scene.get("objects", []))):
        return None
    obj = scene["objects"][idx]
    prefix = str(obj.get("sampled_asset_jid") or obj.get("jid") or obj.get("sampled_jid") or "")[:6]
    hint = obj.get("pos", [0.0, 0.0, 0.0])
    op = str(action.get("op", "move_delta")).strip()
    dx, dz = _safe_float(action.get("dx", 0.0)), _safe_float(action.get("dz", 0.0))
    yaw = None if action.get("yaw_deg") is None else _normalize_angle(_safe_float(action.get("yaw_deg"), 0.0))
    dyaw = _safe_float(action.get("relative_yaw_deg", action.get("dyaw", action.get("yaw_delta_deg", 0.0))), 0.0)
    if op in {"move_delta", "translate", "move"}:
        return _make_object_edit(jid_prefix=prefix, hint_pos=hint, dx=dx, dz=dz, no_rotation=True)
    if op in {"rotate_delta", "rotate"}:
        return _make_object_edit(jid_prefix=prefix, hint_pos=hint, dx=0.0, dz=0.0, relative_yaw_deg=dyaw, no_rotation=False)
    if op == "set_yaw":
        return _make_object_edit(jid_prefix=prefix, hint_pos=hint, dx=0.0, dz=0.0, target_yaw_deg=yaw, no_rotation=False)
    if op == "face_object":
        target = action.get("face_target_idx")
        if not isinstance(target, int) or not (0 <= target < len(scene.get("objects", []))):
            target = _resolve_action_target_idx(scene, {"target_desc": action.get("target_desc")})
        if not isinstance(target, int):
            return None
        spos, tpos = obj.get("pos", [0.0, 0.0, 0.0]), scene["objects"][target].get("pos", [0.0, 0.0, 0.0])
        target_yaw = _normalize_angle(math.degrees(math.atan2(tpos[0] - spos[0], tpos[2] - spos[2])))
        return _make_object_edit(jid_prefix=prefix, hint_pos=hint, dx=dx, dz=dz, target_yaw_deg=target_yaw, no_rotation=False)
    if op in {"align_wall", "snap_wall"}:
        current_yaw = yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
        mode = str(action.get("mode", action.get("wall_mode", "parallel"))).strip().lower()
        target_yaw = _nearest_normal_axis_yaw(scene, hint, current_yaw) if mode in {"normal", "against"} else _nearest_parallel_wall_yaw(scene, hint, current_yaw)
        return _make_object_edit(jid_prefix=prefix, hint_pos=hint, dx=dx, dz=dz, target_yaw_deg=target_yaw, no_rotation=False)
    return None


def _parse_json_loose(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
        if m:
            text = m.group(1)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize_action_json_payload(payload: Any, max_actions: int) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"schema": _V15_SCHEMA, "actions": []}
    actions = payload.get("actions")
    if not isinstance(actions, list):
        actions = []
    normalized = []
    for item in actions[: max(0, max_actions)]:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip().lower()
        if action not in _V15_ALLOWED_ACTIONS:
            continue
        obj_idx = item.get("object_index", item.get("target_idx"))
        try:
            obj_idx = int(obj_idx)
        except Exception:
            continue
        if action == "move":
            normalized.append({
                "action": "move",
                "object_index": obj_idx,
                "dx": float(item.get("dx", 0.0)),
                "dy": float(item.get("dy", 0.0)),
                "dz": float(item.get("dz", 0.0)),
            })
        elif action == "rotate":
            normalized.append({
                "action": "rotate",
                "object_index": obj_idx,
                "yaw_deg": float(item.get("yaw_deg", 0.0)),
            })
        elif action == "scale":
            normalized.append({
                "action": "scale",
                "object_index": obj_idx,
                "sx": float(item.get("sx", 1.0)),
                "sy": float(item.get("sy", 1.0)),
                "sz": float(item.get("sz", 1.0)),
            })
    out = {"schema": _V15_SCHEMA, "actions": normalized}
    if isinstance(payload.get("diagnosis"), list):
        out["diagnosis"] = payload.get("diagnosis")
    return out


def _action_payload_to_edits(payload: Dict[str, Any]) -> Tuple[List[ObjectEdit], List[str]]:
    edits: List[ObjectEdit] = []
    warnings: List[str] = []
    actions = payload.get("actions", [])
    if not isinstance(actions, list):
        return edits, ["payload.actions_not_list"]
    for item in actions:
        if not isinstance(item, dict):
            warnings.append("skip_non_dict_action")
            continue
        action = str(item.get("action", "")).strip().lower()
        try:
            obj_idx = int(item.get("object_index"))
        except Exception:
            warnings.append(f"skip_action_without_valid_object_index:{item!r}")
            continue
        if action == "move":
            dx = _safe_float(item.get("dx", 0.0), 0.0)
            dy = _safe_float(item.get("dy", 0.0), 0.0)
            dz = _safe_float(item.get("dz", 0.0), 0.0)
            if abs(dx) <= 1e-9 and abs(dy) <= 1e-9 and abs(dz) <= 1e-9:
                continue
            edits.append(ObjectEdit(object_index=obj_idx, description=f"obj_{obj_idx}", dx=dx, dy=dy, dz=dz, raw_line=json.dumps(item, ensure_ascii=False)))
        elif action == "rotate":
            yaw_deg = _safe_float(item.get("yaw_deg", 0.0), 0.0)
            if abs(yaw_deg) <= 1e-9:
                continue
            edits.append(ObjectEdit(object_index=obj_idx, description=f"obj_{obj_idx}", relative_yaw_deg=yaw_deg, raw_line=json.dumps(item, ensure_ascii=False)))
        elif action == "scale":
            sx = _safe_float(item.get("sx", 1.0), 1.0)
            sy = _safe_float(item.get("sy", 1.0), 1.0)
            sz = _safe_float(item.get("sz", 1.0), 1.0)
            if abs(sx - 1.0) <= 1e-9 and abs(sy - 1.0) <= 1e-9 and abs(sz - 1.0) <= 1e-9:
                continue
            edits.append(ObjectEdit(object_index=obj_idx, description=f"obj_{obj_idx}", scale_delta=[sx, sy, sz], raw_line=json.dumps(item, ensure_ascii=False)))
        else:
            warnings.append(f"unknown_action:{action}")
    return edits, warnings


def _parse_v15_structured_repair_plan(raw_text: str, scene: Dict[str, Any], cfg: Config) -> Tuple[Dict[str, Any], List[ObjectEdit], List[str]]:
    payload = _parse_json_loose(raw_text)
    plan = _normalize_action_json_payload(payload, cfg.structured_plan_max_actions)
    edits, warnings = _action_payload_to_edits(plan)
    return plan, edits, warnings


def _filter_structured_plan_and_edits_by_allowed_indices(scene: Dict[str, Any], structured_plan: Dict[str, Any], edits: List[ObjectEdit], allowed_indices: Optional[Sequence[int]]) -> Tuple[Dict[str, Any], List[ObjectEdit], List[Dict[str, Any]]]:
    if not allowed_indices:
        return structured_plan, edits, []
    allowed = {int(i) for i in allowed_indices if isinstance(i, int)}
    dropped = []
    filtered_actions = []
    if isinstance(structured_plan.get("actions"), list):
        for action in structured_plan["actions"]:
            idx = _resolve_action_target_idx(scene, action) if isinstance(action, dict) else None
            if idx is None or idx in allowed:
                filtered_actions.append(action)
            else:
                dropped.append({"kind": "action", "object_index": idx, "reason": "target_outside_current_group"})
    structured_plan = dict(structured_plan)
    structured_plan["actions"] = filtered_actions
    filtered_edits = []
    for edit in edits:
        idx = _find_obj_idx_for_edit(scene, edit)
        if idx is None or idx in allowed:
            filtered_edits.append(edit)
        else:
            dropped.append({"kind": "edit", "object_index": idx, "reason": "target_outside_current_group"})
    return structured_plan, filtered_edits, dropped


def _parse_move_prompt_v13_or_legacy(raw_text: str, scene: Dict[str, Any], cfg: Config, step_dir: Path):
    if cfg.use_structured_repair_plan:
        try:
            plan, edits, warnings = _parse_v15_structured_repair_plan(raw_text, scene, cfg)
            _write_json(step_dir / "scene_repair_plan.json", plan)
            legacy_empty = parse_move_prompt("")
            return legacy_empty, edits, "json_actions", plan, warnings
        except Exception:
            _write_text(step_dir / "scene_repair_plan_parse.error.txt", traceback.format_exc())
    legacy = parse_move_prompt(raw_text)
    plan = {"schema": _V15_SCHEMA, "actions": []}
    return legacy, list(getattr(legacy, "edits", [])), "legacy_move_prompt", plan, list(getattr(legacy, "parse_warnings", []))

# ------------------------------
# local search optimizer
# ------------------------------

def _relation_penalty_for_object(scene: Dict[str, Any], role_graph: RoleGraph, priors: List[Dict[str, Any]], idx: int) -> float:
    _, violations = _compute_relation_loss(scene, role_graph, priors)
    return sum(v["penalty"] for v in violations if v.get("src_idx") == idx or v.get("tgt_idx") == idx)


def _functional_penalty_for_object(scene: Dict[str, Any], role_graph: RoleGraph, idx: int) -> float:
    total = 0.0
    for item in compute_functional_loss(scene, role_graph, yaw_from_quaternion).violations:
        if item.get("idx") == idx or item.get("anchor_idx") == idx or item.get("src_idx") == idx or item.get("tgt_idx") == idx:
            total += float(item.get("penalty", 0.0))
    return total


def _quick_candidate_proxy_score(scene: Dict[str, Any], role_graph: RoleGraph, priors: List[Dict[str, Any]], idx: int, cfg: Optional[Config] = None) -> float:
    floor = _get_floor_polygon(scene)
    oob = 0.0 if floor is None else compute_oob(scene["objects"][idx], floor, scene.get("bounds_bottom", []), scene.get("bounds_top", []), is_debug=False)
    collision = sum(area for _, _, _, area in _collision_neighbors(scene, idx))
    relation = _relation_penalty_for_object(scene, role_graph, priors, idx)
    functional = _functional_penalty_for_object(scene, role_graph, idx)
    struct = _compute_structure_stats(scene, role_graph, cfg)
    pbl_proxy = oob + collision
    return _W_PBL * pbl_proxy + _W_REL * relation + _W_FUNC * functional


def _apply_delta(scene: Dict[str, Any], idx: int, dx: float = 0.0, dz: float = 0.0, dyaw: float = 0.0, yaw_abs: Optional[float] = None) -> Dict[str, Any]:
    obj = copy.deepcopy(scene["objects"][idx])
    pos = list(obj.get("pos", [0.0, 0.0, 0.0]))
    pos[0], pos[2] = pos[0] + dx, pos[2] + dz
    obj["pos"] = pos
    if yaw_abs is not None:
        obj["rot"] = quaternion_from_yaw(_normalize_angle(yaw_abs))
    elif abs(dyaw) > 1e-9:
        yaw = yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
        obj["rot"] = quaternion_from_yaw(_normalize_angle(yaw + dyaw))
    return _clone_scene_with_updated_object(scene, idx, obj)


def _anchor_pose_candidate(scene: Dict[str, Any], role_graph: RoleGraph, idx: int) -> Optional[Dict[str, Any]]:
    objs, cat, obj = scene.get("objects", []), role_graph.categories[idx], scene.get("objects", [])[idx]
    pos = obj.get("pos", [0.0, 0.0, 0.0])
    if idx in role_graph.accessory_to_anchor:
        anchor_idx = role_graph.accessory_to_anchor[idx]
        objs[idx]["_category"], objs[anchor_idx]["_category"] = cat, role_graph.categories[anchor_idx]
        target_pos, target_yaw = target_pose_for_attachment(scene, objs, idx, anchor_idx, yaw_from_quaternion)
        return {"kind": "role_attachment", "dx": target_pos[0] - pos[0], "dz": target_pos[2] - pos[2], "yaw_abs": target_yaw, "dyaw": 0.0}
    if cat == "bed":
        yaw_abs = _nearest_normal_axis_yaw(scene, pos, yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0])))
        return None if yaw_abs is None else {"kind": "bed_align_wall", "dx": 0.0, "dz": 0.0, "yaw_abs": yaw_abs, "dyaw": 0.0}
    if cat in {"table", "desk", "counter", "coffee table", "cabinet", "console table"}:
        yaw_abs = _nearest_parallel_wall_yaw(scene, pos, yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0])))
        return None if yaw_abs is None else {"kind": "parallel_wall", "dx": 0.0, "dz": 0.0, "yaw_abs": yaw_abs, "dyaw": 0.0}
    return None


def _zone_release_candidates(scene: Dict[str, Any], role_graph: RoleGraph, idx: int, cfg: Config) -> List[Dict[str, Any]]:
    if not cfg.add_zone_release_candidates or len(scene.get("objects", [])) < 4:
        return []
    groups = _collect_zone_groups(role_graph, len(scene.get("objects", [])))
    if not groups:
        return []
    dominant_label, dominant_idxs = max(groups.items(), key=lambda kv: len(kv[1]))
    if idx not in dominant_idxs or len(dominant_idxs) / max(1, len(scene.get("objects", []))) <= cfg.max_zone_monopoly_ratio:
        return []
    if idx in role_graph.accessory_to_anchor.values():
        return []
    cat = role_graph.categories[idx]
    if cat not in {"chair", "lamp", "table", "coffee table", "sofa", "bench", "plant", "nightstand"}:
        return []
    objs = scene.get("objects", [])
    dom_x = sum(float(objs[i].get("pos", [0.0, 0.0, 0.0])[0]) for i in dominant_idxs) / max(1, len(dominant_idxs))
    dom_z = sum(float(objs[i].get("pos", [0.0, 0.0, 0.0])[2]) for i in dominant_idxs) / max(1, len(dominant_idxs))
    minx, maxx, minz, maxz = _room_extents_xz(scene)
    inset_x, inset_z = (maxx - minx) * cfg.zone_release_inset_ratio, (maxz - minz) * cfg.zone_release_inset_ratio
    anchors = [(minx + inset_x, minz + inset_z), (minx + inset_x, maxz - inset_z), (maxx - inset_x, minz + inset_z), (maxx - inset_x, maxz - inset_z)]
    cur = objs[idx].get("pos", [0.0, 0.0, 0.0])
    scored = []
    for tx, tz in anchors:
        dist_from_dom = math.hypot(tx - dom_x, tz - dom_z)
        move_cost = math.hypot(tx - float(cur[0]), tz - float(cur[2]))
        scored.append((1.2 * dist_from_dom - 0.35 * move_cost, tx, tz))
    scored.sort(reverse=True)
    out = []
    for _, tx, tz in scored[:2]:
        yaw_abs = _normalize_angle(math.degrees(math.atan2(dom_x - tx, dom_z - tz))) if cat in {"chair", "sofa", "bench"} else None
        out.append({"kind": "zone_release", "dx": tx - float(cur[0]), "dz": tz - float(cur[2]), "dyaw": 0.0, "yaw_abs": yaw_abs})
    return out


def _generate_candidates(scene: Dict[str, Any], role_graph: RoleGraph, idx: int, bias_edit: Optional[ObjectEdit], cfg: Config) -> List[Dict[str, Any]]:
    obj = scene["objects"][idx]
    cat = role_graph.categories[idx]
    step_xy = cfg.step_xy * max(0.8, min(1.6, obj_diag_size_xz(obj)))
    step_yaw = cfg.step_yaw
    cands: List[Dict[str, Any]] = []
    if bias_edit is not None:
        dx, dz = _safe_float(getattr(bias_edit, "dx", 0.0), 0.0), _safe_float(getattr(bias_edit, "dz", 0.0), 0.0)
        yaw_abs = None if getattr(bias_edit, "target_yaw_deg", None) is None else _safe_float(getattr(bias_edit, "target_yaw_deg"), 0.0)
        dyaw = 0.0 if getattr(bias_edit, "no_rotation", False) else _safe_float(getattr(bias_edit, "relative_yaw_deg", 0.0), 0.0)
        cands.append({"kind": "gpt_bias", "dx": dx, "dz": dz, "dyaw": dyaw, "yaw_abs": yaw_abs})
    anchor_cand = _anchor_pose_candidate(scene, role_graph, idx)
    if anchor_cand is not None:
        cands.append(anchor_cand)
    allow_free_yaw = cat not in {"bed", "table", "desk", "counter", "coffee table"}
    allow_diag = cat not in {"table", "counter"}
    for sign in (-1.0, 1.0):
        cands += [{"kind": "axis_x", "dx": sign * step_xy, "dz": 0.0, "dyaw": 0.0, "yaw_abs": None}, {"kind": "axis_z", "dx": 0.0, "dz": sign * step_xy, "dyaw": 0.0, "yaw_abs": None}]
        if allow_free_yaw:
            cands.append({"kind": "axis_yaw", "dx": 0.0, "dz": 0.0, "dyaw": sign * step_yaw, "yaw_abs": None})
        if allow_diag:
            cands += [{"kind": "diag", "dx": sign * step_xy * 0.7, "dz": sign * step_xy * 0.7, "dyaw": 0.0, "yaw_abs": None}, {"kind": "diag2", "dx": sign * step_xy * 0.7, "dz": -sign * step_xy * 0.7, "dyaw": 0.0, "yaw_abs": None}]
    cands.extend(_zone_release_candidates(scene, role_graph, idx, cfg))
    uniq = {}
    for c in cands:
        key = (round(c.get("dx", 0.0) * 100), round(c.get("dz", 0.0) * 100), round(c.get("dyaw", 0.0)), None if c.get("yaw_abs") is None else round(float(c["yaw_abs"])))
        uniq[key] = c
    return list(uniq.values())


def _prioritized_object_indices(scene: Dict[str, Any], role_graph: RoleGraph, edits: List[ObjectEdit], metrics: Dict[str, Any], cfg: Config, allowed_indices: Optional[Sequence[int]] = None) -> List[int]:
    hotspot = metrics.get("obj_with_highest_pbl_loss", {})
    hotspot_idx = hotspot.get("idx") if isinstance(hotspot, dict) else None
    edited = []
    for e in edits:
        idx = _find_obj_idx_for_edit(scene, e)
        if idx is not None and idx not in edited:
            edited.append(idx)
    lock_major = _get_float_metric(metrics, "total_pbl_loss") <= cfg.anchor_lock_pbl_threshold
    order = optimization_stage_order(scene, role_graph, edited, hotspot_idx, cfg.max_objects_per_round, lock_major)
    if allowed_indices:
        allowed = {int(i) for i in allowed_indices if isinstance(i, int)}
        order = [i for i in order if i in allowed]
    return order


def _evaluate_best_local_move(scene: Dict[str, Any], role_graph: RoleGraph, priors: List[Dict[str, Any]], idx: int, bias_edit: Optional[ObjectEdit], current_score: float, cfg: Config, *, fixed_priors: bool, current_pbl: float, timing: Optional[TimingStats] = None, vlm_priors: Optional[List[Dict[str, Any]]] = None) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], float, float, float, Dict[str, Any], RoleGraph, List[Dict[str, Any]]]]:
    ranked = []
    for cand in _generate_candidates(scene, role_graph, idx, bias_edit, cfg):
        sc = _apply_delta(scene, idx, float(cand.get("dx", 0.0)), float(cand.get("dz", 0.0)), float(cand.get("dyaw", 0.0)), None if cand.get("yaw_abs") is None else float(cand.get("yaw_abs")))
        _repair_object_local(sc, idx, cfg.local_repair_passes)
        rg = infer_role_graph(sc)
        det = priors if fixed_priors else build_role_based_relation_priors(sc, rg)
        local_priors = _dedup_priors(list(det) + list(vlm_priors or [])) if (not fixed_priors and vlm_priors and cfg.merge_vlm_with_deterministic) else (list(vlm_priors) if (not fixed_priors and vlm_priors and not cfg.merge_vlm_with_deterministic) else list(det))
        ranked.append((_quick_candidate_proxy_score(sc, rg, local_priors, idx, cfg), cand, sc, rg, local_priors))
    ranked.sort(key=lambda x: x[0])
    best = None
    for _, cand, sc, rg, local_priors in ranked[: max(1, cfg.proxy_topk)]:
        score, metrics, rel, func = _score_scene_full(sc, rg, local_priors, timing, cfg)
        next_pbl = _get_float_metric(metrics, "total_pbl_loss")
        if cfg.candidate_filter_reintroduced_pbl and _is_valid_pbl_value(current_pbl, cfg) and not _is_valid_pbl_value(next_pbl, cfg):
            continue
        if score < current_score - cfg.monotonic_eps and (best is None or score < best[4]):
            best = (sc, metrics, rel, func, score, cand, rg, local_priors)
    return best


def _optimize_after_prompt(scene: Dict[str, Any], edits: List[ObjectEdit], cfg: Config, *, initial_priors: Optional[List[Dict[str, Any]]] = None, fixed_priors: bool = False, vlm_priors: Optional[List[Dict[str, Any]]] = None, timing: Optional[TimingStats] = None, allowed_indices: Optional[Sequence[int]] = None) -> Tuple[Dict[str, Any], Dict[str, Any], float, float, List[Dict[str, Any]], List[Dict[str, Any]], RoleGraph]:
    cur_scene = _deepcopy_scene(scene)
    role_graph = infer_role_graph(cur_scene)
    priors = list(initial_priors or []) if fixed_priors else _dedup_priors(list(build_role_based_relation_priors(cur_scene, role_graph)) + list(vlm_priors or []))
    cur_score, cur_metrics, cur_rel, cur_func = _score_scene_full(cur_scene, role_graph, priors, timing, cfg)
    actions: List[Dict[str, Any]] = []
    bias_by_idx = {idx: e for e in edits if (idx := _find_obj_idx_for_edit(cur_scene, e)) is not None}
    for round_id in range(max(1, cfg.max_rounds)):
        improved = False
        role_graph = infer_role_graph(cur_scene)
        order = _prioritized_object_indices(cur_scene, role_graph, edits, cur_metrics, cfg, allowed_indices)
        for idx in order[: max(1, cfg.max_objects_per_round)]:
            best = _evaluate_best_local_move(cur_scene, role_graph, priors, idx, bias_by_idx.get(idx), cur_score, cfg, fixed_priors=fixed_priors, current_pbl=_get_float_metric(cur_metrics, "total_pbl_loss"), timing=timing, vlm_priors=vlm_priors)
            if best is None:
                continue
            cur_scene, cur_metrics, cur_rel, cur_func, cur_score, cand, role_graph, priors = best
            actions.append({"round": round_id, "idx": idx, "candidate": cand, "score": round(cur_score, 6)})
            improved = True
        if not improved:
            break
    role_graph = infer_role_graph(cur_scene)
    priors = list(initial_priors or []) if fixed_priors else _dedup_priors(list(build_role_based_relation_priors(cur_scene, role_graph)) + list(vlm_priors or []))
    cur_pbl = _get_float_metric(cur_metrics, "total_pbl_loss")
    if allowed_indices:
        for idx in allowed_indices:
            _repair_object_local(cur_scene, int(idx), max(1, cfg.local_repair_passes))
    elif not (cfg.skip_post_refine_when_valid_pbl and _is_valid_pbl_value(cur_pbl, cfg)):
        post_refine_role_layout(cur_scene, role_graph, yaw_from_quaternion, quaternion_from_yaw, _repair_object_local, blend=cfg.role_refine_blend, full_repair_passes=cfg.full_repair_after_refine_passes)
    role_graph = infer_role_graph(cur_scene)
    priors = list(initial_priors or []) if fixed_priors else _dedup_priors(list(build_role_based_relation_priors(cur_scene, role_graph)) + list(vlm_priors or []))
    cur_score, cur_metrics, cur_rel, cur_func = _score_scene_full(cur_scene, role_graph, priors, timing, cfg)
    return cur_scene, cur_metrics, cur_rel, cur_func, actions, priors, role_graph


def _run_v15_optimizer_branches(*, scene_after_prompt: Dict[str, Any], edits: List[ObjectEdit], cfg: Config, frozen_deterministic_priors: List[Dict[str, Any]], relation_priors_cache: Optional[List[Dict[str, Any]]], relation_priors_source: str, current_priors: List[Dict[str, Any]], frozen_role_graph: RoleGraph, timing_stats: TimingStats, step_dir: Path, optimize_only_indices: Optional[Sequence[int]]) -> Dict[str, Any]:
    branches = []
    def run(label: str, *, fixed_priors: bool, init_priors: List[Dict[str, Any]], vlm_priors: Optional[List[Dict[str, Any]]]):
        scene2, metrics, rel, func, actions, priors, rg = _optimize_after_prompt(scene_after_prompt, edits, cfg, initial_priors=init_priors, fixed_priors=fixed_priors, vlm_priors=vlm_priors, timing=timing_stats, allowed_indices=optimize_only_indices)
        prop_score = _W_PBL * _get_float_metric(metrics, "total_pbl_loss") + _W_REL * rel + _W_FUNC * func
        judge_score, judge_metrics, judge_rel, judge_func = _score_scene_stable_judge(scene2, rg, frozen_role_graph, frozen_deterministic_priors, timing_stats, cfg)
        branches.append({
            "label": label, "scene": scene2, "metrics": metrics, "rel": rel, "func": func, "actions": actions, "priors": priors, "role_graph": rg,
            "proposal_score": prop_score, "judge_score": judge_score, "judge_metrics": judge_metrics, "judge_rel": judge_rel, "judge_func": judge_func,
        })
    if cfg.enable_stable_branch:
        run("stable", fixed_priors=True, init_priors=current_priors, vlm_priors=None)
    if cfg.enable_dynamic_det_branch:
        run("dynamic_det", fixed_priors=False, init_priors=current_priors, vlm_priors=None)
    if cfg.enable_dynamic_vlm_branch and relation_priors_cache:
        run("dynamic_vlm", fixed_priors=False, init_priors=current_priors, vlm_priors=relation_priors_cache)
    if not branches:
        run(relation_priors_source or "fallback", fixed_priors=False, init_priors=current_priors, vlm_priors=relation_priors_cache)
    _write_json(step_dir / "branch_summaries.json", [{k: v for k, v in b.items() if k not in {"scene", "metrics", "judge_metrics", "priors", "role_graph", "actions"}} for b in branches])
    branches.sort(key=lambda b: (float(b["judge_score"]), float(b["proposal_score"])))
    return branches[0]


def _final_polish_scene_v15(scene: Dict[str, Any], frozen_role_graph: RoleGraph, frozen_deterministic_priors: List[Dict[str, Any]], cfg: Config, timing: TimingStats) -> Dict[str, Any]:
    work = copy.deepcopy(cfg)
    work.max_rounds = 1
    work.max_objects_per_round = min(4, cfg.max_objects_per_round)
    work.proxy_topk = min(2, cfg.proxy_topk)
    work.step_xy = min(work.step_xy, 0.10)
    work.step_yaw = min(work.step_yaw, 5.0)
    cur = _deepcopy_scene(scene)
    for _ in range(max(1, cfg.final_polish_passes)):
        cur, _, _, _, _, _, _ = _optimize_after_prompt(cur, [], work, initial_priors=frozen_deterministic_priors, fixed_priors=True, timing=timing)
    return cur


# ------------------------------
# acceptance helpers
# ------------------------------

def _struct_vals(struct: Dict[str, Any]) -> Tuple[int, float, float, float, float, float]:
    return (
        int(float(struct.get("zone_count", 0.0))),
        float(struct.get("open_space_ratio", 0.0)),
        float(struct.get("max_zone_ratio", 0.0)),
        float(struct.get("spread_penalty", 0.0)),
        float(struct.get("flow_penalty", 0.0)),
        float(struct.get("structure_loss", 0.0)),
    )


def _valid_reject_reason(cfg: Config, pbl_after: float, before: Tuple[int, float, float, float, float, float], after: Tuple[int, float, float, float, float, float]) -> Optional[str]:
    z0, o0, m0, s0, f0, _ = before
    z1, o1, m1, s1, f1, _ = after
    if not _is_valid_pbl_value(pbl_after, cfg):
        return "reintroduced_pbl"
    if cfg.use_structural_guard and cfg.require_zone_count_preserve_after_valid_pbl and z0 >= 2 and z1 < z0:
        return "zone_count_drop"
    if cfg.use_structural_guard and o1 < o0 - cfg.max_open_space_drop_after_valid_pbl:
        return "open_space_drop"
    if cfg.use_structural_guard and m1 > m0 + cfg.max_monopoly_increase_after_valid_pbl:
        return "zone_monopoly_worse"
    if cfg.use_structural_guard and s1 > s0 + cfg.max_spread_increase_after_valid_pbl:
        return "spread_worse"
    if cfg.use_structural_guard and f1 > f0 + cfg.max_flow_increase_after_valid_pbl:
        return "flow_worse"
    return None


def _prevalid_struct_guard(cfg: Config, before: Tuple[int, float, float, float, float, float], after: Tuple[int, float, float, float, float, float]) -> bool:
    if not cfg.use_structural_guard:
        return True
    z0, o0, m0, *_ = before
    z1, o1, m1, *_ = after
    return z1 >= max(1, z0 - 1) and o1 >= o0 - max(0.08, cfg.max_open_space_drop_after_valid_pbl) and m1 <= max(m0 + 0.10, cfg.max_zone_monopoly_ratio + 0.05)


def _diagnosis(reason: str) -> str:
    if reason in {"judge_improvement_too_small", "judge_worse_too_much"}:
        return "Candidate may fit proposal priors but not the stable judge; next step should preserve anchor semantics and avoid macro drift."
    if str(reason).startswith("repeat_"):
        return "A near-duplicate candidate has already been rejected; next step must diversify the edit target or edit order."
    if reason in {"open_space_drop", "zone_monopoly_worse", "spread_worse", "flow_worse"}:
        return "The candidate harmed global structure; next step should preserve circulation and secondary zones."
    return ""


# ------------------------------
# step orchestration
# ------------------------------

def _maybe_refresh_relation_priors(step: int, step_dir: Path, scene: Dict[str, Any], role_graph: RoleGraph, diag_path: Path, top_path: Path, generator: GPTVLMovePromptGeneratorV5, base_priors: List[Dict[str, Any]], cfg: Config, timing: TimingStats, cache: Optional[List[Dict[str, Any]]], source: str, extra_context: str, reason: str = ""):
    raw = canon = None
    if not cfg.use_vlm_relation_priors:
        return list(base_priors), cache, "deterministic", raw, canon
    if _should_refresh_relation_priors(step, cache, 1 if reason else 0, cfg):
        t0 = _now()
        out_dir = step_dir / reason if reason else step_dir
        built = _safe_generate_relation_priors(generator, diag_path, top_path, scene, extra_context, cfg.relation_prior_retries, cfg.relation_prior_temperature, cfg.relation_prior_max_tokens, cfg.relation_prior_confidence, out_dir)
        timing.vlm_sec += _now() - t0
        if built:
            raw = list(built)
            canon, drop_log = _canonicalize_relation_priors(scene, role_graph, raw, cfg)
            cache = canon if str(cfg.relation_use_mode).lower().startswith("canonical") else raw
            source = f"vlm_relation_priors{('_' + reason) if reason else ''}"
            if cfg.relation_debug_write_raw:
                _write_json(out_dir / "relation_priors_raw.json", raw)
            if cfg.relation_debug_write_canonical:
                _write_json(out_dir / "relation_priors_canonical.json", canon)
                _write_json(out_dir / "relation_canonicalization_log.json", drop_log)
        else:
            cache, source = None, "deterministic_fallback"
    if raw is None and cache is not None:
        if str(cfg.relation_use_mode).lower().startswith("canonical"):
            canon = list(cache)
        else:
            raw = list(cache)
    priors = _select_relation_priors_for_optimizer(base_priors, raw or (None if str(cfg.relation_use_mode).lower().startswith("canonical") else cache), canon or (cache if str(cfg.relation_use_mode).lower().startswith("canonical") else None), cfg)
    return priors, cache, source, raw, canon


def _run_cleanup_or_prompt(*, step: int, step_dir: Path, current_scene: Dict[str, Any], metrics_before: Dict[str, Any], rel_before: float, func_before: float, struct_before: Dict[str, Any], current_priors: List[Dict[str, Any]], optimize_only_indices: Sequence[int], valid_pbl_before: bool, force_history_replan: bool, cfg: Config, generator: GPTVLMovePromptGeneratorV5, diag_path: Path, top_path: Path, extra_context: str, relation_priors_cache, relation_priors_source: str, frozen_deterministic_priors, frozen_role_graph, timing: TimingStats):
    if valid_pbl_before and cfg.cleanup_only_after_valid_pbl and not force_history_replan:
        _write_text(step_dir / "move_prompt.txt", "[cleanup-only-after-valid-pbl]")
        _write_json(step_dir / "move_prompt_parse.json", {"room_name": "", "header_line": "", "parse_warnings": ["cleanup_only_after_valid_pbl"], "num_edits": 0, "edits": []})
        _write_json(step_dir / "applied_edits.json", {"applied_count": 0, "changes": [], "parser_mode": "cleanup_only_after_valid_pbl"})
        _write_json(step_dir / "scene_repair_plan.json", {"schema": _V15_SCHEMA, "actions": [], "reason": "cleanup_only_after_valid_pbl"})
        work = copy.deepcopy(cfg)
        work.max_rounds = min(cfg.max_rounds_after_valid_pbl, cfg.max_rounds)
        work.max_objects_per_round = min(cfg.max_objects_after_valid_pbl, cfg.max_objects_per_round)
        work.step_xy, work.step_yaw, work.proxy_topk = min(cfg.step_xy, 0.10), min(cfg.step_yaw, 5.0), 3
        return 0.0, 0.0, 0, _run_v15_optimizer_branches(scene_after_prompt=_deepcopy_scene(current_scene), edits=[], cfg=work, frozen_deterministic_priors=frozen_deterministic_priors, relation_priors_cache=relation_priors_cache if cfg.use_vlm_relation_priors and relation_priors_source.startswith("vlm") else None, relation_priors_source=relation_priors_source, current_priors=current_priors, frozen_role_graph=frozen_role_graph, timing_stats=timing, step_dir=step_dir, optimize_only_indices=optimize_only_indices)

    t0 = _now()
    ctx = _compose_v13_structured_repair_context(extra_context, current_scene, metrics_before, rel_before, func_before, struct_before, current_priors, cfg, allowed_target_indices=optimize_only_indices) if cfg.use_structured_repair_plan else extra_context
    _write_text(step_dir / "scene_repair_plan_prompt_context.txt", ctx)
    result = _safe_generate_move_prompt(generator, diag_path, top_path, current_scene, ctx, cfg.move_prompt_retries, cfg.move_prompt_temperature, cfg.move_prompt_max_tokens, step_dir)
    prompt_sec = _now() - t0
    timing.vlm_sec += prompt_sec
    raw = _extract_text_from_prompt_result(result)
    _write_text(step_dir / "move_prompt.txt", raw)
    _write_text(step_dir / "scene_repair_plan_raw.txt", raw)
    _log(f"[step {step:02d}] repair plan generated ({len(raw)} chars), temp={cfg.move_prompt_temperature}")

    t1 = _now()
    parse_result, edits, parser_mode, plan, warnings = _parse_move_prompt_v13_or_legacy(raw, current_scene, cfg, step_dir)
    plan, edits, drop_log = _filter_structured_plan_and_edits_by_allowed_indices(current_scene, plan, edits, optimize_only_indices)
    if drop_log:
        _write_json(step_dir / "partial_round_target_filter.json", {"optimize_only_indices": list(optimize_only_indices), "dropped": drop_log})
        warnings = list(warnings) + [f"filtered_targets_outside_group:{len(drop_log)}"]
    for e in edits:
        try:
            if hasattr(e, "dx"):
                e.dx = _safe_float(getattr(e, "dx", None), 0.0)
            if hasattr(e, "dz"):
                e.dz = _safe_float(getattr(e, "dz", None), 0.0)
            if hasattr(e, "relative_yaw_deg") and getattr(e, "relative_yaw_deg", None) is not None:
                e.relative_yaw_deg = _safe_float(getattr(e, "relative_yaw_deg", None), 0.0)
            if hasattr(e, "target_yaw_deg") and getattr(e, "target_yaw_deg", None) is not None:
                e.target_yaw_deg = _normalize_angle(_safe_float(getattr(e, "target_yaw_deg", None), 0.0))
            if hasattr(e, "hint_pos"):
                e.hint_pos = _normalize_hint_pos(getattr(e, "hint_pos", None))
        except Exception:
            pass
    parse_sec = _now() - t1
    _write_json(step_dir / "move_prompt_parse.json", {"parser_mode": parser_mode, "room_name": getattr(parse_result, "room_name", ""), "header_line": getattr(parse_result, "header_line", ""), "parse_warnings": warnings, "num_edits": len(edits), "edits": [getattr(e, "__dict__", {}) for e in edits], "structured_plan": plan})

    scene_after_prompt = _deepcopy_scene(current_scene)
    if cfg.structured_plan_apply_before_search:
        try:
            scene_after_prompt, applied_count, changes = apply_edits_to_scene(scene_after_prompt, edits)
        except Exception:
            applied_count, changes = 0, []
            _write_text(step_dir / "applied_edits.error.txt", traceback.format_exc())
    else:
        applied_count, changes = 0, []
    _write_json(step_dir / "applied_edits.json", {"applied_count": applied_count, "changes": changes, "parser_mode": parser_mode})
    best = _run_v15_optimizer_branches(scene_after_prompt=scene_after_prompt, edits=edits, cfg=cfg, frozen_deterministic_priors=frozen_deterministic_priors, relation_priors_cache=relation_priors_cache if cfg.use_vlm_relation_priors and relation_priors_source.startswith("vlm") else None, relation_priors_source=relation_priors_source, current_priors=current_priors, frozen_role_graph=frozen_role_graph, timing_stats=timing, step_dir=step_dir, optimize_only_indices=optimize_only_indices)
    return prompt_sec, parse_sec, applied_count, best


# ------------------------------
# public optimizer
# ------------------------------

def optimize_scene_refactored_v15(*, scene: Dict[str, Any], out_root: Path, respace: ReSpace, generator: GPTVLMovePromptGeneratorV5, extra_hints_text: str, cfg: Config, zero_shot_relation_plan: Optional[Dict[str, Any]] = None, optimize_only_indices: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    timing, overall_t0 = TimingStats(), _now()
    current_scene, history, step_records = _deepcopy_scene(scene), [], []
    rejection_memory, stagnation_count = {}, 0
    frozen_role_graph = infer_role_graph(current_scene)
    frozen_det_priors = build_role_based_relation_priors(current_scene, frozen_role_graph)
    relation_priors_cache, relation_priors_source = None, "deterministic"
    optimize_only_indices = sorted({int(i) for i in (optimize_only_indices or []) if isinstance(i, int)})
    _write_json(out_root / "optimize_scope.json", {"optimize_only_indices": optimize_only_indices, "optimize_only_count": len(optimize_only_indices)})

    zero_raw, zero_drop = _canonicalize_zero_shot_relation_plan(zero_shot_relation_plan, cfg)
    if cfg.use_zero_shot_relation_plan and zero_shot_relation_plan is not None:
        _write_json(out_root / "zero_shot_relation_plan_raw.json", zero_shot_relation_plan)
        _write_json(out_root / "zero_shot_relation_plan_canonical.json", zero_raw)
        _write_json(out_root / "zero_shot_relation_plan_drop_log.json", zero_drop)

    initial_metrics = initial_rel = initial_func = initial_score = initial_judge = None
    for step in range(cfg.max_steps):
        step_t0, step_dir = _now(), out_root / f"step_{step:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        _write_json(step_dir / "scene_before.json", current_scene)

        t0 = _now()
        respace.render_scene_frame(current_scene, filename=f"step_{step:02d}", pth_viz_output=step_dir)
        diag_path = step_dir / "diag" / f"step_{step:02d}.jpg"
        top_path = render_annotated_top_view(current_scene, f"step_{step:02d}", step_dir, resolution=(1024, 1024), show_assets=True, font_size=14)
        render_sec = _now() - t0
        timing.render_sec += render_sec

        role_graph = infer_role_graph(current_scene)
        det_priors = list(frozen_det_priors) if cfg.freeze_deterministic_priors else build_role_based_relation_priors(current_scene, role_graph)
        grounded_zero, grounded_drop = _ground_zero_shot_relation_plan(current_scene, role_graph, zero_raw, cfg) if zero_raw else ([], [])
        if grounded_drop:
            _write_json(step_dir / "zero_shot_relation_grounding_drop_log.json", grounded_drop)
        base_priors = _select_zero_shot_base_priors(det_priors, grounded_zero, cfg)

        planner_mode = "history_replan" if (cfg.use_iteration_history and stagnation_count >= 1) else "normal"
        extra = _compose_step_extra_context(extra_hints_text, history, planner_mode, cfg.max_history_steps)
        current_priors, relation_priors_cache, relation_priors_source, _, _ = _maybe_refresh_relation_priors(step, step_dir, current_scene, role_graph, diag_path, top_path, generator, base_priors, cfg, timing, relation_priors_cache, relation_priors_source, extra)

        proposal_before, metrics_before, rel_before, func_before = _score_scene_full(current_scene, role_graph, current_priors, timing, cfg)
        judge_before, judge_metrics_before, judge_rel_before, judge_func_before = _score_scene_stable_judge(current_scene, role_graph, frozen_role_graph, frozen_det_priors, timing, cfg)
        pbl_before, struct_before = _get_float_metric(metrics_before, "total_pbl_loss"), metrics_before.get("structure_stats", {})
        valid_before = _is_valid_pbl_value(pbl_before, cfg)
        mono_before = float(struct_before.get("max_zone_ratio", 0.0))
        force_history_replan = cfg.use_iteration_history and (stagnation_count >= 1 or (cfg.force_history_replan_on_monopoly and mono_before > cfg.max_zone_monopoly_ratio))
        planner_mode = "history_replan" if force_history_replan else ("cleanup_only" if (valid_before and cfg.cleanup_only_after_valid_pbl) else "normal")
        extra = _compose_step_extra_context(extra_hints_text, history, planner_mode, cfg.max_history_steps)
        if force_history_replan and cfg.use_vlm_relation_priors and cfg.refresh_vlm_relation_priors_on_stagnation:
            current_priors, relation_priors_cache, relation_priors_source, _, _ = _maybe_refresh_relation_priors(step, step_dir, current_scene, role_graph, diag_path, top_path, generator, base_priors, cfg, timing, relation_priors_cache, relation_priors_source, extra, "history_replan_refresh")
            proposal_before, metrics_before, rel_before, func_before = _score_scene_full(current_scene, role_graph, current_priors, timing, cfg)
            judge_before, judge_metrics_before, judge_rel_before, judge_func_before = _score_scene_stable_judge(current_scene, role_graph, frozen_role_graph, frozen_det_priors, timing, cfg)
            pbl_before, struct_before, valid_before = _get_float_metric(metrics_before, "total_pbl_loss"), metrics_before.get("structure_stats", {}), _is_valid_pbl_value(_get_float_metric(metrics_before, "total_pbl_loss"), cfg)

        if step == 0 and initial_score is None:
            initial_metrics, initial_rel, initial_func, initial_score, initial_judge = copy.deepcopy(metrics_before), float(rel_before), float(func_before), float(proposal_before), float(judge_before)

        _write_json(step_dir / "role_graph_before.json", _role_graph_dict(role_graph))
        _write_json(step_dir / "relation_priors.json", current_priors)
        _write_json(step_dir / "relation_priors_meta.json", {"source": relation_priors_source, "count": len(current_priors)})
        _log(f"[step {step:02d}] relation priors source={relation_priors_source} count={len(current_priors)}")
        _log(f"[step {step:02d}] pbl={pbl_before:.6f} oob={_get_float_metric(metrics_before, 'total_oob_loss'):.6f} mbl={_get_float_metric(metrics_before, 'total_mbl_loss'):.6f} rel={rel_before:.4f} func={func_before:.4f} struct={float(struct_before.get('structure_loss', 0.0)):.4f} open={float(struct_before.get('open_space_ratio', 0.0)):.3f} zones={int(float(struct_before.get('zone_count', 0.0)))} mono={float(struct_before.get('max_zone_ratio', 0.0)):.3f} proposal_score={proposal_before:.6f} judge_score={judge_before:.6f} valid_pbl={valid_before}")
        if step > 0 and cfg.stop_when_valid_pbl and valid_before and judge_before <= cfg.stop_score_threshold and not cfg.mandatory_final_polish:
            _log(f"[step {step:02d}] early stop: judge score already good enough (judge_score={judge_before:.6f} <= {cfg.stop_score_threshold:.6f})")
            break

        t0 = _now()
        prompt_sec, parse_sec, applied_count, best = _run_cleanup_or_prompt(step=step, step_dir=step_dir, current_scene=current_scene, metrics_before=metrics_before, rel_before=rel_before, func_before=func_before, struct_before=struct_before, current_priors=current_priors, optimize_only_indices=optimize_only_indices, valid_pbl_before=valid_before, force_history_replan=force_history_replan, cfg=cfg, generator=generator, diag_path=diag_path, top_path=top_path, extra_context=extra, relation_priors_cache=relation_priors_cache, relation_priors_source=relation_priors_source, frozen_deterministic_priors=frozen_det_priors, frozen_role_graph=frozen_role_graph, timing=timing)
        opt_sec = _now() - t0
        timing.optimize_sec += opt_sec

        optimized_scene, metrics_after, rel_after, func_after = best["scene"], best["metrics"], float(best["rel"]), float(best["func"])
        actions, priors_after, role_graph_after = best["actions"], best["priors"], best["role_graph"]
        proposal_after, judge_after = float(best["proposal_score"]), float(best["judge_score"])
        judge_metrics_after, judge_rel_after, judge_func_after = best["judge_metrics"], float(best["judge_rel"]), float(best["judge_func"])
        selected_branch, struct_after = str(best["label"]), metrics_after.get("structure_stats", {})
        _write_json(step_dir / "optimizer_actions.json", actions)
        _write_json(step_dir / "scene_after.json", optimized_scene)
        _write_json(step_dir / "role_graph_after.json", _role_graph_dict(role_graph_after))
        _write_json(step_dir / "relation_priors_after.json", priors_after)
        _write_json(step_dir / "relation_priors_after_meta.json", {"source": selected_branch, "count": len(priors_after)})
        _write_json(step_dir / "judge_after.json", {"judge_score_after": round(judge_after, 6), "judge_rel_after": round(judge_rel_after, 6), "judge_func_after": round(judge_func_after, 6), "judge_stats": judge_metrics_after.get("judge_stats") if isinstance(judge_metrics_after, dict) else None})
        total_sec = _now() - step_t0
        _log(f"[step {step:02d}] time total={total_sec:.2f}s (prompt={prompt_sec:.2f}s, parse={parse_sec:.2f}s, opt={opt_sec:.2f}s) applied={applied_count} branch={selected_branch} pbl {pbl_before:.6f}->{_get_float_metric(metrics_after, 'total_pbl_loss'):.6f} rel {rel_before:.4f}->{rel_after:.4f} func {func_before:.4f}->{func_after:.4f} struct {float(struct_before.get('structure_loss', 0.0)):.4f}->{float(struct_after.get('structure_loss', 0.0)):.4f} proposal {proposal_before:.6f}->{proposal_after:.6f} judge {judge_before:.6f}->{judge_after:.6f}")

        pbl_after = _get_float_metric(metrics_after, "total_pbl_loss")
        before_vals, after_vals = _struct_vals(struct_before), _struct_vals(struct_after)
        accepted, reject_reason = False, "no_improvement"
        if valid_before:
            reject_reason = _valid_reject_reason(cfg, pbl_after, before_vals, after_vals)
            if reject_reason is None and judge_after < judge_before - cfg.judge_min_score_improve_after_valid_pbl:
                accepted = True
            elif reject_reason is None and proposal_after < proposal_before - cfg.min_score_improve_after_valid_pbl and judge_after <= judge_before + cfg.judge_max_score_increase_after_valid_pbl:
                accepted = True
            elif reject_reason is None:
                reject_reason = "judge_improvement_too_small"
        else:
            pbl_improved = pbl_after < pbl_before - cfg.monotonic_eps
            judge_guard = judge_after <= judge_before + cfg.judge_max_score_increase_prevalid
            struct_guard = _prevalid_struct_guard(cfg, before_vals, after_vals)
            accepted = (pbl_improved and judge_guard and struct_guard) or (proposal_after < proposal_before - cfg.monotonic_eps and pbl_after <= pbl_before + 0.005 and judge_guard and struct_guard)
            if not accepted:
                reject_reason = "pbl_not_better" if not pbl_improved else ("structure_worse_too_much" if not struct_guard else "judge_worse_too_much")

        before_hash, after_hash = _scene_state_hash(current_scene), _scene_state_hash(optimized_scene)
        history.append({"step": step, "planner_mode": planner_mode, "selected_branch": selected_branch, "accepted": accepted, "reject_reason": None if accepted else reject_reason, "score_before": float(judge_before), "score_after": float(judge_after), "proposal_score_before": float(proposal_before), "proposal_score_after": float(proposal_after), "rel_before": float(judge_rel_before), "rel_after": float(judge_rel_after), "func_before": float(judge_func_before), "func_after": float(judge_func_after), "struct_before": before_vals[-1], "struct_after": after_vals[-1], "zone_before": before_vals[0], "zone_after": after_vals[0], "mono_before": before_vals[2], "mono_after": after_vals[2], "diagnosis": "" if accepted else _diagnosis(reject_reason), "before_hash": before_hash, "after_hash": after_hash})
        history = history[-max(2, cfg.max_history_steps * 2):]

        if accepted:
            current_scene, stagnation_count = optimized_scene, 0
            if cfg.use_vlm_relation_priors and str(cfg.relation_refresh_mode).lower() in {"every_step", "on_stagnation"}:
                relation_priors_cache = None
            _log(f"[step {step:02d}] accepted")
        else:
            key = (before_hash, after_hash, str(reject_reason))
            rejection_memory[key] = rejection_memory.get(key, 0) + 1
            if rejection_memory[key] >= cfg.repeat_reject_patience:
                history[-1]["reject_reason"] = reject_reason = f"repeat_{reject_reason}"
                history[-1]["diagnosis"] = "Repeated rejected proposal detected. Force VLM history-guided replanning and refresh relation priors next step."
                stagnation_count += 1
                if cfg.refresh_vlm_relation_priors_on_stagnation:
                    relation_priors_cache = None
            else:
                stagnation_count = max(1, stagnation_count)
            _log(f"[step {step:02d}] rejected: {reject_reason}")

        step_records.append({"step": step, "runtime_sec": round(total_sec, 4), "accepted": accepted, "relation_priors_source": relation_priors_source, "selected_branch": selected_branch, "num_relation_priors": len(priors_after), "render_sec": round(render_sec, 4), "prompt_sec": round(prompt_sec, 4), "opt_sec": round(opt_sec, 4), "eval_sec_accum": round(timing.eval_sec, 4), "open_space_ratio_after": round(after_vals[1], 4), "zone_count_after": after_vals[0], "max_zone_ratio_after": round(after_vals[2], 4), "structure_loss_after": round(after_vals[-1], 4), "proposal_score_after": round(proposal_after, 6), "judge_score_after": round(judge_after, 6), "planner_mode": planner_mode, "reject_reason": None if accepted else reject_reason, "stagnation_count": stagnation_count})
        if cfg.stop_when_valid_pbl and accepted and _is_valid_pbl_value(pbl_after, cfg) and judge_after <= cfg.stop_score_threshold and not cfg.mandatory_final_polish:
            _log(f"[step {step:02d}] early stop: accepted judge score is good enough (judge_score={judge_after:.6f} <= {cfg.stop_score_threshold:.6f})")
            break

    if cfg.mandatory_final_polish:
        polish_dir = out_root / "final_polish"
        polish_dir.mkdir(parents=True, exist_ok=True)
        rg = infer_role_graph(current_scene)
        pre_prop, pre_metrics, _, _ = _score_scene_full(current_scene, rg, build_role_based_relation_priors(current_scene, rg), timing, cfg)
        pre_judge, _, _, _ = _score_scene_stable_judge(current_scene, rg, frozen_role_graph, frozen_det_priors, timing, cfg)
        polished = _final_polish_scene_v15(current_scene, frozen_role_graph, frozen_det_priors, cfg, timing)
        rg2 = infer_role_graph(polished)
        post_prop, post_metrics, _, _ = _score_scene_full(polished, rg2, build_role_based_relation_priors(polished, rg2), timing, cfg)
        post_judge, judge_metrics, _, _ = _score_scene_stable_judge(polished, rg2, frozen_role_graph, frozen_det_priors, timing, cfg)
        _write_json(polish_dir / "pre_polish_metrics.json", pre_metrics)
        _write_json(polish_dir / "post_polish_metrics.json", post_metrics)
        _write_json(polish_dir / "post_polish_judge.json", judge_metrics)
        _write_json(polish_dir / "scene_after_polish.json", polished)
        if post_judge <= pre_judge + 1e-9 and _get_float_metric(post_metrics, "total_pbl_loss") <= _get_float_metric(pre_metrics, "total_pbl_loss") + 1e-9:
            current_scene = polished
            _log(f"[final polish] accepted proposal {pre_prop:.6f}->{post_prop:.6f} judge {pre_judge:.6f}->{post_judge:.6f}")
        else:
            _log(f"[final polish] rejected proposal {pre_prop:.6f}->{post_prop:.6f} judge {pre_judge:.6f}->{post_judge:.6f}")

    if cfg.render_final:
        final_dir = out_root / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        t0 = _now()
        respace.render_scene_frame(current_scene, filename="final", pth_viz_output=final_dir)
        render_annotated_top_view(current_scene, "final", final_dir, resolution=(1024, 1024), show_assets=True, font_size=14)
        timing.render_sec += _now() - t0
        _write_json(final_dir / "scene.json", current_scene)

    final_rg = infer_role_graph(current_scene)
    final_priors = build_role_based_relation_priors(current_scene, final_rg)
    final_prop, final_metrics, final_rel, final_func = _score_scene_full(current_scene, final_rg, final_priors, timing, cfg)
    final_judge, final_judge_metrics, final_judge_rel, final_judge_func = _score_scene_stable_judge(current_scene, final_rg, frozen_role_graph, frozen_det_priors, timing, cfg)
    summary = {
        "relation_priors_source": relation_priors_source,
        "judge_mode": "stable_dynamic_blend" if cfg.enable_dual_judge else "stable_only",
        "optimize_only_indices": optimize_only_indices,
        "num_relation_priors": len(final_priors),
        "total_runtime_sec": round(_now() - overall_t0, 4),
        "render_sec": round(timing.render_sec, 4),
        "vlm_sec": round(timing.vlm_sec, 4),
        "optimize_sec": round(timing.optimize_sec, 4),
        "eval_sec": round(timing.eval_sec, 4),
        "step_runtime_records": step_records,
        "history_records": history,
        "initial_metrics": initial_metrics,
        "initial_rel_loss": None if initial_rel is None else round(initial_rel, 4),
        "initial_func_loss": None if initial_func is None else round(initial_func, 4),
        "initial_structure_stats": None if not isinstance(initial_metrics, dict) else initial_metrics.get("structure_stats"),
        "initial_score": None if initial_score is None else round(initial_score, 6),
        "initial_judge_score": None if initial_judge is None else round(initial_judge, 6),
        "final_metrics": final_metrics,
        "final_judge_metrics": final_judge_metrics,
        "final_rel_loss": round(final_rel, 4),
        "final_func_loss": round(final_func, 4),
        "final_judge_rel_loss": round(final_judge_rel, 4),
        "final_judge_func_loss": round(final_judge_func, 4),
        "final_structure_stats": final_metrics.get("structure_stats") if isinstance(final_metrics, dict) else None,
        "final_score": round(final_prop, 6),
        "final_judge_score": round(final_judge, 6),
    }
    _write_json(out_root / "summary.json", summary)
    return summary


optimize_scene_refactored_v13 = optimize_scene_refactored_v15
optimize_scene_refactored_v12 = optimize_scene_refactored_v15


# ------------------------------
# CLI
# ------------------------------


def main() -> None:
    if os.getenv("YUNWU_AI_API_BASE") is None and os.getenv("YUNWU_AI_BASE_URL") is not None:
        os.environ["YUNWU_AI_API_BASE"] = os.environ["YUNWU_AI_BASE_URL"]

    use_local_vlm_optimizer = _env_bool(os.getenv("USE_LOCAL_VLM_OPTIMIZER", "0"))
    if (not use_local_vlm_optimizer) and (not os.getenv("YUNWU_AI_API_KEY")):
        raise RuntimeError("Missing YUNWU_AI_API_KEY env var.")

    summary: Dict[str, Any] = {
        "status": "started",
        "planning_rag_enabled": _env_bool(os.getenv("ENABLE_PLANNING_RAG", "0")),
        "vlm_optimization_enabled": _env_bool(os.getenv("ENABLE_VLM_OPTIMIZATION", "0")),
        "use_group_repair_in_loop": _env_bool(os.getenv("USE_GROUP_REPAIR_IN_LOOP", "1")),
        "use_local_vlm_optimizer": use_local_vlm_optimizer,
    }

    out_root: Optional[Path] = None

    try:
        attach_group_repair_in_loop(ReSpace, Config, optimize_scene_refactored_v15, GPTVLMovePromptGeneratorV5)
        respace = ReSpace()

        scene_json_path = Path(os.getenv("SCENE_JSON_PATH", "")).expanduser()
        if not scene_json_path.exists():
            raise FileNotFoundError(f"scene json not found: {scene_json_path}")

        scene = json.loads(scene_json_path.read_text(encoding="utf-8"))
        room_prompt = os.getenv("ROOM_PROMPT", "").strip()

        out_root = Path(os.getenv("OUT_DIR", "./evaluate/infer_v15")).expanduser()
        out_root.mkdir(parents=True, exist_ok=True)

        cfg = _config_from_env()

        extra_hints_text = (
            "GLOBAL SAFETY CONSTRAINTS:\n"
            "1) All objects must stay fully inside the room.\n"
            "2) Avoid overlaps; keep small but visible clearance.\n"
            "3) Prioritize OOB and collision fixes before aesthetics.\n"
            "4) Preserve dominant-anchor and accessory structure.\n"
            "5) Keep interactive fronts usable and avoid over-crowding one functional anchor.\n"
            "6) Preserve secondary functional zones; do not collapse all seating into one dominant cluster.\n"
            "7) Maintain visible open space and a clear main circulation band through the room.\n"
            "8) Use the step history to avoid repeating rejected edits; propose a different diagnose-and-act plan when the last attempt failed.\n"
        )

        use_group = _env_bool(os.getenv("USE_GROUP_REPAIR_IN_LOOP", "1"))
        include_relation_plan = _env_bool(os.getenv("INCLUDE_RELATION_PLAN", "0"))

        if use_group:
            partial_cfg = copy.deepcopy(cfg)
            for k, v in {
                "max_steps": int(os.getenv("GROUP_PARTIAL_MAX_STEPS", "1")),
                "max_rounds": int(os.getenv("GROUP_PARTIAL_MAX_ROUNDS", "1")),
                "max_objects_per_round": int(os.getenv("GROUP_PARTIAL_MAX_OBJECTS_PER_ROUND", "6")),
                "proxy_topk": int(os.getenv("GROUP_PARTIAL_PROXY_TOPK", "2")),
                "move_prompt_max_tokens": int(os.getenv("GROUP_PARTIAL_MOVE_PROMPT_MAX_TOKENS", str(cfg.move_prompt_max_tokens))),
                "mandatory_final_polish": False,
            }.items():
                setattr(partial_cfg, k, v)

            updated_scene, is_success, aux = respace.handle_prompt_group_repair_in_loop(
                room_prompt,
                scene,
                return_aux=True,
                include_relation_plan=include_relation_plan,
                pth_viz_output=out_root,
                partial_repair_cfg=partial_cfg,
                final_repair_cfg=cfg,
                run_final_global_repair=_env_bool(os.getenv("RUN_FINAL_GLOBAL_REPAIR", "1")),
                max_groups=int(os.getenv("MAX_FUNCTIONAL_GROUPS", "4")),
            )
            _ = is_success

            _write_json(out_root / "group_generation_aux.json", aux)
            _write_json(out_root / "final_scene_from_group_repair.json", updated_scene)

            if aux.get("group_plan") is not None:
                _write_json(out_root / "group_plan.json", aux.get("group_plan"))

            if include_relation_plan:
                plan = aux.get("global_relation_plan") or {"relation_plan": []}
                _write_json(out_root / "zero_shot_relation_plan.json", plan)
                _log("ZERO-SHOT RELATION PLAN:")
                _log(json.dumps(plan, ensure_ascii=False, indent=2))

            summary = aux.get("final_repair_summary") or {
                "mode": "group_repair_in_loop",
                "message": "group-wise generation finished without final optimizer summary",
                "num_groups": len((aux.get("group_plan") or {}).get("groups", [])),
                "num_objects": len(updated_scene.get("objects", [])),
                "planning_rag_enabled": _env_bool(os.getenv("ENABLE_PLANNING_RAG", "0")),
                "vlm_optimization_enabled": _env_bool(os.getenv("ENABLE_VLM_OPTIMIZATION", "0")),
            }

        else:
            if include_relation_plan:
                updated_scene, _, aux = respace.handle_prompt(
                    room_prompt,
                    scene,
                    return_aux=True,
                    include_relation_plan=True,
                )
            else:
                updated_scene, _ = respace.handle_prompt(room_prompt, scene)
                aux = {}

            zero_plan = aux.get("relation_plan", {}) if include_relation_plan else None

            if include_relation_plan:
                _write_json(out_root / "zero_shot_command_response.json", aux.get("raw_response", {}))
                _write_json(out_root / "zero_shot_relation_plan.json", zero_plan or {"relation_plan": []})
                _log("ZERO-SHOT RELATION PLAN:")
                _log(json.dumps(zero_plan, ensure_ascii=False, indent=2))

            enable_vlm_optimization = _env_bool(os.getenv("ENABLE_VLM_OPTIMIZATION", "0"))
            enable_planning_rag = _env_bool(os.getenv("ENABLE_PLANNING_RAG", "0"))

            if enable_vlm_optimization:
                generator = GPTVLMovePromptGeneratorV5(
                    model=os.getenv("MOVE_PROMPT_MODEL", os.getenv("YUNWU_AI_MODEL", "gpt-4o")),
                    api_base=os.getenv("YUNWU_AI_API_BASE"),
                    api_key=os.getenv("YUNWU_AI_API_KEY"),
                    timeout_s=float(os.getenv("MOVE_PROMPT_TIMEOUT_S", "120")),
                )
                summary = optimize_scene_refactored_v15(
                    scene=updated_scene,
                    out_root=out_root,
                    respace=respace,
                    generator=generator,
                    extra_hints_text=extra_hints_text,
                    cfg=cfg,
                    zero_shot_relation_plan=zero_plan,
                )
                summary["planning_rag_enabled"] = enable_planning_rag
                summary["vlm_optimization_enabled"] = True
                summary["use_local_vlm_optimizer"] = use_local_vlm_optimizer
                _write_json(out_root / "summary.json", summary)
            else:
                summary = _summarize_scene_without_vlm_optimization(
                    scene=updated_scene,
                    out_root=out_root,
                    respace=respace,
                    cfg=cfg,
                    zero_shot_relation_plan=zero_plan,
                    planning_rag_bundle=None,
                )

    except Exception as e:
        fail_summary = dict(summary)
        fail_summary.update({
            "status": "failed",
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })

        if out_root is not None:
            try:
                _write_json(out_root / "summary_failed.json", fail_summary)
            except Exception:
                pass

        _log("[infer_v15] failed with original exception:")
        _log(traceback.format_exc())
        raise

    finally:
        _log("\n=== Done ===")
        try:
            _log(json.dumps(summary, ensure_ascii=False, indent=2))
        except Exception as log_exc:
            _log(f"[WARN] failed to print summary: {log_exc!r}")

if __name__ == "__main__":
    main()
