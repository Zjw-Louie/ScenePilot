#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Clean ablation runner for ReSpace
#
# Presets:
#   vanilla_respace : vanilla ReSpace baseline (no RAG / no group / no VLM opt)
#   vlm_only        : vanilla + VLM optimization only (no RAG / no group)
#   rag_only        : vanilla + RAG
#   rag_group       : vanilla + RAG + group repair
#   rag_group_vlm   : vanilla + RAG + group repair + VLM optimization
#
# Notes:
# - vanilla_respace uses the plain ReSpace batch loop (same logic as your uploaded
#   baseline Python: load ReSpace -> handle_prompt(prompt, scene)).
# - other presets use infer_with_planning_rag_ablation.py in single-process mode.
# ============================================================

# -----------------------------
# User config
# -----------------------------
CONDA_ENV_NAME="respace"
PROJECT_ROOT="/home2/zhangjiawei/respace"
PYTHON_BIN="python"
CUDA_DEVICES="1,2,3"

# Enhanced inference entry (used by vlm_only / rag_only / rag_group / rag_group_vlm)
PY_ENHANCED_SCRIPT="/home2/zhangjiawei/respace/infer_v15.py"  # replace this path if your sanitized infer has another filename

# Input benchmark / empty scenes
IN_JSONL="/home2/zhangjiawei/respace/benchmark/sample_benchmark_by_ratio_vlm.jsonl"
EMPTY_ROOT="/home2/zhangjiawei/respace/benchmark/empty_scenes"

# Experiment preset
#   vanilla_respace | vlm_only | rag_only | rag_group | rag_group_vlm
EXPERIMENT_PRESET="without_vlm"
RUN_NAME="123"
RESULTS_ROOT="/home2/zhangjiawei/respace/results"
OUT_ROOT="${RESULTS_ROOT}/ablation_${EXPERIMENT_PRESET}_${RUN_NAME}"

LOCAL_VLM_MODEL="/home2/zhangjiawei/respace/model/qwen3-sft+grpo"
USE_LOCAL_VLM_OPTIMIZER="1"
LOCAL_VLM_DEVICE="cuda"
LOCAL_VLM_DTYPE="bfloat16"

# API / model (only needed by non-vanilla presets)
YUNWU_AI_API_BASE="https://yunwu.ai"
YUNWU_AI_API_KEY="your-key"
MOVE_PROMPT_MODEL="$LOCAL_VLM_MODEL"
YUNWU_AI_MODEL="$LOCAL_VLM_MODEL"
# MOVE_PROMPT_MODEL="gpt-5.2"
# YUNWU_AI_MODEL="gpt-5.2"

# Batch controls
LIMIT=0                 # 0 means all
RESUME=1                # 1: skip scenes with existing success marker
RETRY_FAILED=0          # 1: if resume, rerun failed scenes
SHUFFLE=0               # 1: shuffle records before running

# Vanilla baseline rendering controls
BASELINE_RENDER_FRAME=0 # 1 to render a single RGB frame
BASELINE_RENDER_TOP=1   # 1 to render annotated top view

# Planning RAG (used when preset enables RAG)
PLANNING_RAG_INDEX_DIR="/home2/zhangjiawei/respace/group_mining_outputs_fine/faiss_index_augmented"
PLANNING_RAG_MODEL="/home2/zhangjiawei/respace/model/qwen3-embedding-8B"
PLANNING_RAG_DEVICE="cpu"
PLANNING_RAG_BATCH_SIZE="16"
PLANNING_RAG_TOP_K_PER_ANCHOR="2"
PLANNING_RAG_TOP_K_ROOM_LEVEL="3"
PLANNING_RAG_MAX_DOCS_PER_ANCHOR="2"
PLANNING_RAG_MAX_ANCHORS="3"
PLANNING_RAG_ANCHORS=""
PLANNING_RAG_WRITE_DEBUG="1"
PLANNING_RAG_REQUIRE_INDEX="0"

# Group planner decode controls (reduce OOM risk)
GROUP_PLAN_MAX_NEW_TOKENS="384"
GROUP_PLAN_RETRY_MAX_NEW_TOKENS="160"
GROUP_PLAN_MAX_QUERY_CHARS="2800"
GROUP_PLAN_RETRY_QUERY_CHARS="1200"
GROUP_PLAN_TEMPERATURE="0.0"
GROUP_PLAN_TOP_P="1.0"
GROUP_PLAN_TOP_K="0"
GROUP_SIMPLE_OPT_PASSES="2"

# Keep these fixed across ablations unless you intentionally want another ablation
RUN_FINAL_GLOBAL_REPAIR_DEFAULT="1"
INCLUDE_RELATION_PLAN_DEFAULT="0"

# -----------------------------
# Preset -> module switches
# -----------------------------
case "${EXPERIMENT_PRESET}" in
  vanilla_respace)
    ENABLE_PLANNING_RAG_DEFAULT="0"
    USE_GROUP_REPAIR_IN_LOOP_DEFAULT="0"
    ENABLE_VLM_OPTIMIZATION_DEFAULT="0"
    ;;
  vlm_only)
    ENABLE_PLANNING_RAG_DEFAULT="0"
    USE_GROUP_REPAIR_IN_LOOP_DEFAULT="0"
    ENABLE_VLM_OPTIMIZATION_DEFAULT="1"
    ;;
  rag_only)
    ENABLE_PLANNING_RAG_DEFAULT="1"
    USE_GROUP_REPAIR_IN_LOOP_DEFAULT="0"
    ENABLE_VLM_OPTIMIZATION_DEFAULT="0"
    ;;
  without_vlm)
    ENABLE_PLANNING_RAG_DEFAULT="1"
    USE_GROUP_REPAIR_IN_LOOP_DEFAULT="1"
    ENABLE_VLM_OPTIMIZATION_DEFAULT="0"
    ;;
  rag_group_vlm)
    ENABLE_PLANNING_RAG_DEFAULT="1"
    USE_GROUP_REPAIR_IN_LOOP_DEFAULT="1"
    ENABLE_VLM_OPTIMIZATION_DEFAULT="1"
    ;;
  without_group)
    ENABLE_PLANNING_RAG_DEFAULT="1"
    USE_GROUP_REPAIR_IN_LOOP_DEFAULT="0"
    ENABLE_VLM_OPTIMIZATION_DEFAULT="1"
    ;;
  without_rag)
    ENABLE_PLANNING_RAG_DEFAULT="0"
    USE_GROUP_REPAIR_IN_LOOP_DEFAULT="1"
    ENABLE_VLM_OPTIMIZATION_DEFAULT="1"
    ;;
  *)
    echo "[ERROR] Unknown EXPERIMENT_PRESET=${EXPERIMENT_PRESET}"
    exit 1
    ;;
esac

cd "$PROJECT_ROOT"

# -----------------------------
# Conda activation
# -----------------------------
if [[ -n "${CONDA_ENV_NAME}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    __had_nounset=0
    if [[ $- == *u* ]]; then
      __had_nounset=1
      set +u
    fi
    eval "$(conda shell.bash hook)"
    if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]]; then
      conda activate "$CONDA_ENV_NAME"
    else
      echo "[INFO] already in conda env: $CONDA_ENV_NAME"
    fi
    if [[ $__had_nounset -eq 1 ]]; then
      set -u
    fi
    unset __had_nounset
  else
    echo "[WARN] conda not found, skip activation: ${CONDA_ENV_NAME}"
  fi
fi

# -----------------------------
# Basic checks
# -----------------------------
if [[ ! -f "$IN_JSONL" ]]; then
  echo "[ERROR] Input JSONL not found: $IN_JSONL"
  exit 1
fi
if [[ ! -d "$EMPTY_ROOT" ]]; then
  echo "[ERROR] empty scenes root not found: $EMPTY_ROOT"
  exit 1
fi
if [[ "$EXPERIMENT_PRESET" != "vanilla_respace" ]]; then
  if [[ ! -f "$PY_ENHANCED_SCRIPT" ]]; then
    echo "[ERROR] Enhanced Python script not found: $PY_ENHANCED_SCRIPT"
    exit 1
  fi
  if [[ "${USE_LOCAL_VLM_OPTIMIZER:-0}" != "1" ]]; then
    if [[ -z "$YUNWU_AI_API_KEY" || "$YUNWU_AI_API_KEY" == "YOUR_API_KEY_HERE" ]]; then
      echo "[ERROR] Please fill in YUNWU_AI_API_KEY before running non-vanilla presets."
      exit 1
    fi
  fi
  if [[ "$ENABLE_PLANNING_RAG_DEFAULT" == "1" ]]; then
    if [[ ! -d "$PLANNING_RAG_INDEX_DIR" ]]; then
      echo "[ERROR] PLANNING_RAG_INDEX_DIR not found: $PLANNING_RAG_INDEX_DIR"
      exit 1
    fi
    if [[ ! -e "$PLANNING_RAG_MODEL" ]]; then
      echo "[ERROR] PLANNING_RAG_MODEL not found: $PLANNING_RAG_MODEL"
      exit 1
    fi
  fi
fi

mkdir -p "$OUT_ROOT"
mkdir -p "$OUT_ROOT/batch_logs"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
# -----------------------------
# Shared env exports
# -----------------------------
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export USE_LOCAL_VLM_OPTIMIZER
export LOCAL_VLM_MODEL
export LOCAL_VLM_DEVICE
export LOCAL_VLM_DTYPE
export YUNWU_AI_API_BASE
export YUNWU_AI_API_KEY
export YUNWU_AI_MODEL
export YUNWU_AI_BASE_URL="$YUNWU_AI_API_BASE"
export MOVE_PROMPT_MODEL="$MOVE_PROMPT_MODEL"
export PY_ENHANCED_SCRIPT
export IN_JSONL
export EMPTY_ROOT
export OUT_ROOT
export LIMIT
export RESUME
export RETRY_FAILED
export SHUFFLE

# Stable ablation switches
export ENABLE_PLANNING_RAG="$ENABLE_PLANNING_RAG_DEFAULT"
export USE_GROUP_REPAIR_IN_LOOP="$USE_GROUP_REPAIR_IN_LOOP_DEFAULT"
export ENABLE_VLM_OPTIMIZATION="$ENABLE_VLM_OPTIMIZATION_DEFAULT"
export RUN_FINAL_GLOBAL_REPAIR="$RUN_FINAL_GLOBAL_REPAIR_DEFAULT"
export INCLUDE_RELATION_PLAN="$INCLUDE_RELATION_PLAN_DEFAULT"

# Planning RAG env
export PLANNING_RAG_INDEX_DIR
export PLANNING_RAG_MODEL
export PLANNING_RAG_DEVICE
export PLANNING_RAG_BATCH_SIZE
export PLANNING_RAG_TOP_K_PER_ANCHOR
export PLANNING_RAG_TOP_K_ROOM_LEVEL
export PLANNING_RAG_MAX_DOCS_PER_ANCHOR
export PLANNING_RAG_MAX_ANCHORS
export PLANNING_RAG_ANCHORS
export PLANNING_RAG_WRITE_DEBUG
export PLANNING_RAG_REQUIRE_INDEX
export GROUP_PLAN_MAX_NEW_TOKENS
export GROUP_PLAN_RETRY_MAX_NEW_TOKENS
export GROUP_PLAN_MAX_QUERY_CHARS
export GROUP_PLAN_RETRY_QUERY_CHARS
export GROUP_PLAN_TEMPERATURE
export GROUP_PLAN_TOP_P
export GROUP_PLAN_TOP_K
export GROUP_SIMPLE_OPT_PASSES

# -----------------------------
# Enhanced-mode optimizer / prompt settings
# PBL + REL + FUNC only
# -----------------------------
# VLM action prompt settings. These are still needed by rag_group_vlm.
export MOVE_PROMPT_TIMEOUT_S="120"
export MOVE_PROMPT_TEMPERATURE="0.0"
export MOVE_PROMPT_MAX_TOKENS="1600"
export MOVE_PROMPT_RETRIES="2"

# Relation settings. Keep these because REL is one of the three retained scores.
export USE_VLM_RELATION_PRIORS="1"
export REFRESH_VLM_RELATION_PRIORS_EVERY_STEP="0"
export MERGE_VLM_WITH_DETERMINISTIC="1"
export VLM_PRIOR_WEIGHT_SCALE="0.35"
export FREEZE_DETERMINISTIC_PRIORS="0"
export RELATION_PRIOR_RETRIES="2"
export RELATION_PRIOR_TEMPERATURE="0.0"
export RELATION_PRIOR_MAX_TOKENS="900"
export RELATION_PRIOR_CONFIDENCE="0.55"
export USE_ZERO_SHOT_RELATION_PLAN="1"
export ZERO_SHOT_RELATION_USE_MODE="canonical_merged"
export ZERO_SHOT_RELATION_WEIGHT_SCALE="0.85"
export RELATION_REFRESH_MODE="on_stagnation"

# Core optimizer controls. These are enough for PBL / REL / FUNC scoring.
export MAX_STEPS="1"
export MAX_ROUNDS="2"
export MAX_OBJECTS_PER_ROUND="6"
export PROXY_TOPK="3"
export STEP_XY="0.22"
export STEP_YAW="15.0"
export LOCAL_REPAIR_PASSES="1"
export FULL_REPAIR_AFTER_REFINE_PASSES="1"
export MONOTONIC_EPS="1e-12"
export ANCHOR_LOCK_PBL_THRESHOLD="0.08"
export ROLE_REFINE_BLEND="0.66"
export VALID_PBL_THRESHOLD="0.10"

# Stop / acceptance constraints for the three retained terms only.
export STOP_WHEN_VALID_PBL="1"
export CLEANUP_ONLY_AFTER_VALID_PBL="0"
export STOP_SCORE_THRESHOLD="0.20"
export SKIP_POST_REFINE_WHEN_VALID_PBL="0"
export CANDIDATE_FILTER_REINTRODUCED_PBL="1"
export MAX_STEPS_AFTER_VALID_PBL="2"
export MAX_OBJECTS_AFTER_VALID_PBL="4"
export MAX_ROUNDS_AFTER_VALID_PBL="1"
export MIN_SCORE_IMPROVE_AFTER_VALID_PBL="0.02"
export MAX_REL_INCREASE_AFTER_VALID_PBL="0.12"
export MAX_FUNC_INCREASE_AFTER_VALID_PBL="0.12"
export MAX_SCORE_INCREASE_PREVALID="0.06"
export MAX_REL_INCREASE_PREVALID="0.55"
export MAX_FUNC_INCREASE_PREVALID="0.35"
export RENDER_FINAL="1"

# Explicitly disable modules tied to removed structure/open/mono/spread/flow terms.
export USE_STRUCTURAL_GUARD="0"
export ADD_ZONE_RELEASE_CANDIDATES="0"
export ENABLE_ZONE_LAYOUT_CANDIDATES="0"
export USE_ITERATION_HISTORY="0"
export ADAPTIVE_RELATION_TRADEOFF="0"
export REWRITE_PRIORS_ON_RELAYOUT="0"
export REFRESH_VLM_RELATION_PRIORS_ON_STAGNATION="0"
export FORCE_HISTORY_REPLAN_ON_MONOPOLY="0"
export ENABLE_DUAL_JUDGE="0"
export PROPOSAL_BRANCH_SEARCH="0"
export ENABLE_STABLE_BRANCH="0"
export ENABLE_DYNAMIC_DET_BRANCH="1"
export ENABLE_DYNAMIC_VLM_BRANCH="0"
export MANDATORY_FINAL_POLISH="0"
export FINAL_POLISH_PASSES="0"
export USE_STRUCTURED_REPAIR_PLAN="1"
export STRUCTURED_PLAN_MAX_ACTIONS="6"
export STRUCTURED_PLAN_APPLY_BEFORE_SEARCH="1"
export STRUCTURED_PLAN_FORCE_JSON="1"

# -----------------------------
# Batch metadata
# -----------------------------
RUN_TS=$(date +"%Y%m%d_%H%M%S")
BATCH_LOG="$OUT_ROOT/batch_logs/batch_${RUN_TS}.log"

echo "[batch] PROJECT_ROOT=$PROJECT_ROOT" | tee -a "$BATCH_LOG"
echo "[batch] IN_JSONL=$IN_JSONL" | tee -a "$BATCH_LOG"
echo "[batch] EMPTY_ROOT=$EMPTY_ROOT" | tee -a "$BATCH_LOG"
echo "[batch] OUT_ROOT=$OUT_ROOT" | tee -a "$BATCH_LOG"
echo "[batch] EXPERIMENT_PRESET=$EXPERIMENT_PRESET" | tee -a "$BATCH_LOG"
echo "[batch] ENABLE_PLANNING_RAG=$ENABLE_PLANNING_RAG" | tee -a "$BATCH_LOG"
echo "[batch] USE_GROUP_REPAIR_IN_LOOP=$USE_GROUP_REPAIR_IN_LOOP" | tee -a "$BATCH_LOG"
echo "[batch] ENABLE_VLM_OPTIMIZATION=$ENABLE_VLM_OPTIMIZATION" | tee -a "$BATCH_LOG"
echo "[batch] CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" | tee -a "$BATCH_LOG"

# -----------------------------
# Branch 1: vanilla ReSpace baseline
# -----------------------------
if [[ "$EXPERIMENT_PRESET" == "vanilla_respace" ]]; then
  export BASELINE_RENDER_FRAME
  export BASELINE_RENDER_TOP

  stdbuf -oL -eL env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" - <<'PY' 2>&1 | tee -a "$BATCH_LOG"
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

IN_JSONL = Path(os.environ["IN_JSONL"])
EMPTY_ROOT = Path(os.environ["EMPTY_ROOT"])
OUT_DIR = Path(os.environ["OUT_ROOT"])
LIMIT = int(os.environ.get("LIMIT", "0"))
RESUME = os.environ.get("RESUME", "1") == "1"
RETRY_FAILED = os.environ.get("RETRY_FAILED", "0") == "1"
RENDER_FRAME = os.environ.get("BASELINE_RENDER_FRAME", "0") == "1"
RENDER_TOP = os.environ.get("BASELINE_RENDER_TOP", "1") == "1"


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
        idx.setdefault(p.stem, p)
    return idx


def pick_scene_id(rec: Dict[str, Any]) -> Optional[str]:
    for k in ("scene_id", "sceneId", "id"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def pick_prompt(rec: Dict[str, Any]) -> Optional[str]:
    for k in ("prompt", "room_prompt", "ROOM_PROMPT", "text"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    v = rec.get("data")
    if isinstance(v, dict):
        for k in ("prompt", "room_prompt", "text"):
            vv = v.get(k)
            if isinstance(vv, str) and vv.strip():
                return vv.strip()
    return None


def load_latest_status(results_path: Path) -> Dict[str, str]:
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


from src.respace import ReSpace  # type: ignore
from src.viz import render_annotated_top_view  # type: ignore

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_scenes_dir = OUT_DIR / "updated_scenes"
out_scenes_dir.mkdir(parents=True, exist_ok=True)
renders_dir = OUT_DIR / "renders"
renders_dir.mkdir(parents=True, exist_ok=True)
run_log_path = OUT_DIR / "results.jsonl"
last_status = load_latest_status(run_log_path) if (RESUME and RETRY_FAILED) else {}

respace = ReSpace()
empty_idx = index_empty_scenes(EMPTY_ROOT)

n_total = 0
n_ok = 0
n_fail = 0
n_skip = 0
t0 = time.time()

with run_log_path.open("a", encoding="utf-8") as flog:
    for ln, rec in load_jsonl(IN_JSONL):
        n_total += 1
        if LIMIT > 0 and n_total > LIMIT:
            break

        scene_id = pick_scene_id(rec)
        prompt = pick_prompt(rec)

        if not scene_id or not prompt:
            n_fail += 1
            flog.write(json.dumps({
                "line": ln,
                "status": "bad_record",
                "scene_id": scene_id,
                "has_prompt": bool(prompt),
                "record": rec,
            }, ensure_ascii=False) + "\n")
            continue

        scene_path = empty_idx.get(scene_id)
        if scene_path is None:
            n_fail += 1
            flog.write(json.dumps({
                "line": ln,
                "status": "missing_scene",
                "scene_id": scene_id,
                "prompt": prompt,
            }, ensure_ascii=False) + "\n")
            continue

        out_scene_path = out_scenes_dir / f"{scene_id}_updated.json"
        per_scene_render_dir = renders_dir / scene_id

        if RESUME and out_scene_path.exists():
            if RETRY_FAILED:
                prev = last_status.get(scene_id)
                if prev == "ok":
                    n_skip += 1
                    flog.write(json.dumps({
                        "line": ln,
                        "status": "skipped_existing_ok",
                        "scene_id": scene_id,
                        "prev_status": prev,
                        "scene_path": str(scene_path),
                        "out_scene_path": str(out_scene_path),
                    }, ensure_ascii=False) + "\n")
                    continue
            else:
                n_skip += 1
                flog.write(json.dumps({
                    "line": ln,
                    "status": "skipped_existing",
                    "scene_id": scene_id,
                    "scene_path": str(scene_path),
                    "out_scene_path": str(out_scene_path),
                }, ensure_ascii=False) + "\n")
                continue

        try:
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            updated_scene, is_success = respace.handle_prompt(prompt, scene)

            out_scene_path.write_text(
                json.dumps(updated_scene, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            render_errors = []
            if RENDER_FRAME or RENDER_TOP:
                per_scene_render_dir.mkdir(parents=True, exist_ok=True)

            if RENDER_FRAME:
                try:
                    respace.render_scene_frame(updated_scene, filename="frame", pth_viz_output=per_scene_render_dir)
                except Exception as e:
                    render_errors.append({"what": "render_scene_frame", "error": repr(e)})

            if RENDER_TOP:
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

            flog.write(json.dumps({
                "line": ln,
                "status": status,
                "scene_id": scene_id,
                "prompt": prompt,
                "scene_path": str(scene_path),
                "out_scene_path": str(out_scene_path),
                "is_success": bool(is_success),
                "render_dir": str(per_scene_render_dir) if (RENDER_FRAME or RENDER_TOP) else None,
                "render_errors": render_errors,
            }, ensure_ascii=False) + "\n")

            if RESUME and RETRY_FAILED:
                last_status[scene_id] = status

        except Exception as e:
            n_fail += 1
            flog.write(json.dumps({
                "line": ln,
                "status": "exception",
                "scene_id": scene_id,
                "prompt": prompt,
                "scene_path": str(scene_path),
                "out_scene_path": str(out_scene_path),
                "error": repr(e),
            }, ensure_ascii=False) + "\n")
            if RESUME and RETRY_FAILED:
                last_status[scene_id] = "exception"


dt = time.time() - t0
print("[DONE]")
print(f"  preset     : vanilla_respace")
print(f"  in_jsonl   : {IN_JSONL}")
print(f"  empty_root : {EMPTY_ROOT}")
print(f"  out_dir    : {OUT_DIR}")
print(f"  total      : {n_total}")
print(f"  ok         : {n_ok}")
print(f"  fail       : {n_fail}")
print(f"  skip       : {n_skip}")
print(f"  seconds    : {dt:.1f}")
print(f"  results    : {run_log_path}")
PY

  exit 0
fi

# -----------------------------
# Branch 2: enhanced presets via infer_with_planning_rag_ablation.py
# -----------------------------
stdbuf -oL -eL env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" - <<'PY' 2>&1 | tee -a "$BATCH_LOG"
import os
import sys
import json
import time
import random
import traceback
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

PY_SCRIPT = Path(os.environ["PY_ENHANCED_SCRIPT"])
IN_JSONL = Path(os.environ["IN_JSONL"])
EMPTY_ROOT = Path(os.environ["EMPTY_ROOT"])
OUT_ROOT = Path(os.environ["OUT_ROOT"])
LIMIT = int(os.environ.get("LIMIT", "0"))
RESUME = os.environ.get("RESUME", "1") == "1"
RETRY_FAILED = os.environ.get("RETRY_FAILED", "0") == "1"
SHUFFLE = os.environ.get("SHUFFLE", "0") == "1"


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
        idx.setdefault(p.stem, p)
    return idx


def pick_scene_id(rec: Dict[str, Any]) -> Optional[str]:
    for k in ("scene_id", "sceneId", "id"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def pick_prompt(rec: Dict[str, Any]) -> Optional[str]:
    for k in ("prompt", "room_prompt", "ROOM_PROMPT", "text"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    v = rec.get("data")
    if isinstance(v, dict):
        for k in ("prompt", "room_prompt", "text"):
            vv = v.get(k)
            if isinstance(vv, str) and vv.strip():
                return vv.strip()
    return None


def scene_done_ok(scene_out_dir: Path) -> bool:
    return (
        (scene_out_dir / "summary.json").exists()
        or (scene_out_dir / "final" / "scene.json").exists()
        or (scene_out_dir / "final_scene_from_group_repair.json").exists()
    )


def load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def last_status_map(results_path: Path) -> Dict[str, str]:
    out = {}
    if not results_path.exists():
        return out
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
                out[sid] = status
    return out


OUT_ROOT.mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "batch_logs").mkdir(parents=True, exist_ok=True)
RESULTS_JSONL = OUT_ROOT / "results.jsonl"
RUN_TS = time.strftime("%Y%m%d_%H%M%S")
MANIFEST_JSONL = OUT_ROOT / "batch_logs" / f"manifest_{RUN_TS}.jsonl"

print(f"[singleproc] loading module from {PY_SCRIPT}")
module = load_module_from_file("infer_with_planning_rag_ablation_module", PY_SCRIPT)

print("[singleproc] attaching group repair hook once")
orig_attach = module.attach_group_repair_in_loop
orig_respace_cls = module.ReSpace
orig_generator_cls = module.GPTVLMovePromptGeneratorV5
orig_attach(orig_respace_cls, module.Config, module.optimize_scene_refactored_v15, orig_generator_cls)

print("[singleproc] loading ReSpace once")
shared_respace = orig_respace_cls()

print("[singleproc] keep GPTVLMovePromptGeneratorV5 lazy-loaded")
module.attach_group_repair_in_loop = lambda *args, **kwargs: None
module.ReSpace = lambda: shared_respace


def _cleanup_cuda():
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

if hasattr(module, "PlanningRAGRetriever") and os.getenv("ENABLE_PLANNING_RAG", "0") not in ("0", "false", "False", ""):
    rag_index = os.getenv("PLANNING_RAG_INDEX_DIR", "").strip()
    rag_model = os.getenv("PLANNING_RAG_MODEL", "").strip()
    if rag_index and rag_model:
        try:
            print("[singleproc] loading PlanningRAGRetriever once")
            orig_rag_cls = module.PlanningRAGRetriever
            shared_rag = orig_rag_cls(
                index_dir=rag_index,
                model_name_or_path=rag_model,
                device=(os.getenv("PLANNING_RAG_DEVICE", "").strip() or None),
                batch_size=int(os.getenv("PLANNING_RAG_BATCH_SIZE", "16")),
            )
            module.PlanningRAGRetriever = lambda *args, **kwargs: shared_rag
        except Exception:
            print("[singleproc] warning: failed to preload PlanningRAGRetriever")
            traceback.print_exc()

empty_idx = index_empty_scenes(EMPTY_ROOT)
rows = []
for ln, rec in load_jsonl(IN_JSONL):
    sid = pick_scene_id(rec)
    prompt = pick_prompt(rec)
    if not sid or not prompt:
        continue
    scene_path = empty_idx.get(sid)
    if scene_path is None:
        continue
    rows.append((ln, sid, str(scene_path), prompt))

if SHUFFLE:
    random.shuffle(rows)

with MANIFEST_JSONL.open("w", encoding="utf-8") as f:
    for ln, sid, scene_path, prompt in rows:
        f.write(json.dumps({"line": ln, "scene_id": sid, "scene_path": scene_path, "prompt": prompt}, ensure_ascii=False) + "\n")

prev_status = last_status_map(RESULTS_JSONL) if (RESUME and RETRY_FAILED) else {}

n_total = 0
n_ok = 0
n_fail = 0
n_skip = 0
t0 = time.time()

with RESULTS_JSONL.open("a", encoding="utf-8") as flog:
    for ln, scene_id, scene_path_str, prompt in rows:
        n_total += 1
        if LIMIT > 0 and n_total > LIMIT:
            break

        scene_path = Path(scene_path_str)
        scene_out_dir = OUT_ROOT / scene_id
        scene_out_dir.mkdir(parents=True, exist_ok=True)

        if RESUME and scene_done_ok(scene_out_dir):
            if RETRY_FAILED:
                if prev_status.get(scene_id) == "ok":
                    n_skip += 1
                    rec = {
                        "line": ln,
                        "scene_id": scene_id,
                        "scene_path": str(scene_path),
                        "out_dir": str(scene_out_dir),
                        "status": "skipped_existing_ok",
                    }
                    flog.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"[SKIP] {scene_id} existing success marker")
                    continue
            else:
                n_skip += 1
                rec = {
                    "line": ln,
                    "scene_id": scene_id,
                    "scene_path": str(scene_path),
                    "out_dir": str(scene_out_dir),
                    "status": "skipped_existing",
                }
                flog.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[SKIP] {scene_id} existing success marker")
                continue

        print("======================================================================")
        print(f"[RUN] idx={n_total} scene_id={scene_id}")
        print(f"[RUN] scene_path={scene_path}")
        print(f"[RUN] out_dir={scene_out_dir}")
        print("======================================================================")

        os.environ["SCENE_JSON_PATH"] = str(scene_path)
        os.environ["ROOM_PROMPT"] = prompt
        os.environ["OUT_DIR"] = str(scene_out_dir)

        status = "ok"
        exit_code = 0
        error_text = None

        try:
            module.main()
            n_ok += 1
        except Exception as e:
            status = "failed"
            exit_code = 1
            error_text = repr(e)
            n_fail += 1
            print(f"[FAIL] scene_id={scene_id}")
            traceback.print_exc()

        rec = {
            "line": ln,
            "scene_id": scene_id,
            "scene_path": str(scene_path),
            "out_dir": str(scene_out_dir),
            "prompt": prompt,
            "status": status,
            "exit_code": exit_code,
            "error": error_text,
        }
        flog.write(json.dumps(rec, ensure_ascii=False) + "\n")
        flog.flush()

        if RESUME and RETRY_FAILED:
            prev_status[scene_id] = status

        print(f"[DONE] scene_id={scene_id} status={status} exit_code={exit_code}")
        _cleanup_cuda()


dt = time.time() - t0
print("")
print("======================================================================")
print(f"[singleproc] preset={os.environ.get('EXPERIMENT_PRESET', 'unknown')}")
print(f"[singleproc] total={n_total}")
print(f"[singleproc] ok={n_ok}")
print(f"[singleproc] fail={n_fail}")
print(f"[singleproc] skip={n_skip}")
print(f"[singleproc] seconds={dt:.1f}")
print(f"[singleproc] results={RESULTS_JSONL}")
print("======================================================================")
PY
