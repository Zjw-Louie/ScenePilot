#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# V15 group-wise repair-in-the-loop
# 直接修改下面 CONFIG 区域，然后运行：
#   bash run_v15_group_repair_loop.sh
# ============================================================

CONDA_ENV_NAME="respace"
PROJECT_ROOT="/home2/zhangjiawei/respace"
PY_SCRIPT="/home2/zhangjiawei/respace/infer_v15.py"
SCENE_JSON_PATH="/home2/zhangjiawei/respace/benchmark/empty_scenes/livingroom/032e5dbc-4026-4e03-84f1-e75553e339a3-f13ff5bc-fcd7-4b0d-982f-020ffb0e720c.json"
OUT_DIR="/home2/zhangjiawei/respace/evaluate_date/evaluate_$(date +%m%d)/infer_v15_run1"

YUNWU_AI_API_BASE="https://yunwu.ai"
YUNWU_AI_API_KEY="sk-3PxZML90syfHtBF9PP6gFdG0GGwyUV97hJZ6iIKTwApAvwib"
MOVE_PROMPT_MODEL="gpt-4o"
YUNWU_AI_MODEL="gpt-4o"

ROOM_PROMPT="create a modern living room include gray L-shaped sofa, oval coffee table, dining table with four chairs, TV stand, sideboard, floor lamp, pendant lamps, lounge chair, and corner side table"

CUDA_DEVICES="0,1,2"
PYTHON_BIN="python"

cd "$PROJECT_ROOT"

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

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] Python script not found: $PY_SCRIPT"
  exit 1
fi
if [[ ! -f "$SCENE_JSON_PATH" ]]; then
  echo "[ERROR] Scene json not found: $SCENE_JSON_PATH"
  exit 1
fi
if [[ -z "$YUNWU_AI_API_KEY" || "$YUNWU_AI_API_KEY" == "YOUR_API_KEY_HERE" ]]; then
  echo "[ERROR] Please fill in YUNWU_AI_API_KEY in this script first."
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export SCENE_JSON_PATH
export OUT_DIR
export ROOM_PROMPT
export YUNWU_AI_API_BASE
export YUNWU_AI_API_KEY
export YUNWU_AI_MODEL
export YUNWU_AI_BASE_URL="$YUNWU_AI_API_BASE"
export MOVE_PROMPT_MODEL="${MOVE_PROMPT_MODEL:-gpt-4o}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

# -----------------------------
# ablation master switches
# -----------------------------
export ENABLE_PLANNING_RAG="1"
export ENABLE_VLM_OPTIMIZATION="0"

# -----------------------------
# planning RAG
# -----------------------------
export PLANNING_RAG_INDEX_DIR="/home2/zhangjiawei/respace/group_mining_outputs_fine/faiss_index_augmented"
export PLANNING_RAG_MODEL="/home2/zhangjiawei/respace/model/qwen3-embedding-8B"
export PLANNING_RAG_DEVICE="cuda"
export PLANNING_RAG_BATCH_SIZE="16"
export PLANNING_RAG_TOP_K_PER_ANCHOR="2"
export PLANNING_RAG_TOP_K_ROOM_LEVEL="3"
export PLANNING_RAG_MAX_DOCS_PER_ANCHOR="2"
export PLANNING_RAG_MAX_ANCHORS="4"
export PLANNING_RAG_ANCHORS=""
export PLANNING_RAG_WRITE_DEBUG="1"
export PLANNING_RAG_REQUIRE_INDEX="0"

# -----------------------------
# mode switches
# -----------------------------
export USE_GROUP_REPAIR_IN_LOOP="1"
export RUN_FINAL_GLOBAL_REPAIR="1"
export INCLUDE_RELATION_PLAN="1"

# -----------------------------
# group planner / simple partial repair
# -----------------------------
export MAX_FUNCTIONAL_GROUPS="4"
export GROUP_SIMPLE_OPT_PASSES="3"
export GROUP_PARTIAL_MAX_STEPS="1"
export GROUP_PARTIAL_MAX_ROUNDS="1"
export GROUP_PARTIAL_MAX_OBJECTS_PER_ROUND="6"
export GROUP_PARTIAL_PROXY_TOPK="2"
export GROUP_PARTIAL_MOVE_PROMPT_MAX_TOKENS="1200"

# -----------------------------
# VLM prompt settings
# -----------------------------
export MOVE_PROMPT_TIMEOUT_S="120"
export MOVE_PROMPT_TEMPERATURE="0.7"
export MOVE_PROMPT_MAX_TOKENS="1600"
export MOVE_PROMPT_RETRIES="2"

# -----------------------------
# relation priors
# -----------------------------
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

# -----------------------------
# optimizer core
# -----------------------------
export MAX_STEPS="4"
export MAX_ROUNDS="2"
export MAX_OBJECTS_PER_ROUND="8"
export PROXY_TOPK="3"
export STEP_XY="0.22"
export STEP_YAW="15.0"
export LOCAL_REPAIR_PASSES="1"
export FULL_REPAIR_AFTER_REFINE_PASSES="1"
export MONOTONIC_EPS="1e-12"
export ANCHOR_LOCK_PBL_THRESHOLD="0.08"
export ROLE_REFINE_BLEND="0.66"
export VALID_PBL_THRESHOLD="0.10"

# -----------------------------
# structure guard
# -----------------------------
export USE_STRUCTURAL_GUARD="1"
export MIN_OPEN_SPACE_RATIO="0.40"
export MAX_ZONE_MONOPOLY_RATIO="0.68"
export CORRIDOR_WIDTH_RATIO="0.18"
export MAX_OPEN_SPACE_DROP_AFTER_VALID_PBL="0.04"
export MAX_MONOPOLY_INCREASE_AFTER_VALID_PBL="0.05"
export MAX_SPREAD_INCREASE_AFTER_VALID_PBL="0.06"
export MAX_FLOW_INCREASE_AFTER_VALID_PBL="0.06"
export REQUIRE_ZONE_COUNT_PRESERVE_AFTER_VALID_PBL="1"

# -----------------------------
# history / branch search / final polish
# -----------------------------
export USE_ITERATION_HISTORY="1"
export MAX_HISTORY_STEPS="4"
export REPEAT_REJECT_PATIENCE="2"
export ADAPTIVE_RELATION_TRADEOFF="1"
export REWRITE_PRIORS_ON_RELAYOUT="1"
export REFRESH_VLM_RELATION_PRIORS_ON_STAGNATION="1"
export FORCE_HISTORY_REPLAN_ON_MONOPOLY="1"
export RELAYOUT_ACCEPT_MIN_SCORE_GAIN="0.03"
export RELAYOUT_ACCEPT_MIN_STRUCTURE_GAIN="0.01"
export RELAYOUT_ACCEPT_MIN_MONOPOLY_GAIN="0.15"
export RELAYOUT_ACCEPT_MIN_ZONE_GAIN="1"

export ENABLE_DUAL_JUDGE="1"
export STABLE_JUDGE_WEIGHT="0.75"
export JUDGE_MIN_SCORE_IMPROVE_AFTER_VALID_PBL="0.01"
export JUDGE_MAX_SCORE_INCREASE_AFTER_VALID_PBL="0.02"
export JUDGE_MAX_SCORE_INCREASE_PREVALID="0.08"
export PROPOSAL_BRANCH_SEARCH="1"
export ENABLE_STABLE_BRANCH="1"
export ENABLE_DYNAMIC_DET_BRANCH="1"
export ENABLE_DYNAMIC_VLM_BRANCH="1"
export MANDATORY_FINAL_POLISH="1"
export FINAL_POLISH_PASSES="2"

# -----------------------------
# acceptance / stop
# -----------------------------
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

# -----------------------------
# structured repair plan
# -----------------------------
export USE_STRUCTURED_REPAIR_PLAN="1"
export STRUCTURED_PLAN_MAX_ACTIONS="6"
export STRUCTURED_PLAN_APPLY_BEFORE_SEARCH="1"
export STRUCTURED_PLAN_FORCE_JSON="1"

mkdir -p "$OUT_DIR"

echo "[V15] PROJECT_ROOT=$PROJECT_ROOT"
echo "[V15] PY_SCRIPT=$PY_SCRIPT"
echo "[V15] SCENE_JSON_PATH=$SCENE_JSON_PATH"
echo "[V15] OUT_DIR=$OUT_DIR"
echo "[V15] ROOM_PROMPT=$ROOM_PROMPT"
echo "[V15] USE_GROUP_REPAIR_IN_LOOP=$USE_GROUP_REPAIR_IN_LOOP"
echo "[V15] RUN_FINAL_GLOBAL_REPAIR=$RUN_FINAL_GLOBAL_REPAIR"
echo "[V15] MAX_FUNCTIONAL_GROUPS=$MAX_FUNCTIONAL_GROUPS"
echo "[V15] GROUP_SIMPLE_OPT_PASSES=$GROUP_SIMPLE_OPT_PASSES"
echo "[V15] YUNWU_AI_API_BASE=$YUNWU_AI_API_BASE"
echo "[V15] YUNWU_AI_MODEL=$YUNWU_AI_MODEL"

RUN_TS=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUT_DIR/terminal_logs"
mkdir -p "$LOG_DIR"

FULL_LOG="$LOG_DIR/run_${RUN_TS}.log"
GROUP_LOG="$LOG_DIR/run_${RUN_TS}.group.log"
GROUP_REGEX='^\[group mode\]|^================ GROUP|^\[group info\]|^\[group prompt\]|^\[group add result\]|^\[group summary\]'

echo
echo "======================================================================"
echo "[infer-v15] start_ts=${RUN_TS}"
echo "[infer-v15] out_dir=${OUT_DIR}"
echo "[infer-v15] full_log=${FULL_LOG}"
echo "[infer-v15] group_log=${GROUP_LOG}"
echo "======================================================================"
echo

stdbuf -oL -eL env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$PY_SCRIPT" 2>&1 \
  | tee -a "$FULL_LOG" \
      >(grep --line-buffered -E "$GROUP_REGEX" >> "$GROUP_LOG") \
  | awk '
      BEGIN {
          cyan    = "\033[1;36m";
          yellow  = "\033[1;33m";
          green   = "\033[1;32m";
          blue    = "\033[1;34m";
          magenta = "\033[1;35m";
          red     = "\033[1;31m";
          reset   = "\033[0m";
      }
      /^\[group mode\]/ {
          print cyan $0 reset; fflush(); next
      }
      /^================ GROUP/ {
          print "\n" magenta $0 reset; fflush(); next
      }
      /^\[group info\]/ {
          print yellow $0 reset; fflush(); next
      }
      /^\[group prompt\]/ {
          print blue $0 reset; fflush(); next
      }
      /^\[group add result\]/ {
          print green $0 reset; fflush(); next
      }
      /^\[group summary\]/ {
          print cyan $0 reset; fflush(); next
      }
      /^Traceback/ {
          print red $0 reset; fflush(); next
      }
      { print; fflush(); }
  '

EXIT_CODE=${PIPESTATUS[0]}

echo
echo "======================================================================"
echo "[infer-v15] finish_ts=$(date +"%Y%m%d_%H%M%S")"
echo "[infer-v15] exit_code=${EXIT_CODE}"
echo "[infer-v15] full_log=${FULL_LOG}"
echo "[infer-v15] group_log=${GROUP_LOG}"
echo "======================================================================"
echo

exit ${EXIT_CODE}
