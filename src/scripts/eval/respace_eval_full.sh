#!/bin/bash

N_TEST_SCENES=500
BON_LLM=8

DO_ICL_FOR_PROMPT=true
DO_CLASS_LABELS_FOR_PROMPT=true
DO_PROP_SAMPLING_FOR_PROMPT=true
ICL_K=2

export TOKENIZERS_PARALLELISM=false

# -----------------------------
# env 相关：你的 .env 在这里
ENV_FILE="/home/zhangjiawei/respace/.env"

# 关键：确保 pipeline/eval 能拿到需要的环境变量（至少 PTH_STAGE_3）
# TODO: 把下面路径改成你实际数据路径（否则仍会报 NoneType）
export PTH_STAGE_3="/home/zhangjiawei/respace/splits"
export PTH_STAGE_2_DEDUP="/home/zhangjiawei/respace/scenes"

# 可选：如果你的 .env 里已经定义了这些变量，直接 source 它
# 注意：.env 必须是 "KEY=VALUE" 这种 shell 兼容格式
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo "[ERROR] ENV_FILE not found: $ENV_FILE" >&2
  exit 1
fi

# 运行前检查，避免跑半天才挂
if [ -z "${PTH_STAGE_3:-}" ]; then
  echo "[ERROR] PTH_STAGE_3 is empty. Check your $ENV_FILE or export it in this script." >&2
  exit 1
fi

if [ -z "${PTH_STAGE_2_DEDUP:-}" ]; then
  echo "[ERROR] PTH_STAGE_2_DEDUP is empty. Check your $ENV_FILE or export it in this script." >&2
  exit 1
fi

if [ ! -d "$PTH_STAGE_2_DEDUP" ]; then
  echo "[ERROR] PTH_STAGE_2_DEDUP dir not found: $PTH_STAGE_2_DEDUP" >&2
  exit 1
fi

# source .venv/bin/activate

# ************************************************************************************************************************************************************************************
# bedroom

# ROOM_TYPE=bedroom
# MODEL_ID="64663807/checkpoint-best" # qwen1.5B full all + grpo beta 0.0 (may04)

# OUTPUT_DIR_SCENES="./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-grpo-bon-${BON_LLM}/json"
# OUTPUT_DIR_VIZ="./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-grpo-bon-${BON_LLM}/viz"

# if [ "$DO_ICL_FOR_PROMPT" = "true" ]; then
#   ICL_FLAG="--do-icl-for-prompt"
# else
#   ICL_FLAG=""
# fi

# if [ "$DO_CLASS_LABELS_FOR_PROMPT" = "true" ]; then
#   CLASS_LABELS_FLAG="--do-class-labels-for-prompt"
# else
#   CLASS_LABELS_FLAG=""
# fi

# if [ "$DO_PROP_SAMPLING_FOR_PROMPT" = "true" ]; then
#   PROP_SAMPLING_FLAG="--do-prop-sampling-for-prompt"
# else
#   PROP_SAMPLING_FLAG=""
# fi

# # generate samples
# rm -rf "$OUTPUT_DIR_SCENES"
# mkdir -p "$OUTPUT_DIR_SCENES"/{1234,3456,5678}

# # 这里的 --env 必须是枚举值：local/stanley/sherlock
# PYTHONPATH=. xvfb-run -a python ./src/pipeline.py \
#   --use-gpu \
#   --pth-output="$OUTPUT_DIR_SCENES" \
#   --env="local" \
#   --room-type="$ROOM_TYPE" \
#   --model-id="$MODEL_ID" \
#   --n-test-scenes="$N_TEST_SCENES" \
#   --n-bon-sgllm="$BON_LLM" \
#   $ICL_FLAG $CLASS_LABELS_FLAG $PROP_SAMPLING_FLAG \
#   --icl-k="$ICL_K" \
#   --do-full-scenes \
#   --do-bedroom-testset

# # compute metrics
# rm -rf "$OUTPUT_DIR_VIZ"
# mkdir -p "$OUTPUT_DIR_VIZ"/{1234,3456,5678}

# PYTHONPATH=. xvfb-run -a python ./src/eval.py \
#   --pth-input="$OUTPUT_DIR_SCENES" \
#   --pth-output="$OUTPUT_DIR_VIZ" \
#   --env="local" \
#   --room-type="$ROOM_TYPE" \
#   --do-metrics \
#   --n-test-scenes="$N_TEST_SCENES" \
#   --is-full-scene

# ************************************************************************************************************************************************************************************
# livingroom

# ROOM_TYPE=livingroom
# MODEL_ID=64663807/checkpoint-best # qwen1.5B full all + grpo beta 0.0 (may04)

# OUTPUT_DIR_SCENES=./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-grpo-bon-${BON_LLM}/json
# OUTPUT_DIR_VIZ=./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-grpo-bon-${BON_LLM}/viz

# if [ "$DO_ICL_FOR_PROMPT" = "true" ]; then
#     ICL_FLAG="--do-icl-for-prompt"
# else
#     ICL_FLAG=""
# fi

# if [ "$DO_CLASS_LABELS_FOR_PROMPT" = "true" ]; then
#     CLASS_LABELS_FLAG="--do-class-labels-for-prompt"
# else
#     CLASS_LABELS_FLAG=""
# fi

# if [ "$DO_PROP_SAMPLING_FOR_PROMPT" = "true" ]; then
#     PROP_SAMPLING_FLAG="--do-prop-sampling-for-prompt"
# else
#     PROP_SAMPLING_FLAG=""
# fi

# # generate samples
# rm -rf $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES
# mkdir -p $OUTPUT_DIR_SCENES/1234
# mkdir -p $OUTPUT_DIR_SCENES/3456
# mkdir -p $OUTPUT_DIR_SCENES/5678
# xvfb-run -a python src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env="stanley" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --n-bon-sgllm=$BON_LLM $ICL_FLAG $CLASS_LABELS_FLAG $PROP_SAMPLING_FLAG --icl-k=$ICL_K --do-full-scenes --do-livingroom-testset

# # compute metrics
# rm -rf $OUTPUT_DIR_VIZ
# mkdir -p $OUTPUT_DIR_VIZ
# mkdir -p $OUTPUT_DIR_VIZ/1234
# mkdir -p $OUTPUT_DIR_VIZ/3456
# mkdir -p $OUTPUT_DIR_VIZ/5678
# xvfb-run -a python src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env="stanley" --room-type=$ROOM_TYPE --do-metrics --n-test-scenes=$N_TEST_SCENES --is-full-scene

# ************************************************************************************************************************************************************************************
# all

ROOM_TYPE=all
MODEL_ID=64663807/checkpoint-best # qwen1.5B full all + grpo beta 0.0 (may04)

OUTPUT_DIR_SCENES=./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-grpo-bon-${BON_LLM}/json
OUTPUT_DIR_VIZ=./eval/samples/respace/full/${ROOM_TYPE}-with-qwen1.5b-all-grpo-bon-${BON_LLM}/viz

if [ "$DO_ICL_FOR_PROMPT" = "true" ]; then
    ICL_FLAG="--do-icl-for-prompt"
else
    ICL_FLAG=""
fi

if [ "$DO_CLASS_LABELS_FOR_PROMPT" = "true" ]; then
    CLASS_LABELS_FLAG="--do-class-labels-for-prompt"
else
    CLASS_LABELS_FLAG=""
fi

if [ "$DO_PROP_SAMPLING_FOR_PROMPT" = "true" ]; then
    PROP_SAMPLING_FLAG="--do-prop-sampling-for-prompt"
else
    PROP_SAMPLING_FLAG=""
fi

# generate samples
rm -rf $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES
mkdir -p $OUTPUT_DIR_SCENES/1234
mkdir -p $OUTPUT_DIR_SCENES/3456
mkdir -p $OUTPUT_DIR_SCENES/5678
PYTHONPATH=. xvfb-run -a python ./src/pipeline.py --use-gpu --pth-output=$OUTPUT_DIR_SCENES --env="stanley" --room-type=$ROOM_TYPE --model-id=$MODEL_ID --n-test-scenes=$N_TEST_SCENES --n-bon-sgllm=$BON_LLM $ICL_FLAG $CLASS_LABELS_FLAG $PROP_SAMPLING_FLAG --icl-k=$ICL_K --do-full-scenes

# compute metrics
rm -rf $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ
mkdir -p $OUTPUT_DIR_VIZ/1234
mkdir -p $OUTPUT_DIR_VIZ/3456
mkdir -p $OUTPUT_DIR_VIZ/5678
PYTHONPATH=. xvfb-run -a python ./src/eval.py --pth-input=$OUTPUT_DIR_SCENES --pth-output=$OUTPUT_DIR_VIZ --env="stanley" --room-type=$ROOM_TYPE --do-metrics --n-test-scenes=$N_TEST_SCENES --is-full-scene
