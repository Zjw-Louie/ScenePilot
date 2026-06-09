#!/bin/bash
set -euo pipefail

# Qwen3-VL-8B multimodal SFT for v13 SceneRepairPlan
# 输入：diag/top/annotated_top 三张图 + 当前 scene JSON
# 输出：SceneRepairPlan JSON

MODEL_NAME="/home2/zhangjiawei/models/Qwen3-VL-8B-Instruct"

export PYTHONPATH=src:${PYTHONPATH:-}

# 8B + 三图 + 长 scene JSON，先用保守 batch
GLOBAL_BATCH_SIZE=32
BATCH_PER_DEVICE=1
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

DATA_PATH="/home2/zhangjiawei/respace/training_data/v13_sft_multimodal_qwen3vl.json"
IMAGE_FOLDER="/home2/zhangjiawei/respace/training_data/growth2_frames"

OUTPUT_DIR="output/v13_qwen3vl8b_scene_repair_lora"

deepspeed src/train/train_sft.py \
    --use_liger_kernel True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3.json \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((128 * 32 * 32)) \
    --image_max_pixels $((768 * 32 * 32)) \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing False \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4