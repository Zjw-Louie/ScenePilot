# ReSpace: training script for Stage 1 (SFT) on Qwen-2.5-1.5B model
set -euo pipefail

# 项目根目录（确保能 import src.*）
PROJECT_ROOT="/home2/zhangjiawei/respace"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# if using single GPU training, remove --multi_gpu flag

# set number of GPU tasks here:
export N_TASKS=4

# adjust these params if needed
export JOB_NUM_NODES=1
export NODEID=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled
export WANDB_DISABLED=true

# add --use-logfile if you want to log all stdout into a logfile
# add --multi-gpu directly after 'accelerate launch' if using > 1 GPU

# SFT (full / bedroom)
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="$(uuidgen)" --run-id="apr23-qwen1.5B-full-bdrm" --use-cached-dataset --use-gpu --env=sherlock --epochs=150 --test-bs=4 --llm="qwen-2.5-1.5B" --room-type="bedroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --dvc-batch-size=4 --gas-steps=8

# SFT (full / livingroom)
# accelerate launch --multi_gpu --debug --num_processes="${N_TASKS}" --num_machines="${JOB_NUM_NODES}" --mixed_precision bf16 --machine_rank="${NODEID}" --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --dynamo_backend=no src/main.py --jid="$(uuidgen)" --run-id="apr23-qwen1.5B-full-lvngrm" --use-cached-dataset --use-gpu --env=sherlock --epochs=150 --test-bs=4 --llm="qwen-2.5-1.5B" --room-type="livingroom" --use-wandb --do-augm --lambda-instr-exp=0.0 --dvc-batch-size=4 --gas-steps=8

# SFT (full / all)
accelerate launch --debug \
  --multi-gpu \
  --num_processes="${N_TASKS}" --num_machines="${JOB_NUM_NODES}" \
  --mixed_precision bf16 \
  --machine_rank="${NODEID}" \
  --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" \
  --dynamo_backend=no \
  -m src.main \
  --jid="$(uuidgen)" \
  --run-id="apr24-qwen4B-full-all" \
  --use-cached-dataset --use-gpu --env=".env" \
  --epochs=150 --test-bs=4 \
  --llm="qwen-3-4B" --room-type="all" \
  --do-augm --lambda-instr-exp=0.0 \
  --dvc-batch-size=4 --gas-steps=8