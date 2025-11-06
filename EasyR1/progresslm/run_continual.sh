#!/bin/bash

set -x
# set -euxo pipefail

# ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="/projects/b1222/userdata/jianshu/chengxuan/saved/saved_results/progresslm/models/qwen25vl_3b_small_final"

# === 🟢 确保 wandb 可用 ===
export WANDB_API_KEY="a055c70d645ef2b98309254662429133b73ac639"
export WANDB_PROJECT="progresslm_grpo"
export WANDB_RUN_GROUP="qwen2_5_vl_3b_progresslm_grpo"
export WANDB_NAME="visual_demo_qwen2p5vl3b_20251029-195831"
export WANDB_RUN_ID="xfhwpazk"
export WANDB_RESUME="must"
export WANDB_MODE="online"
export WANDB_DIR="/projects/b1222/userdata/jianshu/code/EasyR1/progresslm/wandb_logs"


# （可选）指定保存路径
export WANDB_DIR="/projects/b1222/userdata/jianshu/code/EasyR1/progresslm/wandb_logs"

echo "WANDB 环境变量："
env | grep WANDB

# export RAY_DISABLE_DASHBOARD=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export TORCH_NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# ulimit -n 65536
TIMESTAMP="20251031-004732"
# CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/projects/p32958/chengxuan/models/easyr1_ckpt/35k_${TIMESTAMP}}"



python3 -m verl.trainer.main \
  config=progresslm/configs/visual_demo_grpo.yaml \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.model.tokenizer_path="${MODEL_PATH}" \
  trainer.save_checkpoint_path="/projects/p32958/chengxuan/models/easyr1_ckpt/newest_35k_20251031-004732" \
  trainer.experiment_name="visual_demo_qwen2p5vl3b_${TIMESTAMP}"
