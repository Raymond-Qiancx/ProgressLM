#!/bin/bash

set -x
# set -euxo pipefail

# ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/projects/b1222/userdata/jianshu/chengxuan/saved/saved_results/progresslm/models/3b_sft_qwen25vl_4epoch}"

# === 🟢 确保 wandb 可用 ===
export WANDB_API_KEY="a055c70d645ef2b98309254662429133b73ac639"   # 可在 https://wandb.ai/settings 获取
export WANDB_PROJECT="easy_r1"
# export WANDB_ENTITY="你的wandb用户名或团队名"  # 可选
export WANDB_RUN_GROUP="qwen2_5_vl_3b_progresslm_grpo"
export WANDB_NAME="$(date +%Y%m%d-%H%M%S)"
export WANDB_MODE="online"    # 确保不是 'offline' 或 'disabled'

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
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
CHECKPOINT_ROOT="/projects/p32958/chengxuan/models/easyr1_ckpt/35k_${TIMESTAMP}"

python3 -m verl.trainer.main \
  config=progresslm/configs/visual_demo_grpo.yaml \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.model.tokenizer_path="${MODEL_PATH}" \
  trainer.save_checkpoint_path="${CHECKPOINT_ROOT}" \
  trainer.experiment_name="visual_demo_qwen2p5vl3b_${TIMESTAMP}"
