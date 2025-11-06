#!/bin/bash

set -x
# set -euxo pipefail

# ===== 🟢 路径设置 =====
# 修改为你新的模型路径
MODEL_PATH="/projects/b1222/userdata/jianshu/chengxuan/saved/saved_results/progresslm/models/qwen25vl_3b_no_coin_final"

# 自动生成时间戳
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

# ===== 🟢 wandb 设置 =====
export WANDB_API_KEY="a055c70d645ef2b98309254662429133b73ac639"
export WANDB_PROJECT="progresslm_grpo"
export WANDB_RUN_GROUP="qwen2_5_vl_3b_progresslm_grpo_no_coin"
export WANDB_NAME="visual_demo_qwen2p5vl3b_no_coin_${TIMESTAMP}"
export WANDB_MODE="online"
export WANDB_DIR="/projects/b1222/userdata/jianshu/code/EasyR1/progresslm/wandb_logs"

# 注意：去掉 resume，让 wandb 新开一条记录
unset WANDB_RUN_ID
unset WANDB_RESUME

echo "WANDB 环境变量："
env | grep WANDB

# ===== 🟢 训练配置 =====
CHECKPOINT_DIR="/projects/p32958/chengxuan/models/easyr1_ckpt/no-coin-3b_35k_${TIMESTAMP}"

python3 -m verl.trainer.main \
  config=progresslm/configs/visual_demo_grpo.yaml \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.model.tokenizer_path="${MODEL_PATH}" \
  trainer.save_checkpoint_path="${CHECKPOINT_DIR}" \
  trainer.experiment_name="qwen2_5vl3b_grpo_no_coin_${TIMESTAMP}"
