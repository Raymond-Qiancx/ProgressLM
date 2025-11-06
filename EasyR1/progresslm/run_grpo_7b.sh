#!/bin/bash

set -x
# set -euxo pipefail

# ===== 🟢 路径设置 =====
# 7B 模型路径需要比 3B 更小的 batch 配置
MODEL_PATH="/projects/b1222/userdata/jianshu/chengxuan/saved/saved_results/progresslm/models/qwen25vl_7b_sft"

# 自动生成时间戳
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

# ===== 🟢 wandb 设置 =====
export WANDB_API_KEY="a055c70d645ef2b98309254662429133b73ac639"
export WANDB_PROJECT="progresslm_grpo"
export WANDB_RUN_GROUP="qwen2_5_vl_7b_progresslm_grpo"
export WANDB_NAME="visual_demo_qwen2p5vl7b_${TIMESTAMP}"
export WANDB_MODE="online"
export WANDB_DIR="/projects/b1222/userdata/jianshu/code/EasyR1/progresslm/wandb_logs"

unset WANDB_RUN_ID
unset WANDB_RESUME

echo "WANDB 环境变量："
env | grep WANDB

# ===== 🟢 训练配置 =====
# 7B 模型显存占用更高，适当减小 batch 相关配置
CHECKPOINT_DIR="/projects/p32958/chengxuan/models/easyr1_ckpt/newest_35k_7b_${TIMESTAMP}"

python3 -m verl.trainer.main \
  config=progresslm/configs/visual_demo_grpo.yaml \
  worker.actor.fsdp.torch_dtype=bfloat16 \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.model.tokenizer_path="${MODEL_PATH}" \
  worker.actor.global_batch_size=8 \
  data.rollout_batch_size=8 \
  worker.rollout.n=4 \
  worker.rollout.limit_images=24 \
  worker.rollout.max_num_batched_tokens=12544 \
  worker.rollout.gpu_memory_utilization=0.75 \
  trainer.save_checkpoint_path="${CHECKPOINT_DIR}" \
  trainer.experiment_name="qwen2_5vl7b_grpo_${TIMESTAMP}"
