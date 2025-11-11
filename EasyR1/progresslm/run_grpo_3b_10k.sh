#!/bin/bash

set -x
# set -euxo pipefail

# ===== 🟢 路径设置 =====
# 修改为你新的模型路径
MODEL_PATH="/projects/p32958/Results/full_model/qwen25vl_3b_sft"
DATA_FILE="/projects/p32958/chengxuan/ProgressLM/data/train/rl/new/new_rl_sampled_10k_ready_for_training.jsonl"

# 自动生成时间戳
# TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
TIMESTAMP="20251109-181118"

# ===== 🟢 wandb 设置 =====
export WANDB_API_KEY="ac3c3d795e02ca8885235198ec9a222725622805"
export WANDB_PROJECT="progresslm_grpo_new"
export WANDB_RUN_GROUP="qwen2_5_vl_3b_progresslm"
export WANDB_NAME="visual_demo_qwen25vl3b_10k_${TIMESTAMP}"
export WANDB_MODE="online"
export WANDB_DIR="/projects/p32876/Results/wandb_logs"

# ===== 🔴 统一缓存目录设置（避免磁盘配额超限） =====
CACHE_ROOT="/gpfs/projects/p32876/chengxuan/.cache"

# HuggingFace 缓存
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export HF_HUB_CACHE="$CACHE_ROOT/huggingface/hub"

# PyTorch 缓存
export TORCH_HOME="$CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch/extensions"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torch/inductor"

# Triton 编译缓存
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"

# Ray 缓存和临时文件（使用超短路径避免 Unix socket 107 字节限制）
export RAY_TMPDIR="/gpfs/projects/p32876/.r/tmp"
export RAY_SESSION_DIR="/gpfs/projects/p32876/.r/session"
export RAY_LOG_DIR="/gpfs/projects/p32876/.r/logs"

# 创建Ray目录
mkdir -p "$RAY_TMPDIR" "$RAY_SESSION_DIR" "$RAY_LOG_DIR"

# Python 字节码缓存
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"

# XDG 缓存标准
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

# 通用临时文件目录
export TMPDIR="$CACHE_ROOT/tmp"
export TEMP="$CACHE_ROOT/tmp"
export TMP="$CACHE_ROOT/tmp"

# 创建所有目录
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"
mkdir -p "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$RAY_TMPDIR" "$RAY_SESSION_DIR" "$RAY_LOG_DIR"
mkdir -p "$PYTHONPYCACHEPREFIX" "$XDG_CACHE_HOME" "$TMPDIR"

# 注意：去掉 resume，让 wandb 新开一条记录
unset WANDB_RUN_ID
unset WANDB_RESUME

echo "WANDB 环境变量："
env | grep WANDB

# ===== 🟢 训练配置 =====
# CHECKPOINT_DIR="/projects/p32958/Results/rl_ckpt/qwen25vl_3b_rl_sampled_10k_${TIMESTAMP}"
CHECKPOINT_DIR="/projects/p32958/Results/rl_ckpt/qwen25vl_3b_rl_sampled_10k_20251109-181118"

python3 -m verl.trainer.main \
  config=progresslm/configs/visual_demo_grpo.yaml \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.model.tokenizer_path="${MODEL_PATH}" \
  data.train_files="${DATA_FILE}" \
  data.val_files="${DATA_FILE}" \
  trainer.save_checkpoint_path="${CHECKPOINT_DIR}" \
  trainer.experiment_name="qwen2_5vl3b_grpo_10k_${TIMESTAMP}"
