#!/bin/bash

set -x

# ===== 🟢 路径设置 =====
MODEL_PATH="/projects/p32958/Results/full_model/qwen25vl_7b_sft"
DATA_FILE="/projects/p32958/chengxuan/ProgressLM/data/train/rl/new/new_rl_sampled_10k_ready_for_training.jsonl"

# 固定时间戳以便复现实验
# TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
TIMESTAMP="20251109-1942"

# ===== 🟢 wandb 设置 =====
export WANDB_API_KEY="ac3c3d795e02ca8885235198ec9a222725622805"
export WANDB_PROJECT="progresslm_grpo_7b"
export WANDB_RUN_GROUP="qwen2_5_vl_7b_progresslm_grpo"
export WANDB_NAME="visual_demo_qwen2p5vl7b_10k_${TIMESTAMP}"
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

unset WANDB_RUN_ID
unset WANDB_RESUME

echo "WANDB 环境变量："
env | grep WANDB

# ===== 🟢 训练配置 =====
CHECKPOINT_DIR="/projects/p32958/Results/rl_ckpt/qwen25_vl_7b_rl/newest_sampled_10k_7b_${TIMESTAMP}"

python3 -m verl.trainer.main \
  config=progresslm/configs/visual_demo_grpo.yaml \
  worker.actor.fsdp.torch_dtype=bfloat16 \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.model.tokenizer_path="${MODEL_PATH}" \
  worker.actor.global_batch_size=8 \
  data.rollout_batch_size=8 \
  data.train_files="${DATA_FILE}" \
  data.val_files="${DATA_FILE}" \
  worker.rollout.n=4 \
  worker.rollout.limit_images=24 \
  worker.rollout.max_num_batched_tokens=30000 \
  worker.rollout.gpu_memory_utilization=0.7 \
  trainer.save_checkpoint_path="${CHECKPOINT_DIR}" \
  trainer.experiment_name="qwen2_5vl7b_grpo_10k_${TIMESTAMP}"
