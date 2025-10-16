#!/bin/bash
################################################################################
# VLAC 评估示例脚本
#
# 这是一个配置好的示例，展示如何使用VLAC评估图像序列
# 复制这个文件并修改配置来适配你的数据
################################################################################

# ============================================================================
# 配置示例
# ============================================================================

# ---------- 模型配置 ----------
MODEL_PATH="./models/VLAC"  # 模型下载到这里
DEVICE="cuda:0"              # 使用第一块GPU

# ---------- 数据配置 ----------
# 示例1: 评估单个轨迹
DATA_DIR="./data/trajectory_001"
REF_DIR="./data/reference"   # 可选的参考轨迹
TASK_DESCRIPTION="Pick up the bowl and place it back in the white storage box"

# 示例2: 评估不同任务（注释掉示例1，取消注释这里）
# DATA_DIR="./data/scoop_rice"
# REF_DIR="./data/scoop_rice_reference"
# TASK_DESCRIPTION="Scoop the rice into the rice cooker"

# ---------- 评估参数 ----------
BATCH_NUM=5      # 批量大小
REF_NUM=6        # 参考图像数量
SKIP=1           # 每1帧评估一次（精确模式）
# SKIP=5         # 每5帧评估一次（快速模式）

RICH_MODE=true   # 启用精确小数输出
THINK_MODE=false # 不启用思维链（更快）

# ---------- 输出配置 ----------
OUTPUT_DIR="./results"
OUTPUT_NAME="trajectory_001_eval"

# ============================================================================
# 执行评估
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

python $PROJECT_ROOT/run_eval.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --ref_dir $REF_DIR \
    --task "$TASK_DESCRIPTION" \
    --device $DEVICE \
    --batch_num $BATCH_NUM \
    --ref_num $REF_NUM \
    --skip $SKIP \
    --output_dir $OUTPUT_DIR \
    --output_name $OUTPUT_NAME \
    $([ "$RICH_MODE" = true ] && echo "--rich") \
    $([ "$THINK_MODE" = true ] && echo "--think")
