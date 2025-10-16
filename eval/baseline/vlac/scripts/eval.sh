#!/bin/bash
################################################################################
# VLAC Image Sequence Evaluation Script
#
# 这个脚本用于一键运行VLAC模型对图像序列进行评估
# 所有配置都在这个脚本中定义，只需修改下面的变量即可
#
# 使用方法:
#   bash scripts/eval.sh
################################################################################

# ============================================================================
# 配置区域 - 根据你的实际情况修改以下变量
# ============================================================================

# ---------- 模型配置 ----------
# VLAC模型路径（必需）
# 从 https://huggingface.co/InternRobotics/VLAC 下载
MODEL_PATH="/path/to/your/VLAC/model"

# 运行设备（cuda:0 表示第一块GPU，cpu 表示使用CPU）
DEVICE="cuda:0"

# 模型类型（默认 internvl2，一般不需要修改）
MODEL_TYPE="internvl2"

# 采样温度（0.1-1.0，越低越确定性）
TEMPERATURE=0.5

# Top-k 采样
TOP_K=1


# ---------- 数据配置 ----------
# 测试图像目录（必需）
DATA_DIR="/path/to/your/test/images"

# 参考轨迹图像目录（可选，如果不需要参考轨迹，留空即可）
# REF_DIR=""  # 不使用参考轨迹
REF_DIR="/path/to/your/reference/images"  # 使用参考轨迹

# 任务描述（必需，描述这个轨迹要完成什么任务）
TASK_DESCRIPTION="Pick up the bowl and place it in the box"

# 图像文件格式（支持多种格式，用逗号分隔）
IMAGE_PATTERN="*.jpg,*.png"

# 最大加载图像数量（留空表示加载全部）
MAX_IMAGES=""  # 留空加载全部
# MAX_IMAGES=100  # 只加载前100张


# ---------- 评估参数配置 ----------
# 批量推理大小（根据显存调整，显存不足时减小这个值）
BATCH_NUM=5

# 参考图像采样数量（从参考轨迹中均匀采样的图像数）
REF_NUM=6

# 帧跳跃步长（skip=1表示每帧都评估，skip=5表示每5帧评估一次）
# 增大skip可以加速评估，但会降低精度
SKIP=1

# 是否启用rich模式（输出小数值，更精确）
RICH_MODE=true  # true 或 false

# 是否启用思维链推理（会更慢但更准确）
THINK_MODE=false  # true 或 false

# 是否反向评估（用于VROC指标计算）
REVERSE_EVAL=false  # true 或 false


# ---------- 输出配置 ----------
# 结果输出目录
OUTPUT_DIR="./results"

# 输出文件名（留空则自动生成带时间戳的文件名）
OUTPUT_NAME=""  # 留空自动生成
# OUTPUT_NAME="my_evaluation_result"  # 自定义输出文件名


# ============================================================================
# 以下是脚本执行逻辑，一般不需要修改
# ============================================================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}VLAC Evaluation Script${NC}"
echo -e "${BLUE}=================================${NC}"

# 检查必需参数
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}[ERROR] Model path does not exist: $MODEL_PATH${NC}"
    echo -e "${YELLOW}Please download the model from: https://huggingface.co/InternRobotics/VLAC${NC}"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}[ERROR] Data directory does not exist: $DATA_DIR${NC}"
    exit 1
fi

if [ -z "$TASK_DESCRIPTION" ]; then
    echo -e "${RED}[ERROR] Task description is empty${NC}"
    exit 1
fi

# 构建命令行参数
CMD="python $PROJECT_ROOT/run_eval.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --task \"$TASK_DESCRIPTION\" \
    --device $DEVICE \
    --model_type $MODEL_TYPE \
    --temperature $TEMPERATURE \
    --top_k $TOP_K \
    --batch_num $BATCH_NUM \
    --ref_num $REF_NUM \
    --skip $SKIP \
    --image_pattern \"$IMAGE_PATTERN\" \
    --output_dir $OUTPUT_DIR"

# 添加可选参数
if [ -n "$REF_DIR" ] && [ -d "$REF_DIR" ]; then
    CMD="$CMD --ref_dir $REF_DIR"
fi

if [ -n "$MAX_IMAGES" ]; then
    CMD="$CMD --max_images $MAX_IMAGES"
fi

if [ -n "$OUTPUT_NAME" ]; then
    CMD="$CMD --output_name $OUTPUT_NAME"
fi

if [ "$RICH_MODE" = true ]; then
    CMD="$CMD --rich"
fi

if [ "$THINK_MODE" = true ]; then
    CMD="$CMD --think"
fi

if [ "$REVERSE_EVAL" = true ]; then
    CMD="$CMD --reverse_eval"
fi

# 显示配置信息
echo -e "${GREEN}[INFO] Configuration:${NC}"
echo "  Model Path: $MODEL_PATH"
echo "  Data Dir: $DATA_DIR"
if [ -n "$REF_DIR" ] && [ -d "$REF_DIR" ]; then
    echo "  Reference Dir: $REF_DIR"
fi
echo "  Task: $TASK_DESCRIPTION"
echo "  Device: $DEVICE"
echo "  Batch Size: $BATCH_NUM"
echo "  Skip: $SKIP"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 执行评估
echo -e "${GREEN}[INFO] Starting evaluation...${NC}"
echo -e "${YELLOW}Command: $CMD${NC}"
echo ""

eval $CMD

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================${NC}"
    echo -e "${GREEN}Evaluation completed successfully!${NC}"
    echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
    echo -e "${GREEN}=================================${NC}"
else
    echo ""
    echo -e "${RED}=================================${NC}"
    echo -e "${RED}Evaluation failed!${NC}"
    echo -e "${RED}Please check the error messages above${NC}"
    echo -e "${RED}=================================${NC}"
    exit 1
fi
