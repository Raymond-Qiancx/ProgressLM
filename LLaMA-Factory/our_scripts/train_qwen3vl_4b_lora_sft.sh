#!/bin/bash

################################################################################
# Qwen3-VL-4B LoRA SFT Training Script
#
# This script trains Qwen3-VL-4B-Instruct model using LoRA for supervised
# fine-tuning (SFT) on multimodal (vision-language) tasks.
#
# Usage:
#   Single GPU:  bash train_qwen3vl_4b_lora_sft.sh
#   Multi GPU:   CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_qwen3vl_4b_lora_sft.sh
################################################################################

set -e  # Exit on error

# ==================== GPU Configuration ====================
# Set which GPUs to use (0,1,2,3 means using 4 GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ==================== W&B Configuration ====================
export WANDB_API_KEY=

# ==================== Configuration ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${SCRIPT_DIR}/qwen3vl_4b_lora_sft.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "Qwen3-VL-4B LoRA SFT Training"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Config File:  $CONFIG_FILE"
echo "Current Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# ==================== Environment Info ====================
echo ""
echo "Environment Information:"
echo "  Python:      $(which python)"
echo "  Python Ver:  $(python --version 2>&1)"
echo "  CUDA Devices: ${CUDA_VISIBLE_DEVICES:-all}"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU Info:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | awk '{print "    GPU "$0}'
fi
echo ""

# ==================== Change to Project Root ====================
cd "$PROJECT_ROOT"

# ==================== Detect Multi-GPU Setup ====================
# Count available GPUs
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Count commas in CUDA_VISIBLE_DEVICES and add 1
    GPU_COUNT=$(($(echo "$CUDA_VISIBLE_DEVICES" | tr -cd ',' | wc -c) + 1))
else
    # Try to detect from nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    else
        GPU_COUNT=1
    fi
fi

echo "Detected $GPU_COUNT GPU(s)"

# ==================== Training Command ====================
if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Starting Multi-GPU Training with FORCE_TORCHRUN..."
    echo ""
    FORCE_TORCHRUN=1 llamafactory-cli train "$CONFIG_FILE"
else
    echo "Starting Single-GPU Training..."
    echo ""
    llamafactory-cli train "$CONFIG_FILE"
fi

# ==================== Training Complete ====================
echo ""
echo "=========================================="
echo "Training completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Check training logs and loss curves"
echo "  2. Evaluate the model performance"
echo "  3. Merge LoRA weights if needed:"
echo "     llamafactory-cli export examples/merge_lora/qwen3vl_lora_sft.yaml"
echo ""
