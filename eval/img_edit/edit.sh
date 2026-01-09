#!/bin/bash

# ================================================
# Qwen-Image-Edit Multi-GPU Processing Script
# ================================================

# Default values
JSONL_FILE="/projects/p32958/chengxuan/data/image_edit/img_edit_1.jsonl"
MODEL_PATH="/projects/p32958/chengxuan/models/Qwen-Image-Edit"
IMAGE_DIR="/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/images"
SAVE_DIR="/projects/p32958/chengxuan/results/progresslm/negative/image"
CHECKPOINT_DIR="/projects/p32958/chengxuan/results/progresslm/negative/ckpt/edit_1"
NUM_GPUS=4
GPU_IDS="0,1,2,3"
MAX_RETRIES=2
CHECKPOINT=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --jsonl)
            JSONL_FILE="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --image-dir)
            IMAGE_DIR="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --no-checkpoint)
            CHECKPOINT=false
            shift
            ;;
        --help)
            echo "Usage: $0 --jsonl <file> [options]"
            echo ""
            echo "Required:"
            echo "  --jsonl <file>       Input JSONL file"
            echo ""
            echo "Optional:"
            echo "  --model-path <path>  Model path (default: $MODEL_PATH)"
            echo "  --image-dir <path>   Image directory (default: $IMAGE_DIR)"
            echo "  --save-dir <path>    Save directory (default: $SAVE_DIR)"
            echo "  --num-gpus <n>       Number of GPUs (default: $NUM_GPUS)"
            echo "  --gpu-ids <ids>      GPU IDs, e.g., '0,1,2,3' (default: $GPU_IDS)"
            echo "  --no-checkpoint      Disable checkpoint recovery"
            echo ""
            echo "Example:"
            echo "  $0 --jsonl input.jsonl --save-dir ./outputs --num-gpus 4"
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (use --help for usage)"
            exit 1
            ;;
    esac
done

# Check if JSONL file exists
if [ ! -f "$JSONL_FILE" ]; then
    echo "Error: JSONL file not found: $JSONL_FILE"
    exit 1
fi

# Set environment
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# Disable NCCL timeout for long-running tasks (each image takes ~2 minutes)
# Default timeout is 10 minutes which may not be enough for slow GPUs
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=0  # 0 = no timeout

# Count tasks
TASK_COUNT=$(wc -l < "$JSONL_FILE")

# Print configuration
clear
echo "============================================================"
echo "         Qwen-Image-Edit Multi-GPU Processing              "
echo "============================================================"
echo ""
echo "Configuration:"
echo "  • JSONL File:    $JSONL_FILE ($TASK_COUNT tasks)"
echo "  • Model Path:    $MODEL_PATH"
echo "  • Image Dir:     $IMAGE_DIR"
echo "  • Save Dir:      $SAVE_DIR"
echo "  • GPUs:          $NUM_GPUS GPUs [$GPU_IDS]"
echo "  • Checkpoint:    $CHECKPOINT"
echo "  • Tasks/GPU:     ~$((TASK_COUNT / NUM_GPUS))"
echo ""
echo "============================================================"
echo ""

# Create directories
mkdir -p "$SAVE_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Build command
CMD="accelerate launch --num_processes=$NUM_GPUS --mixed_precision=bf16 edit_batch.py"
CMD="$CMD --jsonl $JSONL_FILE"
CMD="$CMD --model-path $MODEL_PATH"
CMD="$CMD --image-dir $IMAGE_DIR"
CMD="$CMD --save-dir $SAVE_DIR"
CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR"
CMD="$CMD --max-retries $MAX_RETRIES"

if [ "$CHECKPOINT" = false ]; then
    CMD="$CMD --no-checkpoint"
fi

# Start timer
START_TIME=$(date +%s)
echo "Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Run processing
$CMD

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "============================================================"
echo "Finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total duration: ${MINUTES}m ${SECONDS}s"
echo ""

# Check results
if [ -f "$SAVE_DIR/all_results.json" ]; then
    echo "Results saved to: $SAVE_DIR/all_results.json"
    echo "Edited images saved to: $SAVE_DIR/"
else
    echo "Warning: Results file not found"
fi

echo "============================================================"