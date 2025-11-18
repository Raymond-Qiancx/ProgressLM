#!/bin/bash

#####################################################################
# Text Nega (Negative Sample) Progress Estimation Script - 72B Model
#
# This script runs progress estimation on Text Nega dataset using
# Qwen2-VL-72B model with model parallelism across multiple GPUs.
# Uses FRM's cheat prompt system with ground-truth for negative samples.
#
# Expected JSONL format (NEGATIVE SAMPLE VERSION):
# {
#   "id": "h5_agilex_3rgb/10_packplate_2/2024_09_28-17_07_01-172863393748093664.00",
#   "task_goal": "with both arms placing two cups into a rack",
#   "text_demo": ["[left] move towards the green cup...", ...],
#   "raw_task_goal": "with both arms placing two plates into a rack",
#   "raw_text_demo": ["[left] move towards the green plate...", ...],
#   "total_steps": "10",
#   "stage_to_estimate": "camera_front_0062.jpg",
#   "closest_idx": "n/a",
#   "progress_score": "n/a",
#   "rank": 0,
#   "data_source": "h5_agilex_3rgb"
# }
#####################################################################

# ======================== Configuration ========================

# Model configuration - 72B MODEL
# Support environment variable override: export MODEL_PATH=/custom/path
MODEL_PATH="${MODEL_PATH:-/projects/p32958/chengxuan/models/Qwen2.5-VL-72B-Instruct}"

# Dataset configuration
# Support environment variable override: export DATASET_PATH=/custom/dataset.jsonl
DATASET_PATH="${DATASET_PATH:-/projects/p32958/chengxuan/ProgressLM/data/sft_data/text_nega_new/new_text_nega_merged_with_rank.jsonl}"

IMAGE_ROOT="${IMAGE_ROOT:-/projects/p32958/chengxuan/data/images}"

# Output configuration
# Support environment variable override: export OUTPUT_DIR=/custom/output
OUTPUT_DIR="${OUTPUT_DIR:-/projects/p32958/chengxuan/results/text_nega_think}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TIMESTAMPED_DIR="${OUTPUT_DIR}/text_nega_72b-${TIMESTAMP}"
OUTPUT_FILE="${TIMESTAMPED_DIR}/text_nega_think_72b_${TIMESTAMP}.jsonl"
LOG_FILE="${TIMESTAMPED_DIR}/text_nega_72b_${TIMESTAMP}.log"

# GPU configuration
# Support environment variable override: export GPU_IDS="0,1" BATCH_SIZE=80
GPU_IDS="${GPU_IDS:-0,1,2,3}"  # Comma-separated GPU IDs to use (72B requires multiple GPUs for model parallelism)
BATCH_SIZE="${BATCH_SIZE:-40}"  # Batch size (reduced for 72B model due to high memory requirements)

# Inference configuration
NUM_INFERENCES="${NUM_INFERENCES:-1}"  # Number of inferences per sample (data expansion factor)

# Model parameters
TEMPERATURE="${TEMPERATURE:-0.6}"  # Higher temperature for diversity across multiple inferences
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-40000}"  # Increased from 30000 to 40000 for longer CoT reasoning chains
MIN_PIXELS="${MIN_PIXELS:-$((1280*28*28))}"
MAX_PIXELS="${MAX_PIXELS:-$((5120*28*28))}"

# Processing parameters
LIMIT="${LIMIT:--1}"  # Limit samples to process after expansion (-1 for all)

# Misc
VERBOSE="${VERBOSE:-false}"  # Set to true for detailed output

# ======================== Auto Configuration ========================

echo "======================================================================"
echo "Text Nega Progress Estimation - 72B Model (Model Parallelism + FRM Cheat)"
echo "======================================================================"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_FILE"
echo "GPUs: $GPU_IDS"
echo "Batch Size: $BATCH_SIZE (optimized for 72B)"
echo "Inferences per Sample: $NUM_INFERENCES"
echo "Model: 72B (Single Process with Model Parallelism)"
echo "Mode: Negative Sample Training"
echo "======================================================================"

# ======================== Validation ========================

# Check if dataset path is provided
if [ -z "$DATASET_PATH" ]; then
    echo "Error: DATASET_PATH is not set!"
    echo "Please set DATASET_PATH to your Text Nega dataset JSONL file."
    exit 1
fi

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    exit 1
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$TIMESTAMPED_DIR"

# ======================== Run Inference ========================

# Set CUDA visible devices to all GPUs (model will be distributed across them)
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FRM_DIR="$PROJECT_DIR/frm"

# Change to frm directory
cd "$FRM_DIR" || exit 1

# Build command - using run_text_nega.py for negative samples
CMD="python run_text_nega.py \
    --model-path $MODEL_PATH \
    --dataset-path $DATASET_PATH \
    --output-file $OUTPUT_FILE \
    --batch-size $BATCH_SIZE \
    --num-inferences $NUM_INFERENCES \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-new-tokens $MAX_NEW_TOKENS \
    --min-pixels $MIN_PIXELS \
    --max-pixels $MAX_PIXELS"

# Add image root if specified
if [ -n "$IMAGE_ROOT" ] && [ "$IMAGE_ROOT" != "/path/to/your/images" ]; then
    CMD="$CMD --image-root $IMAGE_ROOT"
fi

# Add limit if specified
if [ $LIMIT -gt 0 ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Add verbose flag if enabled
if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

echo "Starting batch inference with 72B model for negative samples..."
echo ""

# Execute command with logging
$CMD 2>&1 | tee "$LOG_FILE"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Completed | Results: $OUTPUT_FILE"
    echo "======================================================================"

    # Display summary if exists
    SUMMARY_FILE="${OUTPUT_FILE%.jsonl}_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo ""
        echo "Summary:"
        cat "$SUMMARY_FILE"
        echo ""
    fi
else
    echo ""
    echo "======================================================================"
    echo "✗ Failed (exit code $EXIT_CODE) | Log: $LOG_FILE"
    echo "======================================================================"
    exit $EXIT_CODE
fi
