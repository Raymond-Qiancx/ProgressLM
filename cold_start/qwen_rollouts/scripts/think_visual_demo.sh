#!/bin/bash

#####################################################################
# Visual Demo Progress Estimation Script
#
# This script runs progress estimation on Visual Demo dataset using
# Qwen2-VL model with distributed GPU support.
#
# Expected JSONL format (NEW VERSION):
# {
#   "id": "h5_tienkung_xsens_1rgb/brick_piled_then_press_thrice/2024-10-17-10-53-16",
#   "task_goal": "Put the blue block next to the purple block in front.",
#   "visual_demo": ["camera_top_0000.jpg", "camera_top_0041.jpg", "camera_top_0068.jpg", "camera_top_0191.jpg", "camera_top_0394.jpg"],
#   "total_steps": "4",
#   "stage_to_estimate": ["camera_top_0013.jpg"],
#   "closest_idx": "1",
#   "delta": "+7%",
#   "progress_score": "8%",
#   "data_source": "robomind_h5_tienkung_xsens_1rgb"
# }
#####################################################################

# ======================== Configuration ========================

# Model configuration
# Support environment variable override: export MODEL_PATH=/custom/path
MODEL_PATH="${MODEL_PATH:-/projects/b1222/userdata/jianshu/chengxuan/saved/models/Qwen2.5-VL-32B-Instruct}"

# Dataset configuration
# Support environment variable override: export DATASET_PATH=/custom/dataset.jsonl
DATASET_PATH="${DATASET_PATH:-/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/train/visual_demo/visual_h5_franka_3rgb_sft.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/images}"  # Optional: root directory for relative image paths

# Output configuration
# Support environment variable override: export OUTPUT_DIR=/custom/output
OUTPUT_DIR="${OUTPUT_DIR:-/projects/b1222/userdata/jianshu/chengxuan/saved/saved_results/progresslm/visual_think}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TIMESTAMPED_DIR="${OUTPUT_DIR}/visual_demo-${TIMESTAMP}"
OUTPUT_FILE="${TIMESTAMPED_DIR}/visual_demo_results_${TIMESTAMP}.jsonl"
LOG_FILE="${TIMESTAMPED_DIR}/visual_demo_${TIMESTAMP}.log"

# GPU configuration
# Support environment variable override: export GPU_IDS="0,1" BATCH_SIZE=4
GPU_IDS="${GPU_IDS:-0,1,2,3}"  # Comma-separated GPU IDs to use
BATCH_SIZE="${BATCH_SIZE:-2}"  # Batch size per GPU (adjust based on VRAM and image count)

# Inference configuration
NUM_INFERENCES="${NUM_INFERENCES:-1}"  # Number of inferences per sample (data expansion factor)

# Model parameters
TEMPERATURE="${TEMPERATURE:-0.6}"  # Higher temperature for diversity across multiple inferences
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-40000}"  # Increased from 5120 to 40000 for longer CoT reasoning chains
MIN_PIXELS="${MIN_PIXELS:-$((1280*28*28))}"
MAX_PIXELS="${MAX_PIXELS:-$((5120*28*28))}"

# Processing parameters
LIMIT="${LIMIT:--1}"  # Limit samples to process after expansion (-1 for all)

# Misc
VERBOSE="${VERBOSE:-false}"  # Set to true for detailed output

# ======================== Auto Configuration ========================

echo "======================================================================"
echo "Visual Demo Progress Estimation - Batch Inference"
echo "======================================================================"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_FILE"
echo "GPUs: $GPU_IDS"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Inferences per Sample: $NUM_INFERENCES"
echo "======================================================================"

# ======================== Validation ========================

# Check if dataset path is provided
if [ -z "$DATASET_PATH" ]; then
    echo "Error: DATASET_PATH is not set!"
    echo "Please set DATASET_PATH to your Visual Demo dataset JSONL file."
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

# Set CUDA visible devices to all GPUs
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FRM_DIR="$PROJECT_DIR/frm"

# Change to frm directory
cd "$FRM_DIR" || exit 1

# Build command
CMD="python run_visual_demo.py \
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

echo "Starting batch inference..."
echo ""

# Execute command with logging
$CMD 2>&1 | tee "$LOG_FILE"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo " Completed | Results: $OUTPUT_FILE"
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
    echo " Failed (exit code $EXIT_CODE) | Log: $LOG_FILE"
    echo "======================================================================"
    exit $EXIT_CODE
fi
