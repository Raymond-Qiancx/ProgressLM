#!/bin/bash

#####################################################################
# Visual Demo Progress Estimation Evaluation Script
#
# This script runs progress estimation evaluation on Visual Demo dataset
# using Qwen2-VL model with distributed GPU support.
#
# Expected JSONL format:
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
# MODEL_PATH="/projects/p32958/Results/full_model/qwen25_vl_3b_rl_35k"
# MODEL_PATH="/projects/p32958/Results/full_model/qwen25vl_3b_sft"
MODEL_PATH="/projects/p32958/Results/sft_model/qwen25vl_3b_think_sft"

# Dataset configuration - using merged eval dataset
DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/benchmark/tiny-bench/visual-nega.jsonl"
IMAGE_ROOT="/projects/p32958/chengxuan/data/images"

# Output configuration
BASE_OUTPUT_DIR="/projects/p32958/chengxuan/results/new_pro_bench/edit_nega/3B_SFT"


PROJECT_NAME="visual_3B_SFT"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${PROJECT_NAME}_${TIMESTAMP}"
OUTPUT_FILE="${OUTPUT_DIR}/results.jsonl"
LOG_FILE="${OUTPUT_DIR}/run.log"

# GPU configuration
GPU_IDS="0,1,2,3"  # Comma-separated GPU IDs to use
BATCH_SIZE=10  # Batch size per GPU (adjust based on VRAM and image count)

# Inference configuration
NUM_INFERENCES=1  # Number of inferences per sample (data expansion factor)

# Model parameters
TEMPERATURE=0.6  # Higher temperature for diversity across multiple inferences
TOP_P=0.9
TOP_K=50
MAX_NEW_TOKENS=40000  # Increased for longer CoT reasoning chains
MIN_PIXELS=$((1280*28*28))
MAX_PIXELS=$((5120*28*28))

# Processing parameters
LIMIT=-1  # Limit samples to process after expansion (-1 for all)

# Misc
VERBOSE=false  # Set to true for detailed output

# ======================== Auto Configuration ========================

echo "======================================================================"
echo "Visual Demo Progress Estimation - Evaluation"
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
mkdir -p "$OUTPUT_DIR"

# ======================== Run Inference ========================

# Set CUDA visible devices to all GPUs
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN25VL_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
CODES_DIR="$QWEN25VL_DIR/codes"

# Change to eval directory
cd "$CODES_DIR" || exit 1

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

echo "Starting evaluation inference..."
echo ""

# Execute command with logging
$CMD 2>&1 | tee "$LOG_FILE"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo " Completed | Results: $OUTPUT_FILE"
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
    echo " Failed (exit code $EXIT_CODE) | Log: $LOG_FILE"
    echo "======================================================================"
    exit $EXIT_CODE
fi
