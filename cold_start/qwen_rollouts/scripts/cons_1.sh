#!/bin/bash

#####################################################################
# Visual Demo Progress Estimation Script - 72B Model (Model Parallelism)
#
# This script runs progress estimation on Visual Demo dataset using
# Qwen2-VL-72B model with model parallelism across multiple GPUs.
# Uses FRM's cheat prompt system with ground-truth.
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

# Model configuration - 72B MODEL
MODEL_PATH="/projects/p32958/chengxuan/models/Qwen2.5-VL-72B-Instruct"

# Dataset configuration
DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/train/visual_demo/cons_visual_1.jsonl"
IMAGE_ROOT="/projects/p32958/chengxuan/new_extracted_images/images"  # Optional: root directory for relative image paths

# Output configuration
OUTPUT_DIR="/projects/p32958/chengxuan/results/progresslm/cold_data/visual_3rgb_sft"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/visual_demo_72b_results_${TIMESTAMP}.jsonl"
LOG_FILE="${OUTPUT_DIR}/visual_demo_72b_${TIMESTAMP}.log"

# GPU configuration
GPU_IDS="0,1,2,3"  # Comma-separated GPU IDs to use (72B requires multiple GPUs for model parallelism)
BATCH_SIZE=8  # Batch size (reduced to 1 for 72B model due to high memory requirements)

# Inference configuration
NUM_INFERENCES=1  # Number of inferences per sample (data expansion factor)

# Model parameters
TEMPERATURE=0.6  # Higher temperature for diversity across multiple inferences
TOP_P=0.9
TOP_K=50
MAX_NEW_TOKENS=40000  # Increased from 5120 to 40000 for longer CoT reasoning chains
MIN_PIXELS=$((1280*28*28))
MAX_PIXELS=$((5120*28*28))

# Processing parameters
LIMIT=-1  # Limit samples to process after expansion (-1 for all)

# Misc
VERBOSE=false  # Set to true for detailed output

# ======================== Auto Configuration ========================

echo "======================================================================"
echo "Visual Demo Progress Estimation - 72B Model (Model Parallelism + FRM Cheat)"
echo "======================================================================"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_FILE"
echo "GPUs: $GPU_IDS"
echo "Batch Size: $BATCH_SIZE (optimized for 72B)"
echo "Inferences per Sample: $NUM_INFERENCES"
echo "Model: 72B (Single Process with Model Parallelism)"
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

# Set CUDA visible devices to all GPUs (model will be distributed across them)
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FRM_DIR="$PROJECT_DIR/frm"

# Change to frm directory
cd "$FRM_DIR" || exit 1

# Build command - using single process script for 72B
CMD="python run_visual_demo_single.py \
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

echo "Starting batch inference with 72B model (single process, model parallelism)..."
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
