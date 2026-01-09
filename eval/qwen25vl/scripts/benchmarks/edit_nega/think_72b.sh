#!/bin/bash

#####################################################################
# Visual Demo Progress Estimation Evaluation Script - 72B Think Mode
#
# This script runs progress estimation evaluation on Edit Nega dataset
# using Qwen2.5-VL 72B model with single process mode (model parallelism).
#
# THINK MODE: Uses full prompt with intermediate reasoning steps
# (<ref_think>, <ref>, <score_think> tags).
#
# Key features:
# - Single process inference (no multi-GPU data parallelism)
# - Model automatically distributed across 4 GPUs
# - Optimized for 72B large model
#####################################################################

# ======================== Configuration ========================

# Model configuration - 72B Model
MODEL_PATH="/projects/p32958/chengxuan/models/Qwen2.5-VL-72B-Instruct"

# Dataset configuration - using merged eval dataset
DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/benchmark/tiny-bench/visual-nega.jsonl"
IMAGE_ROOT="/projects/p32958/chengxuan/data/images"

# Output configuration
BASE_OUTPUT_DIR="/projects/p32958/chengxuan/results/new_pro_bench/edit_nega/think_72B"
PROJECT_NAME="visual_think_72B"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${PROJECT_NAME}_${TIMESTAMP}"
OUTPUT_FILE="${OUTPUT_DIR}/results.jsonl"
LOG_FILE="${OUTPUT_DIR}/run.log"

# GPU configuration - Use all 4 GPUs for model parallelism
GPU_IDS="0,1,2,3"  # All 4 GPUs will be used for model parallelism
BATCH_SIZE=5  # Batch size for 72B model

# Inference configuration
NUM_INFERENCES=1  # Number of inferences per sample (data expansion factor)

# Model parameters
TEMPERATURE=0.6  # Higher temperature for diversity across multiple inferences
TOP_P=0.9
TOP_K=50
MAX_NEW_TOKENS=4096  # Token limit for reasoning
MIN_PIXELS=$((1280*28*28))
MAX_PIXELS=$((5120*28*28))

# Processing parameters
LIMIT=-1  # Limit samples to process after expansion (-1 for all)

# Misc
VERBOSE=false  # Set to true for detailed output

# ======================== Auto Configuration ========================

echo "======================================================================"
echo "Edit Nega Visual Demo Progress Estimation - 72B Think Mode"
echo "======================================================================"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_FILE"
echo "GPUs: $GPU_IDS (Model Parallelism Mode)"
echo "Batch Size: $BATCH_SIZE"
echo "Inferences per Sample: $NUM_INFERENCES"
echo "======================================================================"
echo "NOTE: Using SINGLE PROCESS with MODEL PARALLELISM"
echo "      Model will be automatically distributed across all 4 GPUs"
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

# Set CUDA visible devices to all GPUs for model parallelism
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN25VL_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
CODES_DIR="$QWEN25VL_DIR/codes"

# Change to eval directory
cd "$CODES_DIR" || exit 1

# Build command - using single process for large model
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

echo "Starting 72B model evaluation inference (think mode)..."
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
