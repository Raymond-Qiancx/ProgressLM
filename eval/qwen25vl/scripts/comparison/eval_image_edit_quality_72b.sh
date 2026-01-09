#!/bin/bash

#####################################################################
# Image Edit Quality Evaluation Script - 72B Model
#
# This script runs quality evaluation on edited images using
# Qwen2-VL 72B model with MODEL PARALLELISM (single process mode).
#
# The evaluation judges whether edited images successfully violate
# their corresponding instructions and represent failure states (0% progress).
#
# Key features:
# - Single process inference (no multi-GPU data parallelism)
# - Model automatically distributed across 4 GPUs
# - Optimized for 72B large models
# - Batch size defaults to 1 for memory efficiency
# - Binary yes/no judgment output
# - Safety checks for malicious/destructive edits
#
# Expected JSONL format:
# {
#   "strategy": "Color Change",
#   "prompt": "Change the green plate to red...",
#   "raw_demo": "[left] grab the plate while [right] lift the plate",
#   "response": "...",
#   "meta_data": {
#     "task_goal": "Place the two plates into the dish rack with both arms",
#     "image": "camera_front_0227_edited.jpg",
#     "text_demo": ["step1", "step2", ...],
#     "id": "h5_agilex_3rgb/10_packplate_2/2024_09_28-17_42_01-172863177768757312.00",
#     "data_source": "h5_agilex_3rgb",
#     "status": "success"
#   }
# }
#####################################################################

# ======================== Configuration ========================

# Model configuration - 72B Model
MODEL_PATH="/projects/p32958/chengxuan/models/Qwen2.5-VL-72B-Instruct"

# Dataset configuration - edited images dataset
DATASET_PATH="/projects/p32958/chengxuan/data/image_edit/edited_all.jsonl"
IMAGE_ROOT="/projects/p32958/chengxuan/results/progresslm/negative/image"

# Output configuration
OUTPUT_DIR="/projects/b1222/userdata/jianshu/chengxuan/saved/eval_results/image_edit_quality_72b"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/image_edit_quality_72b_${TIMESTAMP}.jsonl"
LOG_FILE="${OUTPUT_DIR}/image_edit_quality_72b_${TIMESTAMP}.log"

# GPU configuration - Use all 4 GPUs for model parallelism
GPU_IDS="0,1,2,3"  # All 4 GPUs will be used for model parallelism
BATCH_SIZE=48  # Small batch size for 72B model (increase if memory allows)

# Model parameters
TEMPERATURE=0.1  # Low temperature for more deterministic yes/no output
TOP_P=0.9
TOP_K=50
MAX_NEW_TOKENS=128  # Short responses expected (just yes/no)
MIN_PIXELS=$((1280*28*28))
MAX_PIXELS=$((5120*28*28))

# Processing parameters
LIMIT=-1  # Limit samples to process (-1 for all)

# Misc
VERBOSE=false  # Set to true for detailed output

# ======================== Auto Configuration ========================

echo "======================================================================"
echo "Image Edit Quality Evaluation - 72B Model"
echo "======================================================================"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Image Root: $IMAGE_ROOT"
echo "Output: $OUTPUT_FILE"
echo "GPUs: $GPU_IDS (Model Parallelism Mode)"
echo "Batch Size: $BATCH_SIZE"
echo "Mode: Binary yes/no quality judgment"
echo "======================================================================"
echo "NOTE: Using SINGLE PROCESS with MODEL PARALLELISM"
echo "      Model will be automatically distributed across all 4 GPUs"
echo "======================================================================"

# ======================== Validation ========================

# Check if dataset path is provided
if [ -z "$DATASET_PATH" ]; then
    echo "Error: DATASET_PATH is not set!"
    echo "Please set DATASET_PATH to your edited images JSONL file."
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

# Check if image root exists
if [ ! -d "$IMAGE_ROOT" ]; then
    echo "Error: Image root directory not found: $IMAGE_ROOT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ======================== Run Inference ========================

# Set CUDA visible devices to all GPUs for model parallelism
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN25VL_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CODES_DIR="$QWEN25VL_DIR/codes"

# Change to eval directory
cd "$CODES_DIR" || exit 1

# Build command
CMD="python run_image_edit_quality_eval.py \
    --model-path $MODEL_PATH \
    --dataset-path $DATASET_PATH \
    --output-file $OUTPUT_FILE \
    --image-root $IMAGE_ROOT \
    --batch-size $BATCH_SIZE \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-new-tokens $MAX_NEW_TOKENS \
    --min-pixels $MIN_PIXELS \
    --max-pixels $MAX_PIXELS"

# Add limit if specified
if [ $LIMIT -gt 0 ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Add verbose flag if enabled
if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

echo "Starting 72B model quality evaluation..."
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
