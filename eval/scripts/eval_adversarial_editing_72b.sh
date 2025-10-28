#!/bin/bash

#####################################################################
# Adversarial Image Editing Evaluation Script - 72B Model
#
# This script runs adversarial image editing inference using
# Qwen2.5-VL 72B model with MODEL PARALLELISM (single process mode).
#
# Key features:
# - Single process inference (no multi-GPU data parallelism)
# - Model automatically distributed across 4 GPUs
# - Optimized for 72B/32B large models
# - Batch size defaults to 1 for memory efficiency
#
# Expected JSONL format:
# {
#   "id": "h5_tienkung_xsens_1rgb/tool_liftn_box_place/2024-09-28-10-54-08",
#   "task_goal": "placing a tool into a box",
#   "text_demo": ["reach for the tool", "grasp the tool", "lift the tool", ...],
#   "total_steps": 6,
#   "stage_to_estimate": "camera_top_0556.jpg",
#   "closest_idx": 6,
#   "progress_score": "100%",
#   "data_source": "h5_tienkung_xsens_1rgb"
# }
#####################################################################

# ======================== Configuration ========================

# Model configuration - 72B Model
MODEL_PATH="/projects/p32958/chengxuan/models/Qwen2.5-VL-72B-Instruct"  # UPDATE THIS

# Dataset configuration
# DATASET_PATH="/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/train/text_demo/text_robo_sft_all.jsonl"  # UPDATE THIS
DATASET_PATH="/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/train/text_demo/text_robo_5.jsonl"  # UPDATE THIS
IMAGE_ROOT="/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/images"  # UPDATE THIS - will be prepended to image paths

# Output configuration
OUTPUT_DIR="/projects/p32958/chengxuan/results/progresslm/negative/data"  # UPDATE THIS
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/adversarial_editing_72b_${TIMESTAMP}.jsonl"
LOG_FILE="${OUTPUT_DIR}/adversarial_editing_72b_${TIMESTAMP}.log"

# GPU configuration - Use all 4 GPUs for model parallelism
GPU_IDS="0,1,2,3"  # All 4 GPUs will be used for model parallelism
BATCH_SIZE=48  # Small batch size for 72B model (increase if memory allows)


# Inference configuration
NUM_INFERENCES=1  # Number of inferences per sample (data expansion factor)
LIMIT=-1  # -1 for all samples, or specify a number to limit

# Model parameters
TEMPERATURE=0.7  # Sampling temperature
TOP_P=0.9  # Top-p (nucleus) sampling
TOP_K=50  # Top-k sampling
MAX_NEW_TOKENS=20000  # Maximum tokens to generate
MIN_PIXELS=$((1280*28*28))
MAX_PIXELS=$((5120*28*28))

# Misc
VERBOSE=false  # Set to true for detailed output

# ======================== Auto Configuration ========================

echo "======================================================================"
echo "Adversarial Image Editing - 72B Model (Model Parallelism)"
echo "======================================================================"
echo "Model Path       : $MODEL_PATH"
echo "Dataset Path     : $DATASET_PATH"
echo "Image Root       : $IMAGE_ROOT"
echo "Output File      : $OUTPUT_FILE"
echo "Log File         : $LOG_FILE"
echo "GPUs             : $GPU_IDS (Model Parallelism Mode)"
echo "Batch Size       : $BATCH_SIZE"
echo "Num Inferences   : $NUM_INFERENCES"
echo "Temperature      : $TEMPERATURE"
echo "Max New Tokens   : $MAX_NEW_TOKENS"
if [ $LIMIT -gt 0 ]; then
    echo "Sample Limit     : $LIMIT"
fi
echo "======================================================================"
echo "NOTE: Using SINGLE PROCESS with MODEL PARALLELISM"
echo "      Model will be automatically distributed across all GPUs"
echo "======================================================================"

# ======================== Validation ========================

# Check if dataset path is provided
if [ -z "$DATASET_PATH" ]; then
    echo "Error: DATASET_PATH is not set!"
    echo "Please set DATASET_PATH to your dataset JSONL file."
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

# Check if image root exists (if not default placeholder)
if [ -n "$IMAGE_ROOT" ] && [ "$IMAGE_ROOT" != "/path/to/images" ]; then
    if [ ! -d "$IMAGE_ROOT" ]; then
        echo "Error: Image root directory not found: $IMAGE_ROOT"
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ======================== Run Inference ========================

# Set CUDA visible devices to all GPUs for model parallelism
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EVAL_DIR="$PROJECT_DIR/qwen25vl"

# Change to eval directory
cd "$EVAL_DIR" || exit 1

echo ""
echo "Working directory: $(pwd)"
echo "Starting 72B model adversarial editing inference..."
echo ""

# Build command - using run_adversarial_editing_single.py for 72B model
CMD="python run_adversarial_editing_single.py \
    --model-path \"$MODEL_PATH\" \
    --dataset-path \"$DATASET_PATH\" \
    --output-file \"$OUTPUT_FILE\" \
    --image-root \"$IMAGE_ROOT\" \
    --batch-size $BATCH_SIZE \
    --num-inferences $NUM_INFERENCES \
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

# Execute command with logging
echo "Command: $CMD"
echo ""

# Run with both console output and log file
eval $CMD 2>&1 | tee "$LOG_FILE"

# ======================== Post-Processing ========================

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Inference completed successfully!"
    echo "======================================================================"
    echo "Results saved to  : $OUTPUT_FILE"
    echo "Summary saved to  : ${OUTPUT_FILE%.jsonl}_summary.json"
    echo "Log saved to      : $LOG_FILE"

    # Display summary if exists
    SUMMARY_FILE="${OUTPUT_FILE%.jsonl}_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo ""
        echo "Summary:"
        cat "$SUMMARY_FILE" | python -m json.tool 2>/dev/null || cat "$SUMMARY_FILE"
    fi

    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "✗ ERROR: Inference failed with exit code $EXIT_CODE"
    echo "======================================================================"
    echo "Check log file for details: $LOG_FILE"
    echo "======================================================================"
    exit $EXIT_CODE
fi
