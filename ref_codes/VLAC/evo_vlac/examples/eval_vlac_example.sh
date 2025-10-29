#!/bin/bash

#####################################################################
# VLAC Example Visual Demo Progress Estimation Evaluation Script
#
# This script evaluates the VLAC example dataset (18 samples from
# "Scoop the rice into the rice cooker" task) using your trained model.
#
# Dataset: vlac_example_visual_demo.jsonl (18 records, 3 camera views)
# Task: Progress estimation with CoT reasoning
#####################################################################

# ======================== Configuration ========================

# Model configuration - UPDATE THIS PATH TO YOUR MODEL
MODEL_PATH="/projects/b1222/userdata/jianshu/chengxuan/saved/saved_results/progresslm/models/3b_sft_qwen25vl_4epoch"
# Example paths:
# MODEL_PATH="/projects/b1222/userdata/jianshu/chengxuan/saved/saved_results/progresslm/models/progresslm_sft_epoch2_model"
# MODEL_PATH="/Users/cxqian/Codes/ProgressLM/saved/models/your_model"

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dataset configuration - using the VLAC example dataset
DATASET_PATH="${SCRIPT_DIR}/vlac_example_visual_demo.jsonl"
IMAGE_ROOT="${SCRIPT_DIR}"  # Images are in ./images/ref and ./images/test

# Output configuration
OUTPUT_DIR="${SCRIPT_DIR}/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/eval_vlac_example_${TIMESTAMP}.jsonl"
LOG_FILE="${OUTPUT_DIR}/eval_vlac_example_${TIMESTAMP}.log"

# GPU configuration
GPU_IDS="0"  # Comma-separated GPU IDs to use (e.g., "0,1" for 2 GPUs)
BATCH_SIZE=8  # Small batch size for 18 samples

# Inference configuration
NUM_INFERENCES=1  # Number of inferences per sample (increase for diversity testing)

# Model parameters
TEMPERATURE=0.6  # Temperature for sampling
TOP_P=0.9
TOP_K=50
MAX_NEW_TOKENS=4096  # Sufficient for CoT reasoning
MIN_PIXELS=$((1280*28*28))
MAX_PIXELS=$((5120*28*28))

# Processing parameters
LIMIT=-1  # Process all samples (-1 for all, or set to specific number for testing)

# Misc
VERBOSE=false  # Set to true for detailed output

# ======================== Validation ========================

echo "======================================================================"
echo "VLAC Example Dataset - Visual Demo Progress Estimation Evaluation"
echo "======================================================================"
echo "Dataset: $DATASET_PATH"
echo "Image Root: $IMAGE_ROOT"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_FILE"
echo "GPUs: $GPU_IDS"
echo "Batch Size: $BATCH_SIZE"
echo "Inferences per Sample: $NUM_INFERENCES"
echo "======================================================================"
echo ""

# Check if model path is set
if [ "$MODEL_PATH" = "/path/to/your/model" ]; then
    echo "Error: Please set MODEL_PATH to your actual model directory!"
    echo "Edit this script and update the MODEL_PATH variable at the top."
    exit 1
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    exit 1
fi

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    echo "Please run convert_to_jsonl.py first to generate the dataset."
    exit 1
fi

# Check if image directories exist
if [ ! -d "${IMAGE_ROOT}/images/ref" ] || [ ! -d "${IMAGE_ROOT}/images/test" ]; then
    echo "Error: Image directories not found!"
    echo "Expected: ${IMAGE_ROOT}/images/ref and ${IMAGE_ROOT}/images/test"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ======================== Find Python Script ========================

# Navigate to the eval/qwen25vl directory to run the Python script
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
EVAL_DIR="$PROJECT_ROOT/eval/qwen25vl"
PYTHON_SCRIPT="$EVAL_DIR/run_visual_demo.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

echo "Python script: $PYTHON_SCRIPT"
echo "Project root: $PROJECT_ROOT"
echo ""

# ======================== Run Inference ========================

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Change to eval directory
cd "$EVAL_DIR" || exit 1

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
    --max-pixels $MAX_PIXELS \
    --image-root $IMAGE_ROOT"

# Add limit if specified
if [ $LIMIT -gt 0 ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Add verbose flag if enabled
if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

echo "======================================================================"
echo "Starting evaluation..."
echo "======================================================================"
echo ""

# Execute command with logging
$CMD 2>&1 | tee "$LOG_FILE"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Evaluation Completed Successfully"
    echo "======================================================================"
    echo "Results: $OUTPUT_FILE"
    echo "Log: $LOG_FILE"
    echo ""

    # Display summary if exists
    SUMMARY_FILE="${OUTPUT_FILE%.jsonl}_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Summary:"
        cat "$SUMMARY_FILE"
        echo ""
    fi

    # Show sample count
    RESULT_COUNT=$(wc -l < "$OUTPUT_FILE" | tr -d ' ')
    echo "Total results: $RESULT_COUNT"
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "✗ Evaluation Failed (exit code $EXIT_CODE)"
    echo "======================================================================"
    echo "Log: $LOG_FILE"
    echo "======================================================================"
    exit $EXIT_CODE
fi
