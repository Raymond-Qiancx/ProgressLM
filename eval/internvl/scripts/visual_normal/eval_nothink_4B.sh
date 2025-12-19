#!/bin/bash

#####################################################################
# Visual Demo Progress Estimation Evaluation Script - InternVL 4B (NoThink)
#
# This script runs progress estimation evaluation on Visual Demo dataset
# using InternVL model with simplified output (score only).
#####################################################################

# ======================== Configuration ========================

# Model configuration
MODEL_PATH="/projects/p32958/jianshu/weight/OpenGVLab/InternVL3_5-4B"

# Dataset configuration
DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/benchmark/tiny-bench/visual_single_mini.jsonl"
IMAGE_ROOT="/projects/p32958/chengxuan/data/images"

# Output configuration
BASE_OUTPUT_DIR="/projects/p32958/chengxuan/results/internvl/visual_normal_nothink"
PROJECT_NAME="internvl_4B_nothink"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${PROJECT_NAME}_${TIMESTAMP}"
OUTPUT_FILE="${OUTPUT_DIR}/results.jsonl"
LOG_FILE="${OUTPUT_DIR}/run.log"

# GPU configuration
GPU_IDS="0,1,2,3"

# Inference configuration
NUM_INFERENCES=1

# Model parameters
TEMPERATURE=0.6
TOP_P=0.9
MAX_NEW_TOKENS=512

# InternVL specific parameters
MAX_NUM_TILES=4
INPUT_SIZE=448

# Processing parameters
LIMIT=-1
BATCH_SIZE=40

# Misc
VERBOSE=false

# ======================== Auto Configuration ========================

echo "======================================================================"
echo "Visual Demo Progress Estimation - InternVL 4B (NoThink) Evaluation"
echo "======================================================================"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_FILE"
echo "GPUs: $GPU_IDS"
echo "======================================================================"

# ======================== Validation ========================

if [ -z "$DATASET_PATH" ]; then
    echo "Error: DATASET_PATH is not set!"
    exit 1
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ======================== Run Inference ========================

export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Get script and codes directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERNVL_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CODES_DIR="$INTERNVL_DIR/codes"

cd "$CODES_DIR" || exit 1

# Use nothink version
CMD="python run_visual_demo_nothink.py \
    --model-path $MODEL_PATH \
    --dataset-path $DATASET_PATH \
    --output-file $OUTPUT_FILE \
    --batch-size $BATCH_SIZE \
    --num-inferences $NUM_INFERENCES \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --max-new-tokens $MAX_NEW_TOKENS \
    --max-num-tiles $MAX_NUM_TILES \
    --input-size $INPUT_SIZE"

if [ -n "$IMAGE_ROOT" ]; then
    CMD="$CMD --image-root $IMAGE_ROOT"
fi

if [ $LIMIT -gt 0 ]; then
    CMD="$CMD --limit $LIMIT"
fi

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

echo "Starting evaluation inference..."
echo ""

$CMD 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo " Completed | Results: $OUTPUT_FILE"
    echo "======================================================================"

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
