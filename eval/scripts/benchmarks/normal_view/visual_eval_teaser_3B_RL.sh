#!/bin/bash

#####################################################################
# Visual Demo Progress Estimation Evaluation Script (Teaser Benchmark)
#
# This script runs progress estimation evaluation on the Teaser Visual
# Demo dataset using Qwen2-VL 3B RL model.
#
# Expected JSONL format (same as visual_single_mini.jsonl):
# {
#   "id": "env/task/episode_id",
#   "task_goal": "description of the task",
#   "visual_demo": ["frame_000001.jpg", "frame_000010.jpg", ...],
#   "total_steps": "4",
#   "stage_to_estimate": ["frame_000005.jpg"],
#   "closest_idx": "1",
#   "delta": "+7%",
#   "progress_score": "8%",
#   "data_source": "teaser_bench"
# }
#####################################################################

# ======================== Configuration ========================

# Model configuration
MODEL_PATH="/projects/p32958/Results/full_model/global_step_485/actor/qwen25vl_3b_rl_scale"

# Dataset configuration - Teaser visual demo meta-data
DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/benchmark/teaser/jsonl/teaser_bench/visual_demo_from_demo.jsonl"
IMAGE_ROOT="/projects/p32958/chengxuan/ProgressLM/data/benchmark/teaser/demo"

# Output configuration
BASE_OUTPUT_DIR="/projects/p32958/chengxuan/ProgressLM/data/benchmark/teaser/results"

PROJECT_NAME="visual_teaser_3B_RL"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${PROJECT_NAME}_${TIMESTAMP}"
OUTPUT_FILE="${OUTPUT_DIR}/results.jsonl"
LOG_FILE="${OUTPUT_DIR}/run.log"

# GPU configuration
GPU_IDS="0,1,2,3"  # Comma-separated GPU IDs to use
BATCH_SIZE=20       # Batch size per GPU (adjust based on VRAM and image count)

# Inference configuration
NUM_INFERENCES=1  # Number of inferences per sample (data expansion factor)

# Model parameters
TEMPERATURE=0.6
TOP_P=0.9
TOP_K=50
MAX_NEW_TOKENS=40000
MIN_PIXELS=$((1280*28*28))
MAX_PIXELS=$((5120*28*28))

# Processing parameters
LIMIT=-1  # Limit samples to process after expansion (-1 for all)

# Misc
VERBOSE=false  # Set to true for detailed output

# ======================== Auto Configuration ========================

echo "======================================================================"
echo "Visual Demo Progress Estimation - Teaser Evaluation"
echo "======================================================================"
echo "Dataset: $DATASET_PATH"
echo "Image Root: $IMAGE_ROOT"
echo "Output: $OUTPUT_FILE"
echo "GPUs: $GPU_IDS"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Inferences per Sample: $NUM_INFERENCES"
echo "======================================================================"

# ======================== Validation ========================

if [ -z "$DATASET_PATH" ]; then
    echo "Error: DATASET_PATH is not set!"
    echo "Please set DATASET_PATH to your Visual Demo dataset JSONL file."
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
EVAL_DIR="$PROJECT_DIR/qwen25vl"

cd "$EVAL_DIR" || exit 1

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

if [ -n "$IMAGE_ROOT" ] && [ "$IMAGE_ROOT" != "/path/to/your/images" ]; then
    CMD="$CMD --image-root $IMAGE_ROOT"
fi

if [ $LIMIT -gt 0 ]; then
    CMD="$CMD --limit $LIMIT"
fi

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

echo "Starting Teaser evaluation inference..."
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

