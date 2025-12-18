#!/bin/bash
#####################################################################
# Negative Text Demo - Qwen3VL-32B (NoThink)
#####################################################################

MODEL_PATH="/projects/p32958/jianshu/weight/Qwen/Qwen3-VL-32B-Instruct"
DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/benchmark/tiny-bench/text-neg-mini.jsonl"
IMAGE_ROOT="/projects/p32958/chengxuan/data/images"

BASE_OUTPUT_DIR="/projects/p32958/chengxuan/results/qwen3vl/nega_text"
PROJECT_NAME="qwen3vl_32b_nothink"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${PROJECT_NAME}_${TIMESTAMP}"
OUTPUT_FILE="${OUTPUT_DIR}/results.jsonl"
LOG_FILE="${OUTPUT_DIR}/run.log"

GPU_IDS="0,1,2,3"
BATCH_SIZE=2
NUM_INFERENCES=1
TEMPERATURE=0.4
TOP_P=0.9
TOP_K=50
MAX_NEW_TOKENS=4096
LIMIT=-1
VERBOSE=false

echo "======================================================================"
echo "Negative Text Demo - Qwen3VL-32B (NoThink)"
echo "======================================================================"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_FILE"
echo "======================================================================"

if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES=$GPU_IDS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")/codes"

cd "$EVAL_DIR" || exit 1

CMD="python run_text_demo_nothink.py \
    --model-path $MODEL_PATH \
    --dataset-path $DATASET_PATH \
    --output-file $OUTPUT_FILE \
    --batch-size $BATCH_SIZE \
    --num-inferences $NUM_INFERENCES \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-new-tokens $MAX_NEW_TOKENS"

if [ -n "$IMAGE_ROOT" ]; then
    CMD="$CMD --image-root $IMAGE_ROOT"
fi

if [ $LIMIT -gt 0 ]; then
    CMD="$CMD --limit $LIMIT"
fi

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

$CMD 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo " Completed | Results: $OUTPUT_FILE"
else
    echo " Failed (exit code $EXIT_CODE)"
    exit $EXIT_CODE
fi
