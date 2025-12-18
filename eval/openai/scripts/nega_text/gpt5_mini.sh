#!/bin/bash
#####################################################################
# Negative Text Demo - GPT-5-mini
#####################################################################

# API Configuration
API_KEY=""
MODEL="gpt-5-mini"

# Dataset Configuration
DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/benchmark/tiny-bench/text-neg-mini.jsonl"
IMAGE_ROOT="/projects/p32958/chengxuan/data/images"

# Output Configuration
BASE_OUTPUT_DIR="/projects/p32958/chengxuan/results/openai/nega_text"
PROJECT_NAME="gpt5_mini"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${PROJECT_NAME}_${TIMESTAMP}"
OUTPUT_FILE="${OUTPUT_DIR}/results.jsonl"
LOG_FILE="${OUTPUT_DIR}/run.log"

# Inference Parameters
MAX_WORKERS=5
NUM_INFERENCES=1
TEMPERATURE=1.0
MAX_COMPLETION_TOKENS=3000
LIMIT=-1
RESUME=false

echo "======================================================================"
echo "Negative Text Demo - GPT-5-mini"
echo "======================================================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_FILE"
echo "======================================================================"

# Check API key
if [ -z "$API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

# Check dataset file
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Navigate to codes directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODES_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")/codes"

cd "$CODES_DIR" || exit 1

# Build command
CMD="python run_text_demo.py \
    --api-key $API_KEY \
    --input $DATASET_PATH \
    --output $OUTPUT_FILE \
    --model $MODEL \
    --max-workers $MAX_WORKERS \
    --num-inferences $NUM_INFERENCES \
    --temperature $TEMPERATURE \
    --max-completion-tokens $MAX_COMPLETION_TOKENS"

if [ -n "$IMAGE_ROOT" ]; then
    CMD="$CMD --image-dir $IMAGE_ROOT"
fi

if [ $LIMIT -gt 0 ]; then
    CMD="$CMD --limit $LIMIT"
fi

if [ "$RESUME" = true ]; then
    CMD="$CMD --resume"
fi

# Run evaluation
$CMD 2>&1 | tee "$LOG_FILE"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo "Completed | Results: $OUTPUT_FILE"
    SUMMARY_FILE="${OUTPUT_FILE%.jsonl}_summary.json"
    [ -f "$SUMMARY_FILE" ] && cat "$SUMMARY_FILE"
else
    echo "Failed (exit code $EXIT_CODE)"
    exit $EXIT_CODE
fi
