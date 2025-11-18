#!/bin/bash

#####################################################################
# Text Demo Batch Runner
#
# This script runs multiple text demo inference tasks sequentially.
# Each task can have different configurations (dataset, output dir, batch size, etc.)
#
# Usage:
#   1. Edit the task definitions below
#   2. Run: bash text_run_all.sh
#####################################################################

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Track statistics
TOTAL_TASKS=0
COMPLETED_TASKS=0
FAILED_TASKS=0
START_TIME=$(date +%s)

echo "======================================================================"
echo "Text Demo Batch Runner"
echo "======================================================================"
echo ""

#####################################################################
# Task Definition Helper Function
#####################################################################
run_task() {
    local task_name="$1"
    local script_name="$2"
    shift 2

    TOTAL_TASKS=$((TOTAL_TASKS + 1))

    echo ""
    echo -e "${YELLOW}======================================================================${NC}"
    echo -e "${YELLOW}Task $TOTAL_TASKS: $task_name${NC}"
    echo -e "${YELLOW}Script: $script_name${NC}"
    echo -e "${YELLOW}======================================================================${NC}"
    echo ""

    # Export environment variables and run the script
    TASK_START=$(date +%s)
    bash "$SCRIPT_DIR/$script_name"
    EXIT_CODE=$?
    TASK_END=$(date +%s)
    TASK_DURATION=$((TASK_END - TASK_START))

    # Clear all environment variables after task completion
    unset MODEL_PATH DATASET_PATH IMAGE_ROOT OUTPUT_DIR GPU_IDS BATCH_SIZE
    unset NUM_INFERENCES TEMPERATURE TOP_P TOP_K MAX_NEW_TOKENS MIN_PIXELS MAX_PIXELS LIMIT VERBOSE

    if [ $EXIT_CODE -eq 0 ]; then
        COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
        echo ""
        echo -e "${GREEN}✓ Task $TOTAL_TASKS completed successfully (Duration: ${TASK_DURATION}s)${NC}"
    else
        FAILED_TASKS=$((FAILED_TASKS + 1))
        echo ""
        echo -e "${RED}✗ Task $TOTAL_TASKS failed with exit code $EXIT_CODE (Duration: ${TASK_DURATION}s)${NC}"
        echo -e "${RED}Continuing with next task...${NC}"
    fi
}

#####################################################################
# Task Definitions
#
# Define your tasks below. Each task:
# 1. Sets environment variables for parameters to override
# 2. Calls run_task with a descriptive name and the script to run
#
# Example task structure:
#   export DATASET_PATH="/path/to/dataset.jsonl"
#   export OUTPUT_DIR="/path/to/output"
#   export BATCH_SIZE=40
#   run_task "Task Description" "think_text_demo_72b.sh"
#####################################################################

# Task 1: Text Nega 72B - Negative Samples with Rank

export DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/sft_data/text_nega_new/new_text_nega_merged_with_rank.jsonl"
export IMAGE_ROOT="/projects/p32958/chengxuan/data/images"
export OUTPUT_DIR="/projects/p32958/chengxuan/results/new_begins/text_nega_think"
export GPU_IDS="0,1,2,3"
export BATCH_SIZE=40
run_task "Text Nega 72B - Negative Samples" "think_text_nega_72b.sh"

# Task 2: Example - Text demo with 72B model on dataset 2

# export DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/train/text_demo/text_h5_agilex_3rgb_sft.jsonl"
# export IMAGE_ROOT="/projects/p32958/chengxuan/data/images"
# export OUTPUT_DIR="/projects/p32958/chengxuan/results/new_begins/text_normal_think"
# export GPU_IDS="0,1,2,3"
# export BATCH_SIZE=40
# run_task "Text Demo 72B - Dataset 1" "think_text_demo_72b.sh"

# Task 3: Example - Text demo with 32B model
# Uncomment and modify as needed:
# export MODEL_PATH="/projects/p32958/chengxuan/models/Qwen2.5-VL-32B-Instruct"
# export DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/sft_data/text_nega_new/new_text_nega_raw.jsonl"
# export OUTPUT_DIR="/projects/p32958/chengxuan/results/text_32b_batch1"
# export BATCH_SIZE=10
# run_task "Text Demo 32B - Dataset 3" "think_text_demo.sh"

# export DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/train/text_demo/new/edited_nega_text_sft.jsonl"

# Add more tasks as needed...

#####################################################################
# Summary
#####################################################################

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "======================================================================"
echo "Batch Run Summary"
echo "======================================================================"
echo "Total Tasks: $TOTAL_TASKS"
echo -e "${GREEN}Completed: $COMPLETED_TASKS${NC}"
echo -e "${RED}Failed: $FAILED_TASKS${NC}"
echo "Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "======================================================================"

if [ $FAILED_TASKS -gt 0 ]; then
    exit 1
fi
