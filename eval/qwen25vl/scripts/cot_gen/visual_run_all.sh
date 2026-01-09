#!/bin/bash

#####################################################################
# Visual Demo Batch Runner
#
# This script runs multiple visual demo inference tasks sequentially.
# Each task can have different configurations (dataset, output dir, batch size, etc.)
#
# Usage:
#   1. Edit the task definitions below
#   2. Run: bash visual_run_all.sh
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
echo "Visual Demo Batch Runner"
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
#   export BATCH_SIZE=8
#   run_task "Task Description" "think_visual_demo_72b.sh"
#####################################################################

# Task 1: Example - Visual demo with 72B model on dataset 1

# export DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/train/sft/now/visua_normal_1.jsonl"
# export IMAGE_ROOT="/projects/p32958/chengxuan/data/images"
# export OUTPUT_DIR="/projects/p32958/chengxuan/results/new_begins/visual_normal_think/p1"
# export GPU_IDS="0,1,2,3"
# export BATCH_SIZE=4
# run_task "Visual Demo 72B - Dataset 1" "think_visual_demo_72b.sh"

# Task 1: Example - Visual demo with 72B model on dataset 1

# export DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/train/sft/now/visua_normal_2.jsonl"
# export IMAGE_ROOT="/projects/p32958/chengxuan/data/images"
# export OUTPUT_DIR="/projects/p32958/chengxuan/results/new_begins/visual_normal_think/p2"
# export GPU_IDS="0,1,2,3"
# export BATCH_SIZE=4
# run_task "Visual Demo 72B - Dataset 1" "think_visual_demo_72b.sh"


# export DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/train/sft/now/multi_camera_1.jsonl"
# export IMAGE_ROOT="/projects/p32958/chengxuan/data/images"
# export OUTPUT_DIR="/projects/p32958/chengxuan/results/new_begins/visual_multi_view/p1"
# export GPU_IDS="0,1,2,3"
# export BATCH_SIZE=4
# run_task "Visual Demo 72B - Dataset 1" "think_visual_demo_72b.sh"



# export DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/train/sft/now/multi_camera_2.jsonl"
# export IMAGE_ROOT="/projects/p32958/chengxuan/data/images"
# export OUTPUT_DIR="/projects/p32958/chengxuan/results/new_begins/visual_multi_view/p2"
# export GPU_IDS="0,1,2,3"
# export BATCH_SIZE=4
# run_task "Visual Demo 72B - Dataset 1" "think_visual_demo_72b.sh"



# export DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/train/sft/now/multi_camera_3.jsonl"
# export IMAGE_ROOT="/projects/p32958/chengxuan/data/images"
# export OUTPUT_DIR="/projects/p32958/chengxuan/results/new_begins/visual_multi_view/p3"
# export GPU_IDS="0,1,2,3"
# export BATCH_SIZE=4
# run_task "Visual Demo 72B - Dataset 1" "think_visual_demo_72b.sh"




# Task 2: Example - Visual demo with 72B model on dataset 2


export DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/train/sft/now/batch/edited_visual_nega_raw_2.jsonl"
export IMAGE_ROOT="/projects/p32958/chengxuan/data/images"
export OUTPUT_DIR="/projects/p32958/chengxuan/results/new_begins/visual_edit_think/raw_2"
export GPU_IDS="0,1,2,3"
export BATCH_SIZE=5
run_task "Visual Demo 72B - Dataset 1" "think_visual_nega_72b.sh"


export DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/train/sft/now/batch/edited_visual_nega_raw_2.jsonl"
export IMAGE_ROOT="/projects/p32958/chengxuan/data/images"
export OUTPUT_DIR="/projects/p32958/chengxuan/results/new_begins/visual_edit_think/raw_2"
export GPU_IDS="0,1,2,3"
export BATCH_SIZE=5
run_task "Visual Demo 72B - Dataset 1" "think_visual_nega_72b.sh"



# export DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/train/sft/now/batch/edited_visual_nega_raw_1.jsonl"
# export IMAGE_ROOT="/projects/p32958/chengxuan/data/images"
# export OUTPUT_DIR="/projects/p32958/chengxuan/results/new_begins/visual_edit_think/raw_1"
# export GPU_IDS="0,1,2,3"
# export BATCH_SIZE=5
# run_task "Visual Demo 72B - Dataset 1" "think_visual_nega_72b.sh"

# export DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/train/sft/now/batch/edited_visual_nega_raw_1.jsonl"
# export IMAGE_ROOT="/projects/p32958/chengxuan/data/images"
# export OUTPUT_DIR="/projects/p32958/chengxuan/results/new_begins/visual_edit_think/raw_1"
# export GPU_IDS="0,1,2,3"
# export BATCH_SIZE=5
# run_task "Visual Demo 72B - Dataset 1" "think_visual_nega_72b.sh"






# Task 3: Example - Visual demo with 32B model

# export MODEL_PATH="/projects/p32958/chengxuan/models/Qwen2.5-VL-32B-Instruct"
# export DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/train/visual_demo/dataset3.jsonl"
# export OUTPUT_DIR="/projects/p32958/chengxuan/results/visual_32b_batch1"
# export BATCH_SIZE=4
# run_task "Visual Demo 32B - Dataset 3" "think_visual_demo.sh"












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
