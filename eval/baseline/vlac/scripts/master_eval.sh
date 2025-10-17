#!/bin/bash
################################################################################
# Master Evaluation Pipeline Script
#
# This script runs the main evaluation pipeline, which handles complex data
# preparation and logic for selecting reference trajectories.
#
# Configure the paths and the mode below, then run the script:
#   bash scripts/master_eval.sh
################################################################################

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Mode Selection ---
# true:  Enable cross-trajectory mode. Find a similar but different trajectory
#        as the reference based on the logic in the pipeline script.
# false: Enable intra-trajectory mode. Use each trajectory's own visual_demo
#        as its reference.
cross_trajectory_ref=true

# --- Path Configuration (MUST BE ABSOLUTE PATHS) ---
# Path to the directory containing the pre-processed (split) .jsonl files.
# This directory is created by running the preprocess_data.sh script.
PROCESSED_DATA_DIR="/home/vcj9002/jianshu/workspace/code/ProgressLM/data/h5_tienkung_xsens_converted_split"

# Path to the root directory containing all image folders (e.g., h5_tienkung_xsens_1rgb/...).
IMAGE_ROOT_DIR="/home/vcj9002/jianshu/workspace/data/robomind/data/images"

# Path to the downloaded VLAC model.
MODEL_PATH="/home/vcj9002/jianshu/workspace/weight/VLAC"

# Directory where the final JSON results will be saved.
OUTPUT_DIR="/home/vcj9002/jianshu/workspace/data/robomind/data/annotations/cache"


# --- GPU and Passthrough Arguments ---
# Specify the GPU IDs to use, comma-separated (e.g., "0,1,4,7").
GPU_IDS="0,1,2,3"

# Other arguments to pass directly to the underlying run_eval.py script.
# Example: --batch_num 10 --rich
PASSTHROUGH_ARGS="--batch_num 5"


# ============================================================================
# EXECUTION LOGIC (DO NOT MODIFY)
# ============================================================================

set -e # Exit immediately if a command exits with a non-zero status.

# Get the directory of this script to find the pipeline script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PIPELINE_SCRIPT="$SCRIPT_DIR/../run_evaluation_pipeline.py"

# Check if paths are placeholders
# if [[ "$JSONL_PATH" == "/path/to/your/h5_tienkung_xsens_converted.jsonl" ]] || \
#    [[ "$IMAGE_ROOT_DIR" == "/path/to/your/image_root_directory" ]] || \
#    [[ "$MODEL_PATH" == "/path/to/your/VLAC/model" ]]; then
#     echo -e "\033[0;31m[ERROR] Please update the placeholder paths in this script before running.\033[0m"
#     exit 1
# fi

# Build the command
CMD="python $PIPELINE_SCRIPT \
    --processed_data_dir $PROCESSED_DATA_DIR \
    --image_root_dir $IMAGE_ROOT_DIR \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --gpu_ids $GPU_IDS"

if [ "$cross_trajectory_ref" = true ]; then
    CMD="$CMD --cross_trajectory_ref"
fi

# Append passthrough arguments
if [ -n "$PASSTHROUGH_ARGS" ]; then
    CMD="$CMD $PASSTHROUGH_ARGS"
fi

# Run the command
echo -e "\033[0;32m[INFO] Starting evaluation pipeline...\033[0m"
echo "Mode: cross_trajectory_ref=$cross_trajectory_ref"
echo "Command: $CMD"
echo ""

eval $CMD

echo -e "\n\033[0;32m[INFO] Pipeline finished successfully!\033[0m"

