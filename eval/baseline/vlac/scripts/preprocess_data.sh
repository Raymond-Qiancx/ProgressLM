#!/bin/bash

# This script splits the main JSONL dataset into smaller files based on task_type.

# Get the directory of this script in a portable way
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))

# --- Configuration ---
# Default input file. Can be overridden by command line argument.
INPUT_JSONL="/home/vcj9002/jianshu/workspace/code/ProgressLM/data/h5_tienkung_xsens_converted.jsonl"
# Default output directory.
OUTPUT_DIR="/home/vcj9002/jianshu/workspace/code/ProgressLM/data/h5_tienkung_xsens_split"
# ---------------------

# Allow overriding input file from command line
if [ "$1" ]; then
    INPUT_JSONL="$1"
fi

SPLITTER_SCRIPT="${SCRIPT_DIR}/split_jsonl.py"

echo "Starting JSONL splitting..."
echo "Input file: ${INPUT_JSONL}"
echo "Output directory: ${OUTPUT_DIR}"

# Run the Python splitter script
python3 "${SPLITTER_SCRIPT}" --input_jsonl "${INPUT_JSONL}" --output_dir "${OUTPUT_DIR}"

echo "Splitting complete."
