#!/bin/bash
#####################################################################
# Run All Human Activities Benchmarks - Qwen2.5-VL (3B, 7B, 32B, 72B)
#####################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "Running All Human Activities Benchmarks - Qwen2.5-VL"
echo "======================================================================"

# Define all scripts to run
SCRIPTS=(
    # 3B
    "visual_think_3b.sh"
    "visual_nothink_3b.sh"
    "text_think_3b.sh"
    "text_nothink_3b.sh"
    # 7B
    "visual_think_7b.sh"
    "visual_nothink_7b.sh"
    "text_think_7b.sh"
    "text_nothink_7b.sh"
    # 32B
    "visual_think_32b.sh"
    "visual_nothink_32b.sh"
    "text_think_32b.sh"
    "text_nothink_32b.sh"
    # 72B
    "visual_think_72b.sh"
    "visual_nothink_72b.sh"
    "text_think_72b.sh"
    "text_nothink_72b.sh"
)

# Run each script
for script in "${SCRIPTS[@]}"; do
    SCRIPT_PATH="${SCRIPT_DIR}/${script}"

    if [ -f "$SCRIPT_PATH" ]; then
        echo ""
        echo "======================================================================"
        echo "Starting: ${script}"
        echo "======================================================================"
        bash "$SCRIPT_PATH"
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "Warning: ${script} failed with exit code $EXIT_CODE"
        fi
    else
        echo "Warning: Script not found: $SCRIPT_PATH"
    fi
done

echo ""
echo "======================================================================"
echo "All Qwen2.5-VL Human Activities benchmarks completed!"
echo "======================================================================"
