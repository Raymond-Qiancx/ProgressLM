#!/bin/bash
#####################################################################
# Run All Benchmarks - Qwen2.5-VL-7B (Think Mode)
#####################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(dirname "$SCRIPT_DIR")"

echo "======================================================================"
echo "Running All Benchmarks - Qwen2.5-VL-7B (Think Mode)"
echo "======================================================================"

# Define benchmarks to run
BENCHMARKS=(
    "normal_view"
    "multi_view"
    "normal_text"
    "nega_text"
    "edit_nega"
)

# Run each benchmark
for benchmark in "${BENCHMARKS[@]}"; do
    SCRIPT_PATH="${SCRIPTS_ROOT}/${benchmark}/think_7b.sh"

    if [ -f "$SCRIPT_PATH" ]; then
        echo ""
        echo "======================================================================"
        echo "Starting: ${benchmark} - Qwen2.5-VL-7B (Think Mode)"
        echo "======================================================================"
        bash "$SCRIPT_PATH"
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "Warning: ${benchmark} failed with exit code $EXIT_CODE"
        fi
    else
        echo "Warning: Script not found: $SCRIPT_PATH"
    fi
done

echo ""
echo "======================================================================"
echo "All Qwen2.5-VL-7B (Think Mode) benchmarks completed!"
echo "======================================================================"
