#!/bin/bash
#####################################################################
# Run All Benchmarks - Qwen3VL-2B (NoThink)
#####################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(dirname "$SCRIPT_DIR")"

echo "======================================================================"
echo "Running All Benchmarks - Qwen3VL-2B (NoThink)"
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
    SCRIPT_PATH="${SCRIPTS_ROOT}/${benchmark}/qwen3vl_2b_nothink.sh"

    if [ -f "$SCRIPT_PATH" ]; then
        echo ""
        echo "======================================================================"
        echo "Starting: ${benchmark} - Qwen3VL-2B (NoThink)"
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
echo "All Qwen3VL-2B (NoThink) benchmarks completed!"
echo "======================================================================"
