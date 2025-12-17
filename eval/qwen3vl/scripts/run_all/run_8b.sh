#!/bin/bash
#####################################################################
# Run All Qwen3VL-8B Benchmarks
#####################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(dirname "$SCRIPT_DIR")"

echo "======================================================================"
echo "Running All Qwen3VL-8B Benchmarks"
echo "======================================================================"
echo "Script Root: $SCRIPTS_ROOT"
echo "======================================================================"

# Track overall status
FAILED_BENCHMARKS=()

# Function to run a benchmark
run_benchmark() {
    local benchmark_dir=$1
    local script_name=$2
    local script_path="${SCRIPTS_ROOT}/${benchmark_dir}/${script_name}"

    echo ""
    echo "----------------------------------------------------------------------"
    echo "Running: ${benchmark_dir}/${script_name}"
    echo "----------------------------------------------------------------------"

    if [ -f "$script_path" ]; then
        bash "$script_path"
        if [ $? -ne 0 ]; then
            FAILED_BENCHMARKS+=("${benchmark_dir}/${script_name}")
            echo "WARNING: ${benchmark_dir}/${script_name} failed!"
        fi
    else
        echo "ERROR: Script not found: $script_path"
        FAILED_BENCHMARKS+=("${benchmark_dir}/${script_name} (not found)")
    fi
}

# Run all 8B benchmarks (thinking version)
echo ""
echo "======================================================================"
echo "Running 8B Thinking Benchmarks"
echo "======================================================================"
run_benchmark "normal_view" "qwen3vl_8b.sh"
run_benchmark "multi_view" "qwen3vl_8b.sh"
run_benchmark "normal_text" "qwen3vl_8b.sh"
run_benchmark "nega_text" "qwen3vl_8b.sh"
run_benchmark "edit_nega" "qwen3vl_8b.sh"

# Run all 8B benchmarks (nothink version)
# echo ""
# echo "======================================================================"
# echo "Running 8B NoThink Benchmarks"
# echo "======================================================================"
# run_benchmark "normal_view" "qwen3vl_8b_nothink.sh"
# run_benchmark "multi_view" "qwen3vl_8b_nothink.sh"
# run_benchmark "normal_text" "qwen3vl_8b_nothink.sh"
# run_benchmark "nega_text" "qwen3vl_8b_nothink.sh"
# run_benchmark "edit_nega" "qwen3vl_8b_nothink.sh"

# Summary
echo ""
echo "======================================================================"
echo "Summary"
echo "======================================================================"
if [ ${#FAILED_BENCHMARKS[@]} -eq 0 ]; then
    echo "All benchmarks completed successfully!"
else
    echo "Failed benchmarks:"
    for benchmark in "${FAILED_BENCHMARKS[@]}"; do
        echo "  - $benchmark"
    done
    exit 1
fi
