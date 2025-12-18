#!/bin/bash
#####################################################################
# Run All Benchmarks - All GPT-5 Models
#####################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "Running All GPT-5 Model Benchmarks"
echo "======================================================================"

echo ""
echo "========== GPT-5 =========="
bash "$SCRIPT_DIR/run_gpt5.sh"

echo ""
echo "========== GPT-5-mini =========="
bash "$SCRIPT_DIR/run_gpt5_mini.sh"

echo ""
echo "========== GPT-5-nano =========="
bash "$SCRIPT_DIR/run_gpt5_nano.sh"

echo ""
echo "======================================================================"
echo "All Model Benchmarks Completed"
echo "======================================================================"
