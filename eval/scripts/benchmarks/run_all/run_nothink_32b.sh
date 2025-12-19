#!/bin/bash

#####################################################################
# Run All Nothink Benchmarks - 32B Model
#
# This script runs all nothink evaluation scripts for 32B model
# across all benchmarks (normal_view, multi_view, nega_text, normal_text, edit_nega)
#####################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"

echo "======================================================================"
echo "Running All Nothink Benchmarks - 32B Model"
echo "======================================================================"

echo ""
echo "==================== 32B Models ===================="

echo "[1/5] Running normal_view 32B nothink..."
bash "$BENCHMARK_DIR/normal_view/visual_eval_one_view_nothink_32B.sh"

echo "[2/5] Running multi_view 32B nothink..."
bash "$BENCHMARK_DIR/multi_view/visual_eval_multi_view_nothink_32B.sh"

echo "[3/5] Running nega_text 32B nothink..."
bash "$BENCHMARK_DIR/nega_text/nothink_32b.sh"

echo "[4/5] Running normal_text 32B nothink..."
bash "$BENCHMARK_DIR/normal_text/nothink_32b.sh"

# echo "[5/5] Running edit_nega 32B nothink..."
# bash "$BENCHMARK_DIR/edit_nega/nothink_32b.sh"

echo ""
echo "======================================================================"
echo "All 32B Nothink Benchmarks Completed!"
echo "======================================================================"
