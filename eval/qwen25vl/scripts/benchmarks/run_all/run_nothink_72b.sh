#!/bin/bash

#####################################################################
# Run All Nothink Benchmarks - 72B Model
#
# This script runs all nothink evaluation scripts for 72B model
# across all benchmarks (normal_view, multi_view, nega_text, normal_text, edit_nega)
#####################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"

echo "======================================================================"
echo "Running All Nothink Benchmarks - 72B Model"
echo "======================================================================"

echo ""
echo "==================== 72B Models ===================="

echo "[1/5] Running normal_view 72B nothink..."
bash "$BENCHMARK_DIR/normal_view/visual_eval_one_view_nothink_72B.sh"

echo "[2/5] Running multi_view 72B nothink..."
bash "$BENCHMARK_DIR/multi_view/visual_eval_multi_view_nothink_72B.sh"

echo "[3/5] Running nega_text 72B nothink..."
bash "$BENCHMARK_DIR/nega_text/nothink_72b.sh"

echo "[4/5] Running normal_text 72B nothink..."
bash "$BENCHMARK_DIR/normal_text/nothink_72b.sh"

echo "[5/5] Running edit_nega 72B nothink..."
bash "$BENCHMARK_DIR/edit_nega/nothink_72b.sh"

echo ""
echo "======================================================================"
echo "All 72B Nothink Benchmarks Completed!"
echo "======================================================================"
