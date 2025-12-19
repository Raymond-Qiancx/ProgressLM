#!/bin/bash

#####################################################################
# Run All Nothink Benchmarks - 3B & 7B Models
#
# This script runs all nothink evaluation scripts for 3B and 7B models
# across all benchmarks (normal_view, multi_view, nega_text, normal_text, edit_nega)
#####################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"

echo "======================================================================"
echo "Running All Nothink Benchmarks - 3B & 7B Models"
echo "======================================================================"

# ======================== 3B Models ========================
echo ""
echo "==================== 3B Models ===================="

echo "[1/10] Running normal_view 3B nothink..."
bash "$BENCHMARK_DIR/normal_view/visual_eval_one_view_nothink_3B.sh"

echo "[2/10] Running multi_view 3B nothink..."
bash "$BENCHMARK_DIR/multi_view/visual_eval_multi_view_nothink_3B.sh"

echo "[3/10] Running nega_text 3B nothink..."
bash "$BENCHMARK_DIR/nega_text/nothink_3b.sh"

echo "[4/10] Running normal_text 3B nothink..."
bash "$BENCHMARK_DIR/normal_text/nothink_3b.sh"

echo "[5/10] Running edit_nega 3B nothink..."
bash "$BENCHMARK_DIR/edit_nega/nothink_3b.sh"

# ======================== 7B Models ========================
echo ""
echo "==================== 7B Models ===================="

echo "[6/10] Running normal_view 7B nothink..."
bash "$BENCHMARK_DIR/normal_view/visual_eval_one_view_nothink_7B.sh"

echo "[7/10] Running multi_view 7B nothink..."
bash "$BENCHMARK_DIR/multi_view/visual_eval_multi_view_nothink_7B.sh"

echo "[8/10] Running nega_text 7B nothink..."
bash "$BENCHMARK_DIR/nega_text/nothink_7b.sh"

echo "[9/10] Running normal_text 7B nothink..."
bash "$BENCHMARK_DIR/normal_text/nothink_7b.sh"

echo "[10/10] Running edit_nega 7B nothink..."
bash "$BENCHMARK_DIR/edit_nega/nothink_7b.sh"

echo ""
echo "======================================================================"
echo "All 3B & 7B Nothink Benchmarks Completed!"
echo "======================================================================"
