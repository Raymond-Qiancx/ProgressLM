#!/bin/bash

#####################################################################
# Run All Visual Edit Negation NoThink Benchmarks
#
# This script runs all nothink evaluation scripts for visual_edit_nega
# across all model sizes (4B, 14B, 38B)
#####################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "Running All Visual Edit Negation NoThink Benchmarks"
echo "======================================================================"

echo ""
echo "[1/3] Running InternVL 4B nothink..."
bash "$SCRIPT_DIR/eval_nothink_4B.sh"

echo ""
echo "[2/3] Running InternVL 14B nothink..."
bash "$SCRIPT_DIR/eval_nothink_14B.sh"

echo ""
echo "[3/3] Running InternVL 38B nothink..."
bash "$SCRIPT_DIR/eval_nothink_38B.sh"

echo ""
echo "======================================================================"
echo "All Visual Edit Negation NoThink Benchmarks Completed!"
echo "======================================================================"
