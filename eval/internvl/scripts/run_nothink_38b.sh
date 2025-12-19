#!/bin/bash

#####################################################################
# InternVL Nothink Evaluation - 38B Model
#
# This script runs all nothink benchmarks for 38B model.
#####################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "InternVL Nothink Evaluation - 38B Model"
echo "======================================================================"
echo ""

# ======================== Visual Normal ========================
echo "========== Visual Normal - 38B =========="
bash "$SCRIPT_DIR/visual_normal/eval_nothink_38B.sh"

# ======================== Visual Multi ========================
echo "========== Visual Multi - 38B =========="
bash "$SCRIPT_DIR/visual_multi/eval_nothink_38B.sh"

# ======================== Text Normal ========================
echo "========== Text Normal - 38B =========="
bash "$SCRIPT_DIR/text_normal/eval_nothink_38B.sh"

# ======================== Text Nega ========================
echo "========== Text Nega - 38B =========="
bash "$SCRIPT_DIR/text_nega/eval_nothink_38B.sh"

echo ""
echo "======================================================================"
echo "All 38B nothink benchmarks completed!"
echo "======================================================================"
