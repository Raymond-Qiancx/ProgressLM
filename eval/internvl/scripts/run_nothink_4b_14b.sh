#!/bin/bash

#####################################################################
# InternVL Nothink Evaluation - 4B and 14B Models
#
# This script runs all nothink benchmarks for 4B and 14B models.
# 4B: batch_size=20, 14B: batch_size=10
#####################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "InternVL Nothink Evaluation - 4B and 14B Models"
echo "======================================================================"
echo ""

# ======================== Visual Normal ========================
# echo "========== Visual Normal - 4B =========="
# bash "$SCRIPT_DIR/visual_normal/eval_nothink_4B.sh"

# echo "========== Visual Normal - 14B =========="
# bash "$SCRIPT_DIR/visual_normal/eval_nothink_14B.sh"

# # ======================== Visual Multi ========================
# echo "========== Visual Multi - 4B =========="
# bash "$SCRIPT_DIR/visual_multi/eval_nothink_4B.sh"

# echo "========== Visual Multi - 14B =========="
# bash "$SCRIPT_DIR/visual_multi/eval_nothink_14B.sh"

# # ======================== Text Normal ========================
# echo "========== Text Normal - 4B =========="
# bash "$SCRIPT_DIR/text_normal/eval_nothink_4B.sh"

echo "========== Text Normal - 14B =========="
bash "$SCRIPT_DIR/text_normal/eval_nothink_14B.sh"

# ======================== Text Nega ========================
echo "========== Text Nega - 4B =========="
bash "$SCRIPT_DIR/text_nega/eval_nothink_4B.sh"

echo "========== Text Nega - 14B =========="
bash "$SCRIPT_DIR/text_nega/eval_nothink_14B.sh"

echo ""
echo "======================================================================"
echo "All 4B and 14B nothink benchmarks completed!"
echo "======================================================================"
