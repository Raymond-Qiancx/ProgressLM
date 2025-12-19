#!/bin/bash

#####################################################################
# Run All InternVL NoThink Benchmarks
#
# This script runs all nothink evaluation scripts across all benchmarks
# and all model sizes (4B, 14B, 38B)
#
# Benchmarks:
# - visual_normal: Single-view visual demo
# - visual_multi: Multi-view visual demo
# - text_normal: Normal text demo
# - text_nega: Negation text demo
#####################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "Running All InternVL NoThink Benchmarks"
echo "======================================================================"
echo "Script directory: $SCRIPT_DIR"
echo "======================================================================"

# ==================== Visual Normal ====================
echo ""
echo "========== [1/4] Visual Normal =========="
echo ""

echo "[1.1/1.3] Running visual_normal 4B nothink..."
bash "$SCRIPT_DIR/visual_normal/eval_nothink_4B.sh"

echo ""
echo "[1.2/1.3] Running visual_normal 14B nothink..."
bash "$SCRIPT_DIR/visual_normal/eval_nothink_14B.sh"

echo ""
echo "[1.3/1.3] Running visual_normal 38B nothink..."
bash "$SCRIPT_DIR/visual_normal/eval_nothink_38B.sh"

# ==================== Visual Multi ====================
echo ""
echo "========== [2/4] Visual Multi =========="
echo ""

echo "[2.1/2.3] Running visual_multi 4B nothink..."
bash "$SCRIPT_DIR/visual_multi/eval_nothink_4B.sh"

echo ""
echo "[2.2/2.3] Running visual_multi 14B nothink..."
bash "$SCRIPT_DIR/visual_multi/eval_nothink_14B.sh"

echo ""
echo "[2.3/2.3] Running visual_multi 38B nothink..."
bash "$SCRIPT_DIR/visual_multi/eval_nothink_38B.sh"

# ==================== Text Normal ====================
echo ""
echo "========== [3/4] Text Normal =========="
echo ""

echo "[3.1/3.3] Running text_normal 4B nothink..."
bash "$SCRIPT_DIR/text_normal/eval_nothink_4B.sh"

echo ""
echo "[3.2/3.3] Running text_normal 14B nothink..."
bash "$SCRIPT_DIR/text_normal/eval_nothink_14B.sh"

echo ""
echo "[3.3/3.3] Running text_normal 38B nothink..."
bash "$SCRIPT_DIR/text_normal/eval_nothink_38B.sh"

# ==================== Text Nega ====================
echo ""
echo "========== [4/4] Text Nega =========="
echo ""

echo "[4.1/4.3] Running text_nega 4B nothink..."
bash "$SCRIPT_DIR/text_nega/eval_nothink_4B.sh"

echo ""
echo "[4.2/4.3] Running text_nega 14B nothink..."
bash "$SCRIPT_DIR/text_nega/eval_nothink_14B.sh"

echo ""
echo "[4.3/4.3] Running text_nega 38B nothink..."
bash "$SCRIPT_DIR/text_nega/eval_nothink_38B.sh"

# ==================== Done ====================
echo ""
echo "======================================================================"
echo "All InternVL NoThink Benchmarks Completed!"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  - /projects/p32958/chengxuan/results/internvl/visual_normal_nothink/"
echo "  - /projects/p32958/chengxuan/results/internvl/visual_multi_nothink/"
echo "  - /projects/p32958/chengxuan/results/internvl/text_normal_nothink/"
echo "  - /projects/p32958/chengxuan/results/internvl/text_nega_nothink/"
echo "======================================================================"
