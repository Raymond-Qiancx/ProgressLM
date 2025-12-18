#!/bin/bash
#####################################################################
# Run All Benchmarks (NoThink) - GPT-5
#####################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")"

echo "======================================================================"
echo "Running All GPT-5 Benchmarks (NoThink)"
echo "======================================================================"

# Run all benchmarks sequentially
echo ""
echo "[1/5] Running Normal Text Demo (NoThink)..."
bash "$SCRIPTS_DIR/normal_text/gpt5_nothink.sh"
EXIT_CODE=$?
[ $EXIT_CODE -ne 0 ] && echo "Warning: Normal Text Demo failed with exit code $EXIT_CODE"

echo ""
echo "[2/5] Running Normal View Demo (NoThink)..."
bash "$SCRIPTS_DIR/normal_view/gpt5_nothink.sh"
EXIT_CODE=$?
[ $EXIT_CODE -ne 0 ] && echo "Warning: Normal View Demo failed with exit code $EXIT_CODE"

echo ""
echo "[3/5] Running Multi View Demo (NoThink)..."
bash "$SCRIPTS_DIR/multi_view/gpt5_nothink.sh"
EXIT_CODE=$?
[ $EXIT_CODE -ne 0 ] && echo "Warning: Multi View Demo failed with exit code $EXIT_CODE"

echo ""
echo "[4/5] Running Negative Text Demo (NoThink)..."
bash "$SCRIPTS_DIR/nega_text/gpt5_nothink.sh"
EXIT_CODE=$?
[ $EXIT_CODE -ne 0 ] && echo "Warning: Negative Text Demo failed with exit code $EXIT_CODE"

echo ""
echo "[5/5] Running Edit Negative Demo (NoThink)..."
bash "$SCRIPTS_DIR/edit_nega/gpt5_nothink.sh"
EXIT_CODE=$?
[ $EXIT_CODE -ne 0 ] && echo "Warning: Edit Negative Demo failed with exit code $EXIT_CODE"

echo ""
echo "======================================================================"
echo "All GPT-5 Benchmarks (NoThink) Completed"
echo "======================================================================"
