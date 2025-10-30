#!/bin/bash

# Qwen3VL Visual Demo Progress Estimation - Multi-GPU Batch Inference
# This script runs the Qwen3VL model for visual demo progress estimation

# ===========================
# Configuration
# ===========================

# Model path (change this to your Qwen3VL model path)
MODEL_PATH="/path/to/Qwen3-VL-8B-Instruct"  # e.g., "Qwen/Qwen3-VL-8B-Instruct"

# Dataset configuration
DATASET_PATH="/path/to/visual_demo_dataset.jsonl"
IMAGE_ROOT="/path/to/image/root"  # Optional: root directory for relative image paths

# Output configuration
OUTPUT_DIR="./outputs"
OUTPUT_FILE="${OUTPUT_DIR}/qwen3vl_results.jsonl"

# GPU configuration
# Set CUDA_VISIBLE_DEVICES to specify which GPUs to use
# Examples:
#   Single GPU: export CUDA_VISIBLE_DEVICES=0
#   Multiple GPUs: export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Inference configuration
BATCH_SIZE=8              # Batch size per GPU (adjust based on GPU memory)
NUM_INFERENCES=4          # Number of inferences per sample (for diversity)
LIMIT=-1                  # Limit number of samples (-1 for all)

# Model parameters (for diversity/creativity)
TEMPERATURE=0.7           # Sampling temperature (0.0-1.0)
TOP_P=0.9                 # Top-p sampling
TOP_K=50                  # Top-k sampling
MAX_NEW_TOKENS=512        # Maximum tokens to generate

# Image processing parameters (Qwen3VL specific)
# Qwen3VL uses patch_size=16, so multiply by 16*16=256
MIN_PIXELS=$((256 * 256))        # Min pixels: 256 visual tokens
MAX_PIXELS=$((1280 * 256))       # Max pixels: 1280 visual tokens

# ===========================
# Run Inference
# ===========================

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "======================================"
echo "Qwen3VL Visual Demo Progress Estimation"
echo "======================================"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_FILE}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo "Num inferences: ${NUM_INFERENCES}"
echo "======================================"
echo ""

# Run the inference script
python run.py \
    --model-path "${MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --output-file "${OUTPUT_FILE}" \
    --image-root "${IMAGE_ROOT}" \
    --batch-size ${BATCH_SIZE} \
    --num-inferences ${NUM_INFERENCES} \
    --limit ${LIMIT} \
    --temperature ${TEMPERATURE} \
    --top-p ${TOP_P} \
    --top-k ${TOP_K} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --min-pixels ${MIN_PIXELS} \
    --max-pixels ${MAX_PIXELS}

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Inference completed successfully!"
    echo "Results saved to: ${OUTPUT_FILE}"
    echo "Summary saved to: ${OUTPUT_FILE%.jsonl}_summary.json"
    echo "======================================"
else
    echo ""
    echo "======================================"
    echo "ERROR: Inference failed!"
    echo "======================================"
    exit 1
fi
