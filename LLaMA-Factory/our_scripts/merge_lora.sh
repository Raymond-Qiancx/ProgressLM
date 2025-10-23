#!/bin/bash

################################################################################
# LoRA Weight Merging Script
#
# This script merges LoRA adapter weights with the base model to create
# a standalone HuggingFace model that can be used directly.
#
# Usage:
#   bash merge_lora.sh [checkpoint_name]
#
# Examples:
#   bash merge_lora.sh                    # Merge the final checkpoint
#   bash merge_lora.sh checkpoint-500     # Merge a specific checkpoint
################################################################################

set -e  # Exit on error

# ==================== Configuration ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${SCRIPT_DIR}/merge_lora_custom.yaml"

# Default checkpoint (final model)
CHECKPOINT_NAME="${1:-}"

# Model paths from config
BASE_MODEL="/projects/b1222/userdata/jianshu/chengxuan/saved/models/Qwen2.5-VL-3B-Instruct"
LORA_DIR="/projects/b1222/userdata/jianshu/chengxuan/saved/saved_results/progresslm/models/sft"
OUTPUT_DIR="/projects/b1222/userdata/jianshu/chengxuan/saved/saved_results/progresslm/models/merged_model"

echo "=========================================="
echo "LoRA Weight Merging"
echo "=========================================="
echo "Base Model:   $BASE_MODEL"
echo "LoRA Adapter: $LORA_DIR"
echo "Output Dir:   $OUTPUT_DIR"
echo "=========================================="

# ==================== Find Checkpoint ====================
if [ -n "$CHECKPOINT_NAME" ]; then
    # User specified a checkpoint
    ADAPTER_PATH="${LORA_DIR}/${CHECKPOINT_NAME}"
    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "Error: Checkpoint not found: $ADAPTER_PATH"
        echo ""
        echo "Available checkpoints:"
        ls -1 "$LORA_DIR" | grep checkpoint || echo "  (none found)"
        exit 1
    fi
    OUTPUT_DIR="${OUTPUT_DIR}_${CHECKPOINT_NAME}"
else
    # Use the final checkpoint (highest numbered one)
    LATEST_CHECKPOINT=$(ls -1 "$LORA_DIR" | grep -E '^checkpoint-[0-9]+$' | sort -V | tail -n 1)

    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "Error: No checkpoints found in $LORA_DIR"
        exit 1
    fi

    ADAPTER_PATH="${LORA_DIR}/${LATEST_CHECKPOINT}"
    OUTPUT_DIR="${OUTPUT_DIR}_${LATEST_CHECKPOINT}"
    echo "Using latest checkpoint: $LATEST_CHECKPOINT"
fi

echo "Merging from: $ADAPTER_PATH"
echo "Output to:    $OUTPUT_DIR"
echo ""

# ==================== Create Temporary Config ====================
TEMP_CONFIG=$(mktemp)
cat > "$TEMP_CONFIG" <<EOF
### Temporary config for merging specific checkpoint

### Model Configuration
model_name_or_path: $BASE_MODEL
adapter_name_or_path: $ADAPTER_PATH
template: qwen2_vl
trust_remote_code: true

### Finetuning Configuration (required for export)
stage: sft
finetuning_type: lora

### Export Configuration
export_dir: $OUTPUT_DIR
export_size: 5
export_device: cpu
export_legacy_format: false
EOF

echo "Temporary config created at: $TEMP_CONFIG"
echo ""

# ==================== Check Requirements ====================
if [ ! -d "$BASE_MODEL" ]; then
    echo "Error: Base model not found: $BASE_MODEL"
    exit 1
fi

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "Error: LoRA adapter not found: $ADAPTER_PATH"
    exit 1
fi

# ==================== Run Merge ====================
cd "$PROJECT_ROOT"

echo "Starting merge process..."
echo ""
llamafactory-cli export "$TEMP_CONFIG"

# ==================== Clean Up ====================
rm -f "$TEMP_CONFIG"

# ==================== Verify Output ====================
echo ""
echo "=========================================="
echo "Merge completed!"
echo "=========================================="

if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "Merged model saved to: $OUTPUT_DIR"
    echo ""
    echo "Model files:"
    ls -lh "$OUTPUT_DIR"
    echo ""
    echo "You can now use this model directly with:"
    echo ""
    echo "  from transformers import AutoModel, AutoTokenizer"
    echo "  model = AutoModel.from_pretrained('$OUTPUT_DIR', trust_remote_code=True)"
    echo "  tokenizer = AutoTokenizer.from_pretrained('$OUTPUT_DIR', trust_remote_code=True)"
    echo ""
else
    echo "Warning: Output directory not created. Check logs for errors."
    exit 1
fi
