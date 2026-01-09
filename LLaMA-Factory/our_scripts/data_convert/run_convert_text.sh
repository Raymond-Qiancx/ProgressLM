#!/bin/bash
################################################################################
# Convert Text Demo datasets to LLaMA-Factory format
#
# This script converts Text Demo datasets by merging original data
# with CoT responses into ShareGPT format.
#
# Usage:
#   # Single file mode:
#   bash run_convert_text.sh <dataset_name> <original_file> <cot_file>
#
#   # Batch mode:
#   bash run_convert_text.sh
#
# Configuration:
#   - Edit ORIGINAL_DIR: Directory containing original *_sft.jsonl files
#   - Edit COT_DIR: Directory containing CoT response *_cot.jsonl files
#   - Edit OUTPUT_DIR: Directory for output JSON files
################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================== Configuration ====================
# Original data directory
ORIGINAL_DIR="/projects/p32958/chengxuan/ProgressLM/data/train/text_demo/new"

# CoT responses directory (UPDATE THIS PATH!)
COT_DIR="/projects/p32958/chengxuan/results/progresslm/nega_text/think_72B"

# Output directory
OUTPUT_DIR="/projects/p32958/chengxuan/ProgressLM/data/sft_data/text"

# Dataset configurations: "dataset_name|original_file|cot_file"
DATASETS=(
    "text_h5_agilex_3rgb|text_h5_agilex_3rgb_sft.jsonl|text_agilex_cold.jsonl"
    "text_coin|text_coin_sft.jsonl|text_coin_cold.jsonl"
    "text_h5_franka_3rgb|text_h5_franka_3rgb_sft.jsonl|text_franka_cold.jsonl"
    "text_h5_tienkung_xsens|text_h5_tienkung_xsens_sft.jsonl|text_tienkung_cold.jsonl"
)

# ==================== Parse Arguments ====================
if [ $# -eq 3 ]; then
    # Single file mode
    SINGLE_MODE=true
    SINGLE_DATASET_NAME="$1"
    SINGLE_ORIGINAL_FILE="$2"
    SINGLE_COT_FILE="$3"
    echo "Running in SINGLE FILE mode"
elif [ $# -eq 0 ]; then
    # Batch mode
    SINGLE_MODE=false
    echo "Running in BATCH mode"
else
    echo "❌ Error: Invalid arguments"
    echo ""
    echo "Usage:"
    echo "  Single file: bash run_convert_text.sh <dataset_name> <original_file> <cot_file>"
    echo "  Batch mode:  bash run_convert_text.sh"
    echo ""
    echo "Example:"
    echo "  bash run_convert_text.sh text_h5_agilex_3rgb text_h5_agilex_3rgb_sft.jsonl text_agilex_cold.jsonl"
    exit 1
fi

# ==================== Validation ====================
echo "========================================"
echo "Text Demo SFT Data Conversion"
echo "========================================"
echo "Script directory: $SCRIPT_DIR"
echo "Original data:    $ORIGINAL_DIR"
echo "CoT responses:    $COT_DIR"
echo "Output:           $OUTPUT_DIR"
echo ""

# Check directories
if [ ! -d "$ORIGINAL_DIR" ]; then
    echo "❌ Error: Original data directory not found: $ORIGINAL_DIR"
    exit 1
fi

if [ ! -d "$COT_DIR" ]; then
    echo "⚠️  Warning: CoT directory not found: $COT_DIR"
    echo "Please update COT_DIR in this script to point to your CoT responses directory"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ==================== Process Datasets ====================
SUCCESS_COUNT=0
FAILED_COUNT=0

# Build dataset list based on mode
if [ "$SINGLE_MODE" = true ]; then
    # Single file mode: create a one-element array
    PROCESS_DATASETS=("${SINGLE_DATASET_NAME}|${SINGLE_ORIGINAL_FILE}|${SINGLE_COT_FILE}")
else
    # Batch mode: use the predefined DATASETS array
    PROCESS_DATASETS=("${DATASETS[@]}")
fi

TOTAL_DATASETS=${#PROCESS_DATASETS[@]}

for dataset_config in "${PROCESS_DATASETS[@]}"; do
    IFS='|' read -r dataset_name original_file cot_file <<< "$dataset_config"

    echo ""
    echo "----------------------------------------"
    echo "Processing: $dataset_name"
    echo "----------------------------------------"

    ORIGINAL_PATH="$ORIGINAL_DIR/$original_file"
    COT_PATH="$COT_DIR/$cot_file"
    OUTPUT_PATH="$OUTPUT_DIR/${dataset_name}_llamafactory.json"

    # Check if files exist
    if [ ! -f "$ORIGINAL_PATH" ]; then
        echo "⚠️  Skipping: Original file not found: $ORIGINAL_PATH"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi

    if [ ! -f "$COT_PATH" ]; then
        echo "⚠️  Skipping: CoT file not found: $COT_PATH"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi

    echo "  Original: $ORIGINAL_PATH"
    echo "  CoT:      $COT_PATH"
    echo "  Output:   $OUTPUT_PATH"
    echo ""

    # Run conversion
    python "$SCRIPT_DIR/convert_text_demo.py" \
        --original-data "$ORIGINAL_PATH" \
        --cot-responses "$COT_PATH" \
        --output-file "$OUTPUT_PATH" \
        --filter-success \
        --verbose

    if [ $? -eq 0 ]; then
        echo "✅ Successfully converted: $dataset_name"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        # Run validation
        echo ""
        echo "Validating output..."
        python "$SCRIPT_DIR/validate_output.py" \
            --input-file "$OUTPUT_PATH" \
            --show-samples 0

        if [ $? -eq 0 ]; then
            echo "✅ Validation passed"
        else
            echo "⚠️  Validation found issues (but conversion succeeded)"
        fi
    else
        echo "❌ Failed to convert: $dataset_name"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

# ==================== Summary ====================
echo ""
echo "========================================"
echo "Batch Conversion Summary"
echo "========================================"
echo "Total datasets: $TOTAL_DATASETS"
echo "Successful:     $SUCCESS_COUNT"
echo "Failed:         $FAILED_COUNT"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "Output files saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Register datasets in LLaMA-Factory/data/dataset_info.json"
    echo "  2. Update training config to use: dataset: text_h5_agilex_3rgb_llamafactory,..."
    echo "  3. Run training: bash LLaMA-Factory/our_scripts/train_qwen2_5vl_lora_sft.sh"
fi

echo "========================================"

if [ $FAILED_COUNT -gt 0 ]; then
    exit 1
fi
