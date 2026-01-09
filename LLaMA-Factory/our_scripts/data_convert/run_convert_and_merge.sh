#!/bin/bash
################################################################################
# Batch Convert and Merge All CoT Datasets
#
# This script automatically:
# 1. Converts all CoT response files using appropriate converters
# 2. Merges text_* datasets into all_text_demos_merged.json
# 3. Merges visual_* datasets into all_visual_demos_merged.json
#
# Usage:
#   bash run_convert_and_merge.sh
################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================== Configuration ====================
# CoT responses directory
COT_DIR="/projects/p32958/chengxuan/ProgressLM/data/sft_data/all_new_think"

# Original SFT data directory
ORIGINAL_DIR="/projects/p32958/chengxuan/ProgressLM/data/train/sft/now"

# Output directories
TEMP_OUTPUT_DIR="/projects/p32958/chengxuan/ProgressLM/data/sft_data/merged/temp"
FINAL_OUTPUT_DIR="/projects/p32958/chengxuan/ProgressLM/data/sft_data/merged"

# Dataset mappings: "cot_file|original_file|type"
# type: text or visual
declare -A DATASET_MAPPINGS=(
    ["text_extend_normal_think.jsonl"]="text_normal.jsonl|text"
    ["text_nega_new_think.jsonl"]="text_nega_new_sft.jsonl|text"
    ["text_normal_think.jsonl"]="text_normal.jsonl|text"
    ["visual_edit_think.jsonl"]="visual_edit_sft.jsonl|visual"
    ["visual_multi_view_think.jsonl"]="visual_multi_view_sft.jsonl|visual"
    ["visual_single_view_think.jsonl"]="visual_normal_sft.jsonl|visual"
)

# ==================== Validation ====================
echo "========================================"
echo "Batch Convert and Merge Pipeline"
echo "========================================"
echo "Script directory: $SCRIPT_DIR"
echo "CoT directory:    $COT_DIR"
echo "Original data:    $ORIGINAL_DIR"
echo "Temp output:      $TEMP_OUTPUT_DIR"
echo "Final output:     $FINAL_OUTPUT_DIR"
echo ""

# Check directories
if [ ! -d "$COT_DIR" ]; then
    echo "❌ Error: CoT directory not found: $COT_DIR"
    exit 1
fi

if [ ! -d "$ORIGINAL_DIR" ]; then
    echo "❌ Error: Original data directory not found: $ORIGINAL_DIR"
    exit 1
fi

# Create output directories
mkdir -p "$TEMP_OUTPUT_DIR"
mkdir -p "$FINAL_OUTPUT_DIR"

# ==================== Step 1: Convert Individual Datasets ====================
echo ""
echo "========================================"
echo "Step 1: Converting Individual Datasets"
echo "========================================"

SUCCESS_COUNT=0
FAILED_COUNT=0
TEXT_OUTPUT_FILES=()
VISUAL_OUTPUT_FILES=()

for cot_file in "${!DATASET_MAPPINGS[@]}"; do
    mapping="${DATASET_MAPPINGS[$cot_file]}"
    IFS='|' read -r original_file dataset_type <<< "$mapping"

    # Extract dataset name (remove _think.jsonl suffix)
    dataset_name="${cot_file%_think.jsonl}"

    echo ""
    echo "----------------------------------------"
    echo "Processing: $dataset_name ($dataset_type)"
    echo "----------------------------------------"

    COT_PATH="$COT_DIR/$cot_file"
    ORIGINAL_PATH="$ORIGINAL_DIR/$original_file"
    OUTPUT_PATH="$TEMP_OUTPUT_DIR/${dataset_name}_llamafactory.json"

    # Check if files exist
    if [ ! -f "$COT_PATH" ]; then
        echo "⚠️  Skipping: CoT file not found: $COT_PATH"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi

    if [ ! -f "$ORIGINAL_PATH" ]; then
        echo "⚠️  Skipping: Original file not found: $ORIGINAL_PATH"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi

    echo "  CoT:      $COT_PATH"
    echo "  Original: $ORIGINAL_PATH"
    echo "  Output:   $OUTPUT_PATH"
    echo "  Type:     $dataset_type"
    echo ""

    # Choose converter based on type
    if [ "$dataset_type" = "text" ]; then
        CONVERTER="$SCRIPT_DIR/convert_text_demo.py"
    elif [ "$dataset_type" = "visual" ]; then
        CONVERTER="$SCRIPT_DIR/convert_visual_demo.py"
    else
        echo "❌ Error: Unknown dataset type: $dataset_type"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi

    # Run conversion
    python "$CONVERTER" \
        --original-data "$ORIGINAL_PATH" \
        --cot-responses "$COT_PATH" \
        --output-file "$OUTPUT_PATH" \
        --filter-success \
        --verbose

    if [ $? -eq 0 ]; then
        echo "✅ Successfully converted: $dataset_name"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        # Add to appropriate merge list
        if [ "$dataset_type" = "text" ]; then
            TEXT_OUTPUT_FILES+=("$OUTPUT_PATH")
        elif [ "$dataset_type" = "visual" ]; then
            VISUAL_OUTPUT_FILES+=("$OUTPUT_PATH")
        fi
    else
        echo "❌ Failed to convert: $dataset_name"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

# ==================== Step 1 Summary ====================
echo ""
echo "========================================"
echo "Step 1 Summary"
echo "========================================"
echo "Total datasets:  ${#DATASET_MAPPINGS[@]}"
echo "Successful:      $SUCCESS_COUNT"
echo "Failed:          $FAILED_COUNT"
echo "Text datasets:   ${#TEXT_OUTPUT_FILES[@]}"
echo "Visual datasets: ${#VISUAL_OUTPUT_FILES[@]}"
echo ""

if [ $SUCCESS_COUNT -eq 0 ]; then
    echo "❌ No datasets were successfully converted. Exiting."
    exit 1
fi

# ==================== Step 2: Merge Text Datasets ====================
if [ ${#TEXT_OUTPUT_FILES[@]} -gt 0 ]; then
    echo ""
    echo "========================================"
    echo "Step 2: Merging Text Datasets"
    echo "========================================"

    TEXT_MERGED_OUTPUT="$FINAL_OUTPUT_DIR/all_text_demos_merged.json"

    echo "Merging ${#TEXT_OUTPUT_FILES[@]} text datasets..."
    echo "Output: $TEXT_MERGED_OUTPUT"
    echo ""

    python "$SCRIPT_DIR/merge_datasets.py" \
        --input-files "${TEXT_OUTPUT_FILES[@]}" \
        --output-file "$TEXT_MERGED_OUTPUT" \
        --verbose

    if [ $? -eq 0 ]; then
        echo "✅ Text datasets merged successfully"
        TEXT_MERGE_SUCCESS=true
    else
        echo "❌ Failed to merge text datasets"
        TEXT_MERGE_SUCCESS=false
    fi
else
    echo ""
    echo "⚠️  No text datasets to merge"
    TEXT_MERGE_SUCCESS=false
fi

# ==================== Step 3: Merge Visual Datasets ====================
if [ ${#VISUAL_OUTPUT_FILES[@]} -gt 0 ]; then
    echo ""
    echo "========================================"
    echo "Step 3: Merging Visual Datasets"
    echo "========================================"

    VISUAL_MERGED_OUTPUT="$FINAL_OUTPUT_DIR/all_visual_demos_merged.json"

    echo "Merging ${#VISUAL_OUTPUT_FILES[@]} visual datasets..."
    echo "Output: $VISUAL_MERGED_OUTPUT"
    echo ""

    python "$SCRIPT_DIR/merge_datasets.py" \
        --input-files "${VISUAL_OUTPUT_FILES[@]}" \
        --output-file "$VISUAL_MERGED_OUTPUT" \
        --verbose

    if [ $? -eq 0 ]; then
        echo "✅ Visual datasets merged successfully"
        VISUAL_MERGE_SUCCESS=true
    else
        echo "❌ Failed to merge visual datasets"
        VISUAL_MERGE_SUCCESS=false
    fi
else
    echo ""
    echo "⚠️  No visual datasets to merge"
    VISUAL_MERGE_SUCCESS=false
fi

# ==================== Final Summary ====================
echo ""
echo "========================================"
echo "Pipeline Completion Summary"
echo "========================================"
echo "Individual conversions: $SUCCESS_COUNT successful, $FAILED_COUNT failed"

if [ "$TEXT_MERGE_SUCCESS" = true ]; then
    echo "Text merge:             ✅ $TEXT_MERGED_OUTPUT"
fi

if [ "$VISUAL_MERGE_SUCCESS" = true ]; then
    echo "Visual merge:           ✅ $VISUAL_MERGED_OUTPUT"
fi

echo ""
echo "Temporary files:        $TEMP_OUTPUT_DIR"
echo "Final outputs:          $FINAL_OUTPUT_DIR"
echo ""

if [ "$TEXT_MERGE_SUCCESS" = true ] || [ "$VISUAL_MERGE_SUCCESS" = true ]; then
    echo "Next steps:"
    echo "  1. Validate merged files:"
    if [ "$TEXT_MERGE_SUCCESS" = true ]; then
        echo "     python $SCRIPT_DIR/validate_output.py --input-file $TEXT_MERGED_OUTPUT --show-samples 1"
    fi
    if [ "$VISUAL_MERGE_SUCCESS" = true ]; then
        echo "     python $SCRIPT_DIR/validate_output.py --input-file $VISUAL_MERGED_OUTPUT --show-samples 1"
    fi
    echo ""
    echo "  2. Register datasets in LLaMA-Factory/data/dataset_info.json"
    echo "  3. Update training config and run training"
fi

echo "========================================"

# Clean up temp files (optional - comment out if you want to keep them)
# echo ""
# echo "Cleaning up temporary files..."
# rm -rf "$TEMP_OUTPUT_DIR"
# echo "✅ Cleanup complete"

if [ "$TEXT_MERGE_SUCCESS" = true ] || [ "$VISUAL_MERGE_SUCCESS" = true ]; then
    echo ""
    echo "✅ Pipeline completed successfully!"
    exit 0
else
    echo ""
    echo "❌ Pipeline completed with errors"
    exit 1
fi
