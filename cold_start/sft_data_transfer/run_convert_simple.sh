#!/bin/bash
################################################################################
# Batch Convert SFT Data (Simple Version)
#
# This script converts SFT data using ground truth progress_score as responses,
# without requiring CoT response files.
#
# Usage:
#   bash run_convert_simple.sh
################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================== Configuration ====================
# Input SFT data directory
INPUT_DIR="/projects/p32958/chengxuan/ProgressLM/data/train/sft/now"

# Output directories
TEMP_OUTPUT_DIR="/projects/p32958/chengxuan/ProgressLM/data/sft_data/simple/temp"
FINAL_OUTPUT_DIR="/projects/p32958/chengxuan/ProgressLM/data/sft_data/simple"

# Dataset configurations: "filename|type"
# type: text or visual
declare -A DATASET_CONFIGS=(
    ["text_nega_new_sft.jsonl"]="text"
    ["text_normal_sft.jsonl"]="text"
    ["visual_edit_sft.jsonl"]="visual"
    ["visual_multi_view_sft.jsonl"]="visual"
    ["visual_normal_sft.jsonl"]="visual"
)

# ==================== Validation ====================
echo "========================================"
echo "Simple SFT Data Conversion Pipeline"
echo "========================================"
echo "Script directory: $SCRIPT_DIR"
echo "Input data:       $INPUT_DIR"
echo "Temp output:      $TEMP_OUTPUT_DIR"
echo "Final output:     $FINAL_OUTPUT_DIR"
echo ""

# Check directories
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: Input directory not found: $INPUT_DIR"
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

for input_file in "${!DATASET_CONFIGS[@]}"; do
    dataset_type="${DATASET_CONFIGS[$input_file]}"

    # Extract dataset name (remove _sft.jsonl suffix)
    dataset_name="${input_file%_sft.jsonl}"

    echo ""
    echo "----------------------------------------"
    echo "Processing: $dataset_name ($dataset_type)"
    echo "----------------------------------------"

    INPUT_PATH="$INPUT_DIR/$input_file"
    OUTPUT_PATH="$TEMP_OUTPUT_DIR/${dataset_name}_simple.json"

    # Check if file exists
    if [ ! -f "$INPUT_PATH" ]; then
        echo "⚠️  Skipping: Input file not found: $INPUT_PATH"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi

    echo "  Input:  $INPUT_PATH"
    echo "  Output: $OUTPUT_PATH"
    echo "  Type:   $dataset_type"
    echo ""

    # Choose converter based on type
    if [ "$dataset_type" = "text" ]; then
        CONVERTER="$SCRIPT_DIR/convert_text_demo_simple.py"
    elif [ "$dataset_type" = "visual" ]; then
        CONVERTER="$SCRIPT_DIR/convert_visual_demo_simple.py"
    else
        echo "❌ Error: Unknown dataset type: $dataset_type"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi

    # Run conversion
    python "$CONVERTER" \
        --input-file "$INPUT_PATH" \
        --output-file "$OUTPUT_PATH" \
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
echo "Total datasets:  ${#DATASET_CONFIGS[@]}"
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

    TEXT_MERGED_OUTPUT="$FINAL_OUTPUT_DIR/all_text_simple_merged.json"

    echo "Merging ${#TEXT_OUTPUT_FILES[@]} text datasets..."
    echo "Output: $TEXT_MERGED_OUTPUT"
    echo ""

    python "$SCRIPT_DIR/merge_datasets.py" \
        --input-files "${TEXT_OUTPUT_FILES[@]}" \
        --output-file "$TEXT_MERGED_OUTPUT" \
        --no-validate \
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

    VISUAL_MERGED_OUTPUT="$FINAL_OUTPUT_DIR/all_visual_simple_merged.json"

    echo "Merging ${#VISUAL_OUTPUT_FILES[@]} visual datasets..."
    echo "Output: $VISUAL_MERGED_OUTPUT"
    echo ""

    python "$SCRIPT_DIR/merge_datasets.py" \
        --input-files "${VISUAL_OUTPUT_FILES[@]}" \
        --output-file "$VISUAL_MERGED_OUTPUT" \
        --no-validate \
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

# ==================== Step 4: Merge All to JSONL ====================
if [ "$TEXT_MERGE_SUCCESS" = true ] || [ "$VISUAL_MERGE_SUCCESS" = true ]; then
    echo ""
    echo "========================================"
    echo "Step 4: Creating Combined JSONL"
    echo "========================================"

    ALL_MERGED_FILES=()
    [ "$TEXT_MERGE_SUCCESS" = true ] && ALL_MERGED_FILES+=("$TEXT_MERGED_OUTPUT")
    [ "$VISUAL_MERGE_SUCCESS" = true ] && ALL_MERGED_FILES+=("$VISUAL_MERGED_OUTPUT")

    COMBINED_JSONL="$FINAL_OUTPUT_DIR/all_simple_combined.jsonl"

    echo "Creating combined JSONL from ${#ALL_MERGED_FILES[@]} files..."
    echo "Output: $COMBINED_JSONL"
    echo ""

    python "$SCRIPT_DIR/merge_to_jsonl.py" \
        --input-files "${ALL_MERGED_FILES[@]}" \
        --output-file "$COMBINED_JSONL" \
        --verbose

    if [ $? -eq 0 ]; then
        echo "✅ Combined JSONL created successfully"
        JSONL_SUCCESS=true
    else
        echo "❌ Failed to create combined JSONL"
        JSONL_SUCCESS=false
    fi
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

if [ "$JSONL_SUCCESS" = true ]; then
    echo "Combined JSONL:         ✅ $COMBINED_JSONL"

    # Show sample count
    LINE_COUNT=$(wc -l < "$COMBINED_JSONL")
    echo ""
    echo "Total samples in combined JSONL: $LINE_COUNT"
fi

echo ""
echo "Temporary files:        $TEMP_OUTPUT_DIR"
echo "Final outputs:          $FINAL_OUTPUT_DIR"
echo ""

if [ "$JSONL_SUCCESS" = true ]; then
    echo "Next steps:"
    echo "  1. Validate combined file:"
    echo "     head -1 $COMBINED_JSONL | python3 -m json.tool"
    echo ""
    echo "  2. Register dataset in LLaMA-Factory/data/dataset_info.json"
    echo "  3. Update training config and run training"
fi

echo "========================================"

# Optional: Clean up temp files
# echo ""
# echo "Cleaning up temporary files..."
# rm -rf "$TEMP_OUTPUT_DIR"
# echo "✅ Cleanup complete"

if [ "$JSONL_SUCCESS" = true ]; then
    echo ""
    echo "✅ Pipeline completed successfully!"
    exit 0
else
    echo ""
    echo "⚠️  Pipeline completed with warnings"
    exit 0
fi
