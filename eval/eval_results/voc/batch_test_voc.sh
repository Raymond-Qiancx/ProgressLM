#!/bin/bash
#
# Batch VOC Test Script (Filter N/A Mode)
#
# Usage: bash batch_test_voc.sh
#
# This script tests all nothink models and outputs a summary table.
# Edit the FILES array below to add/remove test files.
#

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOC_SCRIPT="${SCRIPT_DIR}/calc_voc_filter_na.py"

# Output files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_OUTPUT="${SCRIPT_DIR}/voc_summary_${TIMESTAMP}.csv"
TXT_OUTPUT="${SCRIPT_DIR}/voc_summary_${TIMESTAMP}.txt"

# ============================================================================
# TEST FILES - Edit this section to add/remove files
# Format: "MODEL_NAME|CATEGORY|FILE_PATH"
# ============================================================================

declare -a FILES=(
    # ============ new_pro_bench (Qwen2.5-VL) ============

    # ------ 3B RL & SFT ------
    # multi_view
    "3B-RL|multi_view|/projects/p32958/chengxuan/results/new_pro_bench/visual_multi_view/rl_3B/visual_multi_view_rl_3B_20251217_021555/results.jsonl"
    "3B-SFT|multi_view|/projects/p32958/chengxuan/results/new_pro_bench/visual_multi_view/sft_3b_think/think_sft_3B_20251217_022124/results.jsonl"

    # normal_text
    "3B-RL|normal_text|/projects/p32958/chengxuan/results/new_pro_bench/text_normal/rl_3b/rl_3b_20251217_022013/results.jsonl"
    "3B-SFT|normal_text|/projects/p32958/chengxuan/results/new_pro_bench/text_normal/sft_3b/text_normal_sft_3b_20251217_021805/results.jsonl"

    # normal_view
    "3B-RL|normal_view|/projects/p32958/chengxuan/results/new_pro_bench/visual_single_view/one_view_3B_RL/visual_one_view_3B_RL_20251217_121528/results.jsonl"
    "3B-SFT|normal_view|/projects/p32958/chengxuan/results/new_pro_bench/visual_single_view/one_view_3B_SFT_think/visual_one_view_3B_SFT_20251217_025645/results.jsonl"

    # ------ Qwen2.5-VL nothink ------
    # multi_view
    "Qwen2.5-VL-3B|multi_view|/projects/p32958/chengxuan/results/new_pro_bench/visual_multi_view/nothink_3B/visual_multi_view_3B_nothink_20251218_020250/results.jsonl"
    "Qwen2.5-VL-7B|multi_view|/projects/p32958/chengxuan/results/new_pro_bench/visual_multi_view/nothink_7B/visual_multi_view_7B_nothink_20251218_022343/results.jsonl"
    "Qwen2.5-VL-32B|multi_view|/projects/p32958/chengxuan/results/new_pro_bench/visual_multi_view/nothink_32B/visual_multi_view_32B_nothink_20251218_024616/results.jsonl"
    "Qwen2.5-VL-72B|multi_view|/projects/p32958/chengxuan/results/new_pro_bench/visual_multi_view/nothink_72B/visual_multi_view_72B_nothink_20251218_035535/results.jsonl"

    # normal_text (text_normal)
    "Qwen2.5-VL-3B|normal_text|/projects/p32958/chengxuan/results/new_pro_bench/text_normal/nothink_3b/text_normal_nothink_3b_20251218_020745/results.jsonl"
    "Qwen2.5-VL-7B|normal_text|/projects/p32958/chengxuan/results/new_pro_bench/text_normal/nothink_7b/text_normal_nothink_7b_20251218_195031/results.jsonl"
    "Qwen2.5-VL-32B|normal_text|/projects/p32958/chengxuan/results/new_pro_bench/text_normal/nothink_32b/text_normal_nothink_32b_20251218_030902/results.jsonl"
    "Qwen2.5-VL-72B|normal_text|/projects/p32958/chengxuan/results/new_pro_bench/text_normal/nothink_72b/text_normal_nothink_72b_20251218_043523/results.jsonl"

    # normal_view (visual_single_view)
    "Qwen2.5-VL-3B|normal_view|/projects/p32958/chengxuan/results/new_pro_bench/visual_single_view/one_view_nothink_3B/visual_one_view_nothink_3B_20251218_015519/results.jsonl"
    "Qwen2.5-VL-7B|normal_view|/projects/p32958/chengxuan/results/new_pro_bench/visual_single_view/one_view_nothink_7B/visual_one_view_nothink_7B_20251218_021505/results.jsonl"
    "Qwen2.5-VL-32B|normal_view|/projects/p32958/chengxuan/results/new_pro_bench/visual_single_view/one_view_nothink_32B/visual_one_view_nothink_32B_20251218_023058/results.jsonl"
    "Qwen2.5-VL-72B|normal_view|/projects/p32958/chengxuan/results/new_pro_bench/visual_single_view/one_view_nothink_72B/visual_one_view_nothink_72B_20251218_023110/results.jsonl"

    # ============ OpenAI ============
    # multi_view
    "GPT-4o-mini|multi_view|/projects/p32958/chengxuan/results/openai/multi_view_nothink/gpt5_mini_nothink_20251217_220229/results.jsonl"
    "GPT-5|multi_view|/projects/p32958/chengxuan/results/openai/multi_view_nothink/gpt5_nothink_20251219_215639/results.jsonl"

    # normal_text
    "GPT-4o-mini|normal_text|/projects/p32958/chengxuan/results/openai/normal_text_nothink/gpt5_mini_nothink_20251217_215959/results.jsonl"
    "GPT-5|normal_text|/projects/p32958/chengxuan/results/openai/normal_text_nothink/gpt5_nothink_20251219_215746/results.jsonl"

    # normal_view
    "GPT-4o-mini|normal_view|/projects/p32958/chengxuan/results/openai/normal_view_nothink/gpt5_mini_nothink_20251217_215930/results.jsonl"
    "GPT-5|normal_view|/projects/p32958/chengxuan/results/openai/normal_view_nothink/gpt5_nothink_20251219_215355/results.jsonl"

    # ============ Qwen3-VL ============
    # multi_view
    "Qwen3-VL-2B|multi_view|/projects/p32958/chengxuan/results/qwen3vl/multi_view/qwen3vl_2b_nothink_20251218_002913/results.jsonl"
    "Qwen3-VL-4B|multi_view|/projects/p32958/chengxuan/results/qwen3vl/multi_view/qwen3vl_4b_nothink_20251218_002958/results.jsonl"
    "Qwen3-VL-8B|multi_view|/projects/p32958/chengxuan/results/qwen3vl/multi_view/qwen3vl_8b_nothink_20251217_233423/results.jsonl"
    "Qwen3-VL-32B|multi_view|/projects/p32958/chengxuan/results/qwen3vl/multi_view/qwen3vl_32b_nothink_20251217_232659/results.jsonl"

    # normal_text
    "Qwen3-VL-2B|normal_text|/projects/p32958/chengxuan/results/qwen3vl/normal_text/qwen3vl_2b_nothink_20251218_003212/results.jsonl"
    "Qwen3-VL-4B|normal_text|/projects/p32958/chengxuan/results/qwen3vl/normal_text/qwen3vl_4b_nothink_20251218_003248/results.jsonl"
    "Qwen3-VL-8B|normal_text|/projects/p32958/chengxuan/results/qwen3vl/normal_text/qwen3vl_8b_nothink_20251218_185437/results.jsonl"
    "Qwen3-VL-32B|normal_text|/projects/p32958/chengxuan/results/qwen3vl/normal_text/qwen3vl_32b_nothink_20251217_233444/results.jsonl"

    # normal_view
    "Qwen3-VL-2B|normal_view|/projects/p32958/chengxuan/results/qwen3vl/normal_view/qwen3vl_2b_nothink_20251218_002324/results.jsonl"
    "Qwen3-VL-4B|normal_view|/projects/p32958/chengxuan/results/qwen3vl/normal_view/qwen3vl_4b_nothink_20251218_002340/results.jsonl"
    "Qwen3-VL-8B|normal_view|/projects/p32958/chengxuan/results/qwen3vl/normal_view/qwen3vl_8b_nothink_20251217_232704/results.jsonl"
    "Qwen3-VL-32B|normal_view|/projects/p32958/chengxuan/results/qwen3vl/normal_view/qwen3vl_32b_nothink_20251217_232648/results.jsonl"

    # ============ InternVL ============
    # multi_view
    "InternVL-4B|multi_view|/projects/p32958/chengxuan/results/internvl/visual_multi_nothink/internvl_4B_multi_nothink_20251218_030235/results.jsonl"
    "InternVL-14B|multi_view|/projects/p32958/chengxuan/results/internvl/visual_multi_nothink/internvl_14B_multi_nothink_20251218_030757/results.jsonl"
    "InternVL-38B|multi_view|/projects/p32958/chengxuan/results/internvl/visual_multi_nothink/internvl_38B_multi_nothink_20251218_230758/results.jsonl"

    # normal_text
    "InternVL-4B|normal_text|/projects/p32958/chengxuan/results/internvl/text_normal_nothink/internvl_4B_text_normal_nothink_20251218_195139/results.jsonl"
    "InternVL-14B|normal_text|/projects/p32958/chengxuan/results/internvl/text_normal_nothink/internvl_14B_text_normal_nothink_20251218_030248/results.jsonl"
    "InternVL-38B|normal_text|/projects/p32958/chengxuan/results/internvl/text_normal_nothink/internvl_38B_text_normal_nothink_20251218_195750/results.jsonl"

    # normal_view
    "InternVL-4B|normal_view|/projects/p32958/chengxuan/results/internvl/visual_normal_nothink/internvl_4B_nothink_20251218_022635/results.jsonl"
    "InternVL-14B|normal_view|/projects/p32958/chengxuan/results/internvl/visual_normal_nothink/internvl_14B_nothink_20251218_205628/results.jsonl"
    "InternVL-38B|normal_view|/projects/p32958/chengxuan/results/internvl/visual_normal_nothink/internvl_38B_nothink_20251218_230326/results.jsonl"
)

# ============================================================================
# Main Script
# ============================================================================

echo "================================================================================"
echo "                    VOC Batch Test (Filter N/A Mode)"
echo "================================================================================"
echo "Start time: $(date)"
echo "Total files to test: ${#FILES[@]}"
echo ""

# Initialize CSV with header
echo "Model,Category,N/A Rate,Mean VOC,Valid Samples,Total Samples,Valid Trajectories" > "$CSV_OUTPUT"

# Arrays to store results
declare -a RESULTS=()

# Process each file
for entry in "${FILES[@]}"; do
    IFS='|' read -r MODEL CATEGORY FILEPATH <<< "$entry"

    echo "Testing: $MODEL ($CATEGORY)"

    if [[ ! -f "$FILEPATH" ]]; then
        echo "  WARNING: File not found: $FILEPATH"
        RESULTS+=("$MODEL|$CATEGORY|N/A|N/A|0|0|0")
        echo "$MODEL,$CATEGORY,N/A,N/A,0,0,0" >> "$CSV_OUTPUT"
        continue
    fi

    # Run VOC calculation and capture output
    OUTPUT=$(python3 "$VOC_SCRIPT" "$FILEPATH" 2>/dev/null)

    # Parse output
    TOTAL_SAMPLES=$(echo "$OUTPUT" | grep "Total samples:" | awk '{print $NF}')
    VALID_SAMPLES=$(echo "$OUTPUT" | grep "Valid samples:" | awk '{print $NF}')
    NA_SAMPLES=$(echo "$OUTPUT" | grep "Skipped N/A samples:" | awk '{print $NF}')
    VALID_TRAJ=$(echo "$OUTPUT" | grep "Valid trajectories:" | awk '{print $NF}')
    MEAN_VOC=$(echo "$OUTPUT" | grep "Mean VOC:" | awk '{print $NF}')

    # Calculate N/A rate
    if [[ -n "$TOTAL_SAMPLES" && "$TOTAL_SAMPLES" -gt 0 ]]; then
        NA_RATE=$(printf "%.2f" $(echo "scale=4; $NA_SAMPLES * 100 / $TOTAL_SAMPLES" | bc))
        NA_RATE="${NA_RATE}%"
    else
        NA_RATE="N/A"
    fi

    # Handle missing values
    [[ -z "$MEAN_VOC" ]] && MEAN_VOC="N/A"
    [[ -z "$VALID_SAMPLES" ]] && VALID_SAMPLES="0"
    [[ -z "$TOTAL_SAMPLES" ]] && TOTAL_SAMPLES="0"
    [[ -z "$VALID_TRAJ" ]] && VALID_TRAJ="0"

    echo "  N/A Rate: $NA_RATE, Mean VOC: $MEAN_VOC"

    RESULTS+=("$MODEL|$CATEGORY|$NA_RATE|$MEAN_VOC|$VALID_SAMPLES|$TOTAL_SAMPLES|$VALID_TRAJ")
    echo "$MODEL,$CATEGORY,$NA_RATE,$MEAN_VOC,$VALID_SAMPLES,$TOTAL_SAMPLES,$VALID_TRAJ" >> "$CSV_OUTPUT"
done

echo ""
echo "================================================================================"
echo "                              Summary Table"
echo "================================================================================"
printf "%-20s | %-12s | %10s | %10s\n" "Model" "Category" "N/A Rate" "Mean VOC"
echo "--------------------------------------------------------------------------------"

for result in "${RESULTS[@]}"; do
    IFS='|' read -r MODEL CATEGORY NA_RATE MEAN_VOC _ _ _ <<< "$result"
    printf "%-20s | %-12s | %10s | %10s\n" "$MODEL" "$CATEGORY" "$NA_RATE" "$MEAN_VOC"
done

echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  CSV: $CSV_OUTPUT"
echo "  TXT: $TXT_OUTPUT"
echo ""
echo "End time: $(date)"

# Also save summary to TXT
{
    echo "================================================================================"
    echo "                    VOC Batch Test Summary (Filter N/A Mode)"
    echo "================================================================================"
    echo "Generated: $(date)"
    echo ""
    printf "%-20s | %-12s | %10s | %10s\n" "Model" "Category" "N/A Rate" "Mean VOC"
    echo "--------------------------------------------------------------------------------"
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r MODEL CATEGORY NA_RATE MEAN_VOC _ _ _ <<< "$result"
        printf "%-20s | %-12s | %10s | %10s\n" "$MODEL" "$CATEGORY" "$NA_RATE" "$MEAN_VOC"
    done
    echo "================================================================================"
} > "$TXT_OUTPUT"
