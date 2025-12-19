#!/bin/bash
#
# Batch VOC Evaluation Script
#
# Usage:
#   ./batch_eval_voc.sh                          # Evaluate all results in default directory
#   ./batch_eval_voc.sh /path/to/results         # Evaluate all results in specified directory
#   ./batch_eval_voc.sh /path/to/file.jsonl      # Evaluate a single file
#
# Output:
#   - Prints VOC results for each file
#   - Saves individual results as JSON files next to the original JSONL
#   - Saves summary CSV to the script directory
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALC_SCRIPT="${SCRIPT_DIR}/calc_voc_auto.py"

# Default results directory
DEFAULT_RESULTS_DIR="/gpfs/projects/p32958/chengxuan/results/new_pro_bench"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Summary file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${SCRIPT_DIR}/voc_summary_${TIMESTAMP}.csv"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    VOC Batch Evaluation Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if calc script exists
if [ ! -f "$CALC_SCRIPT" ]; then
    echo -e "${RED}Error: Calculator script not found: $CALC_SCRIPT${NC}"
    exit 1
fi

# Determine input
INPUT="${1:-$DEFAULT_RESULTS_DIR}"

# Initialize summary CSV
echo "file,format,total_samples,total_trajectories,valid_trajectories,voc_mean,voc_std,voc_min,voc_max,voc_median" > "$SUMMARY_FILE"

# Counter
TOTAL_FILES=0
SUCCESS_FILES=0
FAILED_FILES=0

process_file() {
    local jsonl_file="$1"
    local output_file="${jsonl_file%.jsonl}_voc.json"

    echo -e "${YELLOW}Processing: ${jsonl_file}${NC}"

    # Run VOC calculation
    result=$(python3 "$CALC_SCRIPT" "$jsonl_file" --output "$output_file" 2>&1)
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}  ✓ Success${NC}"

        # Extract VOC mean from output
        voc_mean=$(echo "$result" | grep "Mean VOC:" | awk '{print $3}')
        if [ -n "$voc_mean" ]; then
            echo -e "  Mean VOC: ${GREEN}${voc_mean}${NC}"
        fi

        # Parse JSON output for summary
        if [ -f "$output_file" ]; then
            format=$(python3 -c "import json; print(json.load(open('$output_file'))['format'])" 2>/dev/null)
            total_samples=$(python3 -c "import json; print(json.load(open('$output_file'))['total_samples'])" 2>/dev/null)
            total_traj=$(python3 -c "import json; print(json.load(open('$output_file'))['total_trajectories'])" 2>/dev/null)
            valid_traj=$(python3 -c "import json; print(json.load(open('$output_file'))['valid_trajectories'])" 2>/dev/null)
            voc_mean=$(python3 -c "import json; v=json.load(open('$output_file'))['voc_mean']; print(v if v else 'N/A')" 2>/dev/null)
            voc_std=$(python3 -c "import json; v=json.load(open('$output_file'))['voc_std']; print(v if v else 'N/A')" 2>/dev/null)
            voc_min=$(python3 -c "import json; v=json.load(open('$output_file'))['voc_min']; print(v if v else 'N/A')" 2>/dev/null)
            voc_max=$(python3 -c "import json; v=json.load(open('$output_file'))['voc_max']; print(v if v else 'N/A')" 2>/dev/null)
            voc_median=$(python3 -c "import json; v=json.load(open('$output_file'))['voc_median']; print(v if v else 'N/A')" 2>/dev/null)

            echo "\"$jsonl_file\",\"$format\",$total_samples,$total_traj,$valid_traj,$voc_mean,$voc_std,$voc_min,$voc_max,$voc_median" >> "$SUMMARY_FILE"
        fi

        ((SUCCESS_FILES++))
    else
        echo -e "${RED}  ✗ Failed${NC}"
        echo "$result" | head -5
        ((FAILED_FILES++))
    fi

    ((TOTAL_FILES++))
    echo ""
}

# Process input
if [ -f "$INPUT" ]; then
    # Single file
    if [[ "$INPUT" == *.jsonl ]]; then
        process_file "$INPUT"
    else
        echo -e "${RED}Error: Not a JSONL file: $INPUT${NC}"
        exit 1
    fi
elif [ -d "$INPUT" ]; then
    # Directory - find all results.jsonl files
    echo -e "Searching for JSONL files in: ${BLUE}$INPUT${NC}"
    echo ""

    # Find all results.jsonl or *results*.jsonl files
    while IFS= read -r -d '' jsonl_file; do
        process_file "$jsonl_file"
    done < <(find "$INPUT" -name "results*.jsonl" -type f -print0 | sort -z)

    # Also check for files ending with _results.jsonl
    while IFS= read -r -d '' jsonl_file; do
        process_file "$jsonl_file"
    done < <(find "$INPUT" -name "*_results.jsonl" -type f -print0 | sort -z)

else
    echo -e "${RED}Error: Path not found: $INPUT${NC}"
    exit 1
fi

# Print summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total files processed: ${TOTAL_FILES}"
echo -e "Successful: ${GREEN}${SUCCESS_FILES}${NC}"
echo -e "Failed: ${RED}${FAILED_FILES}${NC}"
echo ""
echo -e "Summary saved to: ${BLUE}${SUMMARY_FILE}${NC}"
echo ""

# Print summary table
if [ -f "$SUMMARY_FILE" ] && [ $(wc -l < "$SUMMARY_FILE") -gt 1 ]; then
    echo -e "${YELLOW}VOC Results Summary:${NC}"
    echo ""
    column -t -s',' "$SUMMARY_FILE" 2>/dev/null || cat "$SUMMARY_FILE"
fi
