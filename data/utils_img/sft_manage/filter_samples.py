#!/usr/bin/env python3
"""
JSONL Sample Filtering Script
Removes samples from visual_franka_3rgb_new.jsonl that appear in rl_sampled_35k.jsonl
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Configuration
INPUT_JSONL = "/home/vcj9002/jianshu/chengxuan/ProgressLM/data/train/rl/text_rl_all_3to7.jsonl"
FILTER_JSONL = "/home/vcj9002/jianshu/chengxuan/ProgressLM/data/train/rl/rl_sampled_35k.jsonl"
OUTPUT_JSONL = "/home/vcj9002/jianshu/chengxuan/ProgressLM/data/train/rl/text_rl_all_3to7_clean.jsonl"

SCRIPT_DIR = "/home/vcj9002/jianshu/chengxuan/ProgressLM/data/utils_img/sft_manage"
LOG_FILE = Path(SCRIPT_DIR) / f"filter_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_jsonl(file_path):
    """
    Load JSONL file and return list of samples
    Args:
        file_path: Path to JSONL file
    Returns:
        List of dictionaries (parsed JSON objects)
    """
    logger = logging.getLogger(__name__)
    samples = []

    logger.info(f"Loading file: {file_path}")
    print(f"\nLoading: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Reading lines", unit=" lines"), 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                        print(f"Warning: Failed to parse line {line_num}")

        logger.info(f"Loaded {len(samples)} samples from {file_path}")
        print(f"Loaded {len(samples):,} samples")
        return samples

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        print(f"ERROR: File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        print(f"ERROR: {e}")
        return []


def create_sample_set(samples):
    """
    Create a set of sample identifiers for fast lookup
    Uses JSON string representation for comparison
    Args:
        samples: List of sample dictionaries
    Returns:
        Set of JSON string representations
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating sample lookup set...")
    print("\nCreating sample lookup set...")

    sample_set = set()
    for sample in tqdm(samples, desc="Building set", unit=" samples"):
        # Convert to JSON string for consistent comparison
        # sort_keys ensures consistent ordering
        sample_str = json.dumps(sample, sort_keys=True, ensure_ascii=False)
        sample_set.add(sample_str)

    logger.info(f"Created set with {len(sample_set)} unique samples")
    print(f"Set contains {len(sample_set):,} unique samples")
    return sample_set


def filter_samples_from_set(input_samples, filter_set):
    """
    Filter out samples that appear in the filter set
    Args:
        input_samples: List of input samples
        filter_set: Set of sample identifiers to filter out
    Returns:
        List of filtered samples, count of removed samples
    """
    logger = logging.getLogger(__name__)
    logger.info("Filtering samples...")
    print("\nFiltering samples...")

    filtered_samples = []
    removed_count = 0

    for sample in tqdm(input_samples, desc="Filtering", unit=" samples"):
        sample_str = json.dumps(sample, sort_keys=True, ensure_ascii=False)

        if sample_str not in filter_set:
            filtered_samples.append(sample)
        else:
            removed_count += 1

    logger.info(f"Removed {removed_count} samples")
    logger.info(f"Kept {len(filtered_samples)} samples")
    print(f"\nRemoved: {removed_count:,} samples")
    print(f"Kept: {len(filtered_samples):,} samples")

    return filtered_samples, removed_count


def save_jsonl(samples, output_path):
    """
    Save samples to JSONL file
    Args:
        samples: List of sample dictionaries
        output_path: Path to output file
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Saving to: {output_path}")
    print(f"\nSaving to: {output_path}")

    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in tqdm(samples, desc="Writing", unit=" samples"):
                json_line = json.dumps(sample, ensure_ascii=False)
                f.write(json_line + '\n')

        logger.info(f"Successfully saved {len(samples)} samples")
        print(f"Successfully saved {len(samples):,} samples")

    except Exception as e:
        logger.error(f"Error saving file: {e}")
        print(f"ERROR: Failed to save file: {e}")


def print_statistics(input_count, filter_count, removed_count, output_count):
    """Print final statistics"""
    print("\n" + "="*60)
    print("FILTERING STATISTICS")
    print("="*60)
    print(f"Input samples:         {input_count:,}")
    print(f"Filter samples:        {filter_count:,}")
    print(f"Removed samples:       {removed_count:,}")
    print(f"Output samples:        {output_count:,}")
    print(f"Removal rate:          {removed_count/input_count*100:.2f}%" if input_count > 0 else "N/A")
    print("="*60)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Filter JSONL samples based on another JSONL file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python filter_samples.py

  # Specify custom paths
  python filter_samples.py --input input.jsonl --filter filter.jsonl --output output.jsonl
        """
    )

    parser.add_argument('--input', type=str, default=INPUT_JSONL,
                        help=f'Input JSONL file (default: {INPUT_JSONL})')
    parser.add_argument('--filter', type=str, default=FILTER_JSONL,
                        help=f'Filter JSONL file (default: {FILTER_JSONL})')
    parser.add_argument('--output', type=str, default=OUTPUT_JSONL,
                        help=f'Output JSONL file (default: {OUTPUT_JSONL})')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    # Print initial info
    print("="*60)
    print("JSONL Sample Filtering Script")
    print("="*60)
    print(f"Input file:  {args.input}")
    print(f"Filter file: {args.filter}")
    print(f"Output file: {args.output}")
    print(f"Log file:    {LOG_FILE}")
    print("="*60)

    logger.info("="*60)
    logger.info("JSONL Sample Filtering Script Started")
    logger.info("="*60)
    logger.info(f"Input file:  {args.input}")
    logger.info(f"Filter file: {args.filter}")
    logger.info(f"Output file: {args.output}")

    # Step 1: Load input samples
    input_samples = load_jsonl(args.input)
    if not input_samples:
        logger.error("No input samples loaded. Exiting.")
        print("\nERROR: No input samples loaded. Exiting.")
        return

    # Step 2: Load filter samples
    filter_samples = load_jsonl(args.filter)
    if not filter_samples:
        logger.error("No filter samples loaded. Exiting.")
        print("\nERROR: No filter samples loaded. Exiting.")
        return

    # Step 3: Create filter set for fast lookup
    filter_set = create_sample_set(filter_samples)

    # Step 4: Filter samples
    filtered_samples, removed_count = filter_samples_from_set(input_samples, filter_set)

    # Step 5: Save output
    if filtered_samples:
        save_jsonl(filtered_samples, args.output)
    else:
        logger.warning("No samples to save after filtering!")
        print("\nWARNING: No samples to save after filtering!")

    # Print statistics
    print_statistics(
        len(input_samples),
        len(filter_samples),
        removed_count,
        len(filtered_samples)
    )

    print("\n" + "="*60)
    print("Script completed successfully")
    print("="*60)
    print(f"Check log file for details: {LOG_FILE}")

    logger.info("="*60)
    logger.info("Script completed successfully")
    logger.info("="*60)


if __name__ == "__main__":
    main()
