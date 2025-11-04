#!/usr/bin/env python3
"""
Convert negation-processed JSONL files back to original format.

This script extracts the meta_data from negation-processed JSONL files
and converts them back to the original training data format.

Usage:
    python convert_negation_to_original.py <input_file> [output_file]

Example:
    python convert_negation_to_original.py input.jsonl output.jsonl
    python convert_negation_to_original.py input.jsonl  # Auto-generates output name
"""

import json
import sys
import os
from pathlib import Path


def convert_entry(entry):
    """
    Convert a single negation-processed entry back to original format.
    Uses edited_goal and edited_demo to replace the original task_goal and text_demo.

    Args:
        entry: Dictionary containing the negation-processed data

    Returns:
        Dictionary in original format with edited content
    """
    # Extract meta_data
    if 'meta_data' not in entry:
        raise ValueError("Entry does not contain 'meta_data' field")

    meta_data = entry['meta_data']

    # Use edited_goal and edited_demo instead of original
    # Fallback to original if edited versions don't exist
    task_goal = entry.get('edited_goal', meta_data['task_goal'])
    text_demo = entry.get('edited_demo', meta_data['text_demo'])

    # Handle stage_to_estimate: extract filename only
    # Use stage_to_estimate_original if available, otherwise extract from path
    if 'stage_to_estimate_original' in meta_data:
        stage_to_estimate = meta_data['stage_to_estimate_original']
    else:
        # Extract filename from full path
        stage_path = meta_data['stage_to_estimate']
        stage_to_estimate = os.path.basename(stage_path)

    # Build the converted entry with required fields in the exact order
    converted = {
        'id': meta_data['id'],
        'task_goal': task_goal,  # Use edited_goal
        'text_demo': text_demo,  # Use edited_demo
        'total_steps': str(meta_data['total_steps']),  # Convert int to string
        'stage_to_estimate': stage_to_estimate,  # Filename only
        'closest_idx': "n/a",  # Set to n/a
        'progress_score': "n/a",  # Set to n/a
        'data_source': meta_data['data_source']
    }

    return converted


def convert_jsonl(input_path, output_path):
    """
    Convert an entire JSONL file from negation format to original format.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    converted_count = 0
    error_count = 0

    print(f"Converting: {input_path}")
    print(f"Output to: {output_path}")

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Parse JSON
                entry = json.loads(line)

                # Convert to original format
                converted = convert_entry(entry)

                # Write to output file
                outfile.write(json.dumps(converted, ensure_ascii=False) + '\n')
                converted_count += 1

            except Exception as e:
                print(f"Error on line {line_num}: {e}", file=sys.stderr)
                error_count += 1

    print(f"\nConversion complete!")
    print(f"  Successfully converted: {converted_count} entries")
    if error_count > 0:
        print(f"  Errors encountered: {error_count} entries")

    return converted_count, error_count


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_negation_to_original.py <input_file> [output_file]")
        print("\nExample:")
        print("  python convert_negation_to_original.py input.jsonl output.jsonl")
        print("  python convert_negation_to_original.py input.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]

    # Generate output filename if not provided
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Auto-generate output filename
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_converted{input_path.suffix}"

    try:
        convert_jsonl(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
