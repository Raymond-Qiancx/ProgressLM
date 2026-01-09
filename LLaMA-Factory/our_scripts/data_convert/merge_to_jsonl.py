#!/usr/bin/env python3
"""
Merge multiple JSON files into a single JSONL file.

This script reads JSON files containing arrays of samples and combines them
into a single JSONL file where each line is a JSON object.
"""
import argparse
import json
import os
import sys
from typing import List, Dict, Any


def load_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON file containing a list of samples.

    Args:
        file_path: Path to JSON file

    Returns:
        List of samples
    """
    if not os.path.exists(file_path):
        print(f"⚠️  Warning: File not found: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list in {file_path}, got {type(data)}")

    return data


def save_jsonl(data: List[Dict[str, Any]], output_path: str):
    """
    Save data to JSONL file (one JSON object per line).

    Args:
        data: List of samples to save
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def merge_json_to_jsonl(
    input_files: List[str],
    output_file: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Merge multiple JSON files into a single JSONL file.

    Args:
        input_files: List of input JSON file paths
        output_file: Output JSONL file path
        verbose: Print detailed logs

    Returns:
        Statistics dictionary
    """
    all_samples = []
    stats = {
        'input_files': len(input_files),
        'total_samples': 0,
        'per_file_samples': {}
    }

    print("=" * 60)
    print("JSON to JSONL Conversion")
    print("=" * 60)

    for file_path in input_files:
        file_name = os.path.basename(file_path)
        print(f"\nLoading: {file_name}")

        try:
            samples = load_json(file_path)
            print(f"  Loaded {len(samples)} samples")

            all_samples.extend(samples)
            stats['total_samples'] += len(samples)
            stats['per_file_samples'][file_name] = len(samples)

        except Exception as e:
            print(f"  ❌ Error loading file: {e}")
            stats['per_file_samples'][file_name] = 0
            continue

    # Save as JSONL
    print(f"\n{'=' * 60}")
    print(f"Saving {len(all_samples)} samples to JSONL: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_jsonl(all_samples, output_file)

    # Print summary
    print(f"\n{'=' * 60}")
    print("Conversion Summary")
    print(f"{'=' * 60}")
    print(f"Input files:     {stats['input_files']}")
    print(f"Total samples:   {stats['total_samples']}")
    print(f"Output file:     {output_file}")
    print(f"{'=' * 60}")

    if verbose and stats['per_file_samples']:
        print("\nPer-file Sample Counts:")
        for file_name, count in stats['per_file_samples'].items():
            print(f"  {file_name}: {count} samples")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple JSON files into a single JSONL file'
    )
    parser.add_argument(
        '--input-files',
        nargs='+',
        required=True,
        help='List of input JSON files to merge'
    )
    parser.add_argument(
        '--output-file',
        required=True,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed logs'
    )

    args = parser.parse_args()

    # Validate that we have at least one input file
    if not args.input_files:
        print("❌ Error: No input files specified")
        sys.exit(1)

    # Run merge
    try:
        stats = merge_json_to_jsonl(
            input_files=args.input_files,
            output_file=args.output_file,
            verbose=args.verbose
        )

        if stats['total_samples'] == 0:
            print("\n❌ Error: No samples were merged")
            sys.exit(1)

        print("\n✅ JSON to JSONL conversion completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
