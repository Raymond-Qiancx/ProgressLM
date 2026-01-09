#!/usr/bin/env python3
"""
Merge multiple LLaMA-Factory ShareGPT format JSON files into a single file.

This script combines multiple converted dataset JSON files, maintaining
the ShareGPT message format and image paths.
"""
import argparse
import json
import os
import sys
from typing import List, Dict, Any
from utils import validate_image_tag_count, validate_xml_tags, count_image_tags


def load_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON file containing a list of samples.

    Args:
        file_path: Path to JSON file

    Returns:
        List of samples
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list in {file_path}, got {type(data)}")

    return data


def save_json(data: List[Dict[str, Any]], output_path: str):
    """
    Save data to JSON file with proper formatting.

    Args:
        data: List of samples to save
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def validate_sample(sample: Dict[str, Any], idx: int) -> bool:
    """
    Validate a single sample.

    Args:
        sample: Sample dictionary
        idx: Sample index for error reporting

    Returns:
        True if valid, False otherwise
    """
    # Check required fields
    if 'messages' not in sample:
        print(f"  Warning: Sample {idx} missing 'messages' field")
        return False

    if 'images' not in sample:
        print(f"  Warning: Sample {idx} missing 'images' field")
        return False

    messages = sample['messages']
    images = sample['images']

    # Validate messages structure
    if not isinstance(messages, list) or len(messages) != 2:
        print(f"  Warning: Sample {idx} should have exactly 2 messages (user and assistant)")
        return False

    # Check message roles
    expected_roles = ['user', 'assistant']
    for i, (msg, expected_role) in enumerate(zip(messages, expected_roles)):
        if msg.get('role') != expected_role:
            print(f"  Warning: Sample {idx} message {i} has incorrect role: "
                  f"{msg.get('role')} (expected {expected_role})")
            return False

        if 'content' not in msg:
            print(f"  Warning: Sample {idx} message {i} missing 'content'")
            return False

    # Validate image tags in user message
    user_content = messages[0]['content']
    image_tag_count = count_image_tags(user_content)

    if image_tag_count != len(images):
        print(f"  Warning: Sample {idx} image tag count mismatch: "
              f"tags={image_tag_count}, images={len(images)}")
        return False

    # Validate assistant response XML tags
    assistant_content = messages[1]['content']
    if not validate_xml_tags(assistant_content):
        print(f"  Warning: Sample {idx} has invalid XML tags in assistant response")
        return False

    return True


def merge_datasets(
    input_files: List[str],
    output_file: str,
    validate: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Merge multiple LLaMA-Factory format JSON files.

    Args:
        input_files: List of input JSON file paths
        output_file: Output merged JSON file path
        validate: Whether to validate samples
        verbose: Print detailed logs

    Returns:
        Statistics dictionary
    """
    merged_samples = []
    stats = {
        'input_files': len(input_files),
        'total_samples': 0,
        'valid_samples': 0,
        'invalid_samples': 0,
        'per_file_stats': {}
    }

    print("=" * 60)
    print("Dataset Merging")
    print("=" * 60)

    for file_path in input_files:
        file_name = os.path.basename(file_path)
        print(f"\nLoading: {file_name}")

        if not os.path.exists(file_path):
            print(f"  ⚠️  Warning: File not found, skipping: {file_path}")
            stats['per_file_stats'][file_name] = {
                'loaded': 0,
                'valid': 0,
                'invalid': 0,
                'status': 'not_found'
            }
            continue

        try:
            samples = load_json(file_path)
            file_stats = {
                'loaded': len(samples),
                'valid': 0,
                'invalid': 0,
                'status': 'success'
            }

            print(f"  Loaded {len(samples)} samples")

            # Validate and add samples
            for idx, sample in enumerate(samples):
                stats['total_samples'] += 1

                if validate:
                    if validate_sample(sample, idx):
                        merged_samples.append(sample)
                        stats['valid_samples'] += 1
                        file_stats['valid'] += 1
                    else:
                        stats['invalid_samples'] += 1
                        file_stats['invalid'] += 1
                else:
                    merged_samples.append(sample)
                    stats['valid_samples'] += 1
                    file_stats['valid'] += 1

            stats['per_file_stats'][file_name] = file_stats
            print(f"  Added {file_stats['valid']} valid samples")

            if file_stats['invalid'] > 0:
                print(f"  Skipped {file_stats['invalid']} invalid samples")

        except Exception as e:
            print(f"  ❌ Error loading file: {e}")
            stats['per_file_stats'][file_name] = {
                'loaded': 0,
                'valid': 0,
                'invalid': 0,
                'status': f'error: {str(e)}'
            }
            continue

    # Save merged output
    print(f"\n{'=' * 60}")
    print(f"Saving {len(merged_samples)} merged samples to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_json(merged_samples, output_file)

    # Print summary
    print(f"\n{'=' * 60}")
    print("Merge Summary")
    print(f"{'=' * 60}")
    print(f"Input files:     {stats['input_files']}")
    print(f"Total samples:   {stats['total_samples']}")
    print(f"Valid samples:   {stats['valid_samples']}")
    print(f"Invalid samples: {stats['invalid_samples']}")
    print(f"Output file:     {output_file}")
    print(f"{'=' * 60}")

    if verbose and stats['per_file_stats']:
        print("\nPer-file Statistics:")
        for file_name, file_stat in stats['per_file_stats'].items():
            print(f"  {file_name}:")
            print(f"    Loaded: {file_stat['loaded']}")
            print(f"    Valid:  {file_stat['valid']}")
            if file_stat['invalid'] > 0:
                print(f"    Invalid: {file_stat['invalid']}")
            print(f"    Status: {file_stat['status']}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple LLaMA-Factory format JSON files'
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
        help='Output merged JSON file path'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation of samples'
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
        stats = merge_datasets(
            input_files=args.input_files,
            output_file=args.output_file,
            validate=not args.no_validate,
            verbose=args.verbose
        )

        if stats['valid_samples'] == 0:
            print("\n❌ Error: No valid samples were merged")
            sys.exit(1)

        print("\n✅ Dataset merging completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during merging: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
