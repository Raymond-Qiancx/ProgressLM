#!/usr/bin/env python3
"""
Validate LLaMA-Factory format output files.

This script validates that converted SFT data files are correctly formatted
with proper image tag counts, required fields, and valid structure.
"""
import argparse
import json
import sys
from typing import Dict, Any, List
from utils import (
    count_image_tags,
    validate_xml_tags
)


def validate_sample(sample: Dict[str, Any], idx: int, verbose: bool = False) -> Dict[str, Any]:
    """
    Validate a single sample.

    Args:
        sample: Sample dict to validate
        idx: Sample index
        verbose: Print detailed validation info

    Returns:
        Validation result dict with 'valid' boolean and 'errors' list
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # Check required fields
    if 'messages' not in sample:
        result['valid'] = False
        result['errors'].append("Missing 'messages' field")
        return result

    if 'images' not in sample:
        result['valid'] = False
        result['errors'].append("Missing 'images' field")
        return result

    # Check messages structure
    messages = sample['messages']
    if not isinstance(messages, list):
        result['valid'] = False
        result['errors'].append("'messages' must be a list")
        return result

    # Support both 2-message (user, assistant) and 3-message (system, user, assistant) formats
    if len(messages) not in [2, 3]:
        result['valid'] = False
        result['errors'].append(f"'messages' should have 2 or 3 items, got {len(messages)}")
        return result

    # Determine message indices based on count
    if len(messages) == 3:
        # Format: [system, user, assistant]
        system_msg = messages[0]
        user_msg = messages[1]
        assistant_msg = messages[2]

        # Check system message
        if system_msg.get('role') != 'system':
            result['valid'] = False
            result['errors'].append(f"First message role should be 'system', got '{system_msg.get('role')}'")

        if 'content' not in system_msg:
            result['valid'] = False
            result['errors'].append("System message missing 'content' field")
    else:
        # Format: [user, assistant]
        user_msg = messages[0]
        assistant_msg = messages[1]

    # Check user message
    if user_msg.get('role') != 'user':
        result['valid'] = False
        expected_idx = "second" if len(messages) == 3 else "first"
        result['errors'].append(f"{expected_idx.capitalize()} message role should be 'user', got '{user_msg.get('role')}'")

    if 'content' not in user_msg:
        result['valid'] = False
        result['errors'].append("User message missing 'content' field")
        return result

    # Check assistant message
    if assistant_msg.get('role') != 'assistant':
        result['valid'] = False
        expected_idx = "third" if len(messages) == 3 else "second"
        result['errors'].append(f"{expected_idx.capitalize()} message role should be 'assistant', got '{assistant_msg.get('role')}'")

    if 'content' not in assistant_msg:
        result['valid'] = False
        result['errors'].append("Assistant message missing 'content' field")
        return result

    # Check images array
    images = sample['images']
    if not isinstance(images, list):
        result['valid'] = False
        result['errors'].append("'images' must be a list")
        return result

    if len(images) == 0:
        result['valid'] = False
        result['errors'].append("'images' array is empty")

    # Validate image tag count
    user_content = user_msg['content']
    image_tag_count = count_image_tags(user_content)

    if image_tag_count != len(images):
        result['valid'] = False
        result['errors'].append(
            f"Image tag count mismatch: {image_tag_count} <image> tags, {len(images)} images"
        )

    # Validate assistant XML tags
    assistant_content = assistant_msg['content']
    if not validate_xml_tags(assistant_content):
        result['warnings'].append("Assistant response missing some required XML tags")

    # Check for empty content
    if not user_content.strip():
        result['valid'] = False
        result['errors'].append("User content is empty")

    if not assistant_content.strip():
        result['valid'] = False
        result['errors'].append("Assistant content is empty")

    return result


def validate_file(file_path: str, verbose: bool = False, show_samples: int = 3) -> Dict[str, Any]:
    """
    Validate entire output file.

    Args:
        file_path: Path to JSON output file
        verbose: Print detailed validation info
        show_samples: Number of sample validations to show in detail

    Returns:
        Overall validation statistics
    """
    print(f"Validating file: {file_path}")
    print("=" * 60)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Failed to parse JSON file: {e}")
        return {'valid': False, 'error': 'JSON parse error'}
    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        return {'valid': False, 'error': 'File not found'}

    if not isinstance(data, list):
        print(f"❌ Error: Root element should be a list, got {type(data)}")
        return {'valid': False, 'error': 'Root not a list'}

    print(f"Total samples: {len(data)}\n")

    stats = {
        'total': len(data),
        'valid': 0,
        'invalid': 0,
        'errors': [],
        'warnings': []
    }

    # Validate each sample
    for idx, sample in enumerate(data):
        result = validate_sample(sample, idx, verbose)

        if result['valid']:
            stats['valid'] += 1
        else:
            stats['invalid'] += 1
            stats['errors'].extend(result['errors'])

        stats['warnings'].extend(result['warnings'])

        # Show detailed info for first N samples
        if verbose or idx < show_samples:
            status = "✅" if result['valid'] else "❌"
            print(f"{status} Sample {idx + 1}:")

            if result['valid']:
                # Show brief info
                images_count = len(sample.get('images', []))
                user_content = sample['messages'][0]['content']
                image_tags = count_image_tags(user_content)
                print(f"  Images: {images_count}, <image> tags: {image_tags}")
            else:
                # Show errors
                for error in result['errors']:
                    print(f"  ❌ {error}")

            if result['warnings']:
                for warning in result['warnings']:
                    print(f"  ⚠️  {warning}")

            print()

    # Print summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"Total samples: {stats['total']}")
    print(f"Valid samples: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
    print(f"Invalid samples: {stats['invalid']} ({stats['invalid']/stats['total']*100:.1f}%)")

    if stats['invalid'] > 0:
        print(f"\n❌ Validation FAILED: {stats['invalid']} invalid samples found")
        print("\nError summary (showing unique errors):")
        unique_errors = list(set(stats['errors']))
        for i, error in enumerate(unique_errors[:10], 1):
            print(f"  {i}. {error}")
        if len(unique_errors) > 10:
            print(f"  ... and {len(unique_errors) - 10} more unique errors")
    else:
        print(f"\n✅ Validation PASSED: All samples are valid!")

    if stats['warnings']:
        print(f"\n⚠️  {len(stats['warnings'])} warnings found")
        unique_warnings = list(set(stats['warnings']))
        for i, warning in enumerate(unique_warnings[:5], 1):
            print(f"  {i}. {warning}")

    print("=" * 60)

    return stats


def validate_dataset_structure(file_path: str) -> None:
    """
    Validate and display dataset structure statistics.

    Args:
        file_path: Path to JSON output file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if not data:
        print("No data in file")
        return

    print("\n" + "=" * 60)
    print("Dataset Structure Analysis")
    print("=" * 60)

    # Analyze image counts
    image_counts = [len(sample.get('images', [])) for sample in data]
    min_images = min(image_counts) if image_counts else 0
    max_images = max(image_counts) if image_counts else 0
    avg_images = sum(image_counts) / len(image_counts) if image_counts else 0

    print(f"Image counts per sample:")
    print(f"  Min: {min_images}")
    print(f"  Max: {max_images}")
    print(f"  Average: {avg_images:.1f}")

    # Distribution
    from collections import Counter
    count_dist = Counter(image_counts)
    print(f"\nImage count distribution:")
    for count in sorted(count_dist.keys())[:5]:
        print(f"  {count} images: {count_dist[count]} samples")

    # Analyze message lengths
    user_lengths = []
    assistant_lengths = []

    for sample in data[:100]:  # Sample first 100
        if 'messages' in sample and len(sample['messages']) >= 2:
            user_lengths.append(len(sample['messages'][0].get('content', '')))
            assistant_lengths.append(len(sample['messages'][1].get('content', '')))

    if user_lengths:
        print(f"\nMessage content lengths (chars, sampled):")
        print(f"  User avg: {sum(user_lengths)/len(user_lengths):.0f}")
        print(f"  Assistant avg: {sum(assistant_lengths)/len(assistant_lengths):.0f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Validate LLaMA-Factory format output files'
    )
    parser.add_argument(
        '--input-file',
        required=True,
        help='Path to JSON output file to validate'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed validation info for all samples'
    )
    parser.add_argument(
        '--show-samples',
        type=int,
        default=3,
        help='Number of sample validations to show in detail (default: 3)'
    )
    parser.add_argument(
        '--structure',
        action='store_true',
        help='Also show dataset structure analysis'
    )

    args = parser.parse_args()

    # Run validation
    stats = validate_file(
        file_path=args.input_file,
        verbose=args.verbose,
        show_samples=args.show_samples
    )

    # Show structure analysis if requested
    if args.structure and stats.get('total', 0) > 0:
        validate_dataset_structure(args.input_file)

    # Exit with appropriate code
    if stats.get('invalid', 0) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
