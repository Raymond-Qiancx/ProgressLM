#!/usr/bin/env python3
"""
Convert Visual Demo data to LLaMA-Factory ShareGPT format.

This script merges original visual demo data with CoT responses to create
SFT training data in LLaMA-Factory's image-text interleaved format.
"""
import argparse
import os
import sys
from typing import Dict, Any, List
from utils import (
    load_jsonl,
    save_json,
    build_match_key,
    build_image_path,
    format_visual_demo_progress_shifts,
    normalize_stage_to_estimate,
    normalize_total_steps,
    validate_image_tag_count,
    validate_xml_tags,
    count_image_tags,
    print_conversion_stats
)


# System prompt for inference mode
VISUAL_DEMO_SYSTEM_PROMPT = """You are a progress estimator that evaluates the progress of the current state during an ongoing task based on a visual demonstration. The demonstration consists of a sequence of vision-based states and their corresponding progress value (ranging from 0% to 100%), showing how the task evolves from start to completion."""


# Task instruction text (from original prompt, excluding ground_truth section)
VISUAL_DEMO_INSTRUCTION = """Your task:
1. Check the current state image carefully.
2. Analyze the overall task goal and visual demonstration to understand how the task progresses from start to completion.
3. Identify the reference states from the visual demonstration that are most related to the current state image.
4. Compare the current state image with the chosen reference state, determining whether the image is behind or after the reference state.
5. Estimate the progress numerically as a floating-point value between 0% and 100%.
6. If you really cannot match the current state image to any of the states from demonstration, you need to explain the reason within `<ref_think></ref_think>` and output "n/a" within `<ref></ref>`, `<score_think></score_think>`, and `<score></score>`.

Your response **must** strictly follow this format:
<ref_think>Reason for choosing the most related state from the demonstration as the reference or explanation of why the current state image does not match the task goal or any steps from demonstration</ref_think>
<ref>which state from the visual demonstration is most related to the current state (output only the number of the state) or "n/a"</ref>
<score_think>Reason for comparing the current state image with the reference state or "n/a"</score_think>
<score>Your final estimated progress score or "n/a"</score>"""


def normalize_stage_filename(stage_path: str) -> str:
    """
    Normalize stage_to_estimate filename for matching.

    Extracts the filename and removes '_edit' suffix before extension.
    Example: 'camera_top_0536_edit.jpg' -> 'camera_top_0536.jpg'

    Args:
        stage_path: Full path or filename of stage_to_estimate

    Returns:
        Normalized filename without _edit suffix
    """
    if isinstance(stage_path, list):
        stage_path = stage_path[0] if stage_path else ''

    # Extract filename from path
    filename = os.path.basename(stage_path) if stage_path else ''

    # Remove _edit suffix before extension
    if '_edit.' in filename:
        filename = filename.replace('_edit.', '.')

    return filename


def build_user_message(item: Dict[str, Any]) -> str:
    """
    Build user message content for Visual Demo task.

    Args:
        item: Original data item with task_goal, visual_demo, total_steps, etc.

    Returns:
        Formatted user message string with N+1 <image> tags
    """
    parts = []

    # 0. System prompt at the beginning of user message
    parts.append(VISUAL_DEMO_SYSTEM_PROMPT)

    # 1. Task goal
    parts.append(f"\n\nOur goal is {item['task_goal']}.")

    # 2. Demonstration introduction
    parts.append("\n\nHere is the demonstration:")

    # 3. Visual demo progress shifts (e.g., "<image> 0% <image> 25% ... <image> 100%")
    total_steps = normalize_total_steps(item['total_steps'])
    progress_shifts = format_visual_demo_progress_shifts(total_steps)
    parts.append(progress_shifts)

    # 4. Current state introduction
    parts.append("\n\nHere is the current state that you need to estimate:")
    parts.append("<image>")  # Final image tag for stage_to_estimate

    # 5. Task instructions (NO ground_truth section)
    parts.append("\n\n" + VISUAL_DEMO_INSTRUCTION)

    return "\n".join(parts)


def convert_visual_demo_item(
    original_item: Dict[str, Any],
    cot_response: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert a single Visual Demo item to LLaMA-Factory format.

    Args:
        original_item: Original data item
        cot_response: CoT response with 'response' field

    Returns:
        Converted item in ShareGPT format with messages and images
    """
    # Build user message
    user_content = build_user_message(original_item)

    # Extract assistant response from CoT
    assistant_content = cot_response['response']

    # Build image paths
    id = original_item['id']

    # Visual demo images (N images: 0% to 100%)
    visual_demo_paths = [
        build_image_path(id, filename)
        for filename in original_item['visual_demo']
    ]

    # Stage to estimate image (1 image)
    stage_filename = normalize_stage_to_estimate(original_item['stage_to_estimate'])
    stage_path = build_image_path(id, stage_filename)

    # Combine: visual_demo images + stage_to_estimate
    all_image_paths = visual_demo_paths + [stage_path]

    # Construct ShareGPT format (system prompt now in user message)
    converted = {
        "messages": [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ],
        "images": all_image_paths  # N+1 images
    }

    return converted


def load_cot_responses(cot_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load CoT responses and build a lookup dictionary.

    Args:
        cot_path: Path to CoT responses JSONL file

    Returns:
        Dictionary mapping (id, normalized_stage_filename) to list of CoT responses
    """
    cot_data = load_jsonl(cot_path)
    cot_lookup = {}

    for cot_item in cot_data:
        # Extract id and stage_to_estimate from meta_data
        meta = cot_item.get('meta_data', {})
        id = meta.get('id')
        stage_path = meta.get('stage_to_estimate', '')

        if not id:
            print(f"Warning: CoT response missing 'id' in meta_data, skipping")
            continue

        # Normalize stage filename for matching
        stage_filename = normalize_stage_filename(stage_path)

        # Build match key: (id, normalized_stage_filename)
        match_key = f"{id}|{stage_filename}"

        # Store as list to support multiple CoT responses per key
        if match_key not in cot_lookup:
            cot_lookup[match_key] = []
        cot_lookup[match_key].append(cot_item)

    total_responses = sum(len(v) for v in cot_lookup.values())
    print(f"Loaded {len(cot_data)} CoT responses with {len(cot_lookup)} unique keys ({total_responses} total)")
    return cot_lookup


def convert_visual_demo_dataset(
    original_data_path: str,
    cot_responses_path: str,
    output_file: str,
    filter_success: bool = True,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Convert entire Visual Demo dataset.

    Args:
        original_data_path: Path to original JSONL data
        cot_responses_path: Path to CoT responses JSONL
        output_file: Path to output JSON file
        filter_success: Only include samples with status='success'
        verbose: Print detailed logs

    Returns:
        Statistics dictionary
    """
    print(f"Loading original data from: {original_data_path}")
    original_data = load_jsonl(original_data_path)
    print(f"Loaded {len(original_data)} original samples")

    print(f"\nLoading CoT responses from: {cot_responses_path}")
    cot_lookup = load_cot_responses(cot_responses_path)

    # Convert samples
    converted_samples = []
    stats = {
        'dataset_name': os.path.basename(original_data_path).replace('.jsonl', ''),
        'total_original': len(original_data),
        'total_cot': len(cot_lookup),
        'matched': 0,
        'success': 0,
        'failed_status': 0,
        'unmatched': 0,
        'output_samples': 0,
        'validation_passed': 0,
        'output_file': output_file
    }

    print(f"\nProcessing {len(original_data)} original samples...")

    for idx, orig_item in enumerate(original_data):
        try:
            id = orig_item['id']
            stage_path = orig_item.get('stage_to_estimate', '')

            # Normalize stage filename for matching
            stage_filename = normalize_stage_filename(stage_path)

            # Build match key: (id, normalized_stage_filename)
            match_key = f"{id}|{stage_filename}"
            cot_items = cot_lookup.get(match_key)

            if not cot_items:
                stats['unmatched'] += 1
                if verbose and idx < 10:  # Show first few warnings
                    print(f"  Warning: No CoT match for {id} | {stage_filename}")
                continue

            # Process each CoT response for this original sample
            for cot_item in cot_items:
                stats['matched'] += 1

                # Check status
                meta = cot_item.get('meta_data', {})
                status = meta.get('status', 'unknown')

                if filter_success and status != 'success':
                    stats['failed_status'] += 1
                    if verbose and stats['failed_status'] <= 5:
                        print(f"  Skipping (status={status}): {id}")
                    continue

                stats['success'] += 1

                # Convert item
                converted = convert_visual_demo_item(orig_item, cot_item)

                # Validate: number of <image> tags should equal number of images
                # messages[0] = user, messages[1] = assistant
                user_content = converted['messages'][0]['content']
                images = converted['images']

                image_tag_count = count_image_tags(user_content)
                if image_tag_count != len(images):
                    print(f"  Warning: Image tag count mismatch for {id}: "
                          f"tags={image_tag_count}, images={len(images)}")
                    continue

                stats['validation_passed'] += 1

                # Validate assistant response
                assistant_content = converted['messages'][1]['content']
                if not validate_xml_tags(assistant_content):
                    print(f"  Warning: Invalid XML tags in assistant response for {id}")
                    continue

                converted_samples.append(converted)
                stats['output_samples'] += 1

        except Exception as e:
            print(f"  Error processing item {idx}: {e}")
            import traceback
            if verbose:
                traceback.print_exc()
            continue

    # Save output
    print(f"\nSaving {len(converted_samples)} samples to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_json(converted_samples, output_file)

    # Print statistics
    print_conversion_stats(stats)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Convert Visual Demo data to LLaMA-Factory ShareGPT format'
    )
    parser.add_argument(
        '--original-data',
        required=True,
        help='Path to original Visual Demo JSONL file'
    )
    parser.add_argument(
        '--cot-responses',
        required=True,
        help='Path to CoT responses JSONL file'
    )
    parser.add_argument(
        '--output-file',
        required=True,
        help='Path to output JSON file'
    )
    parser.add_argument(
        '--filter-success',
        action='store_true',
        help='Only include samples with status=success'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed logs'
    )

    args = parser.parse_args()

    # Validate input files exist
    if not os.path.exists(args.original_data):
        print(f"Error: Original data file not found: {args.original_data}")
        sys.exit(1)

    if not os.path.exists(args.cot_responses):
        print(f"Error: CoT responses file not found: {args.cot_responses}")
        sys.exit(1)

    # Run conversion
    convert_visual_demo_dataset(
        original_data_path=args.original_data,
        cot_responses_path=args.cot_responses,
        output_file=args.output_file,
        filter_success=args.filter_success,
        verbose=args.verbose
    )

    print("\n✅ Visual Demo conversion completed successfully!")


if __name__ == '__main__':
    main()
