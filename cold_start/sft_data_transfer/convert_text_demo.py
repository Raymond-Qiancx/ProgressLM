#!/usr/bin/env python3
"""
Convert Text Demo data to LLaMA-Factory ShareGPT format.

This script merges original text demo data with CoT responses to create
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
    format_text_demo_with_progress,
    normalize_stage_to_estimate,
    normalize_total_steps,
    validate_image_tag_count,
    validate_xml_tags,
    print_conversion_stats
)


# System prompt for inference mode
TEXT_DEMO_SYSTEM_PROMPT = """You are a progress estimator that evaluates the progress of the current state during an ongoing task based on a textual demonstration. The demonstration consists of a sequence of text-based steps and their corresponding progress value (ranging from 0% to 100%), showing how the task evolves from start to completion."""


# Task instruction text (from original prompt, excluding ground_truth section)
TEXT_DEMO_INSTRUCTION = """Your task:
1. Check the current state image carefully.
2. Analyze the textual demonstration to understand how the task progresses from start to completion.
3. Identify the reference step from the textual demonstration that are most related to the current state image.
4. Compare the current state image with the chosen reference step, determining whether the image is behind or after the reference step.
5. Estimate the progress numerically as a floating-point value between 0% and 100%, or directly output n/a if you really cannot match the current state image to any of the steps from demonstration.


Your response must strictly follow this format:
<ref_think>Your reasoning for choosing the most similar text_demo step as the reference</ref_think>
<ref>which text demo is most semantically similar to the current state, and output only the number of that text demo</ref>
<score_think>Your reasoning for comparing the current state image with the reference step(s)</score_think>
<score>Your final estimated progress score here</score>"""


def build_user_message(item: Dict[str, Any]) -> str:
    """
    Build user message content for Text Demo task.

    Args:
        item: Original data item with task_goal, text_demo, total_steps, etc.

    Returns:
        Formatted user message string with single <image> tag
    """
    parts = []

    # 0. System prompt at the beginning of user message
    parts.append(TEXT_DEMO_SYSTEM_PROMPT)

    # 1. Task goal
    parts.append(f"\n\nOur goal is {item['task_goal']}.")

    # 2. Demonstration introduction
    parts.append("\n\nHere is the demonstration:")

    # 3. Formatted text demo with progress
    total_steps = normalize_total_steps(item['total_steps'])
    formatted_demo = format_text_demo_with_progress(item['text_demo'], total_steps)
    parts.append(formatted_demo)

    # 4. Current state introduction
    parts.append("\n\nHere is the current state that you need to estimate:")
    parts.append("<image>")  # Single image tag

    # 5. Task instructions (NO ground_truth section)
    parts.append("\n\n" + TEXT_DEMO_INSTRUCTION)

    return "\n".join(parts)


def convert_text_demo_item(
    original_item: Dict[str, Any],
    cot_response: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert a single Text Demo item to LLaMA-Factory format.

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

    # Build image path: {id}/{stage_to_estimate}
    stage_filename = normalize_stage_to_estimate(original_item['stage_to_estimate'])
    image_path = build_image_path(original_item['id'], stage_filename)

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
        "images": [image_path]  # Single image
    }

    return converted


def load_cot_responses(cot_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load CoT responses and build a lookup dictionary.

    Args:
        cot_path: Path to CoT responses JSONL file

    Returns:
        Dictionary mapping match_key to CoT response
    """
    cot_data = load_jsonl(cot_path)
    cot_lookup = {}

    for cot_item in cot_data:
        # Extract id and stage from meta_data
        meta = cot_item.get('meta_data', {})
        id = meta.get('id')

        if not id:
            print(f"Warning: CoT response missing 'id' in meta_data, skipping")
            continue

        # For Text Demo, we need to infer stage_to_estimate from the response or match
        # Since CoT doesn't always have stage_to_estimate, we'll match by id only first
        # and handle multiple stages per id during matching
        # Store by id for now
        if id not in cot_lookup:
            cot_lookup[id] = []
        cot_lookup[id].append(cot_item)

    print(f"Loaded {len(cot_data)} CoT responses covering {len(cot_lookup)} unique IDs")
    return cot_lookup


def convert_text_demo_dataset(
    original_data_path: str,
    cot_responses_path: str,
    output_file: str,
    filter_success: bool = True,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Convert entire Text Demo dataset.

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
        'total_cot': sum(len(v) for v in cot_lookup.values()),
        'matched': 0,
        'success': 0,
        'failed_status': 0,
        'unmatched': 0,
        'output_samples': 0,
        'validation_passed': 0,
        'output_file': output_file
    }

    # Build a more precise lookup: {id}|{stage} -> cot_responses
    cot_precise_lookup = {}
    for id, cot_list in cot_lookup.items():
        for cot_item in cot_list:
            # Try to infer stage from response or use a counter
            # Since we don't have stage in CoT, we'll match sequentially
            # Better approach: use closest_idx or other fields if available
            pass

    # Alternative: Build lookup by (id, closest_idx) or (id, ground_truth_score)
    # For now, let's try matching by id + progress_score
    cot_by_id_score = {}
    for id, cot_list in cot_lookup.items():
        for cot_item in cot_list:
            meta = cot_item.get('meta_data', {})
            # Build a match key that includes more info
            # Since we don't have stage in CoT, we'll need to match differently
            # Let's store all CoT for an ID and match by closest_idx or score
            key = f"{id}|{cot_item.get('ground_truth_score', '')}"
            cot_by_id_score[key] = cot_item

    print(f"\nProcessing {len(original_data)} original samples...")

    for idx, orig_item in enumerate(original_data):
        try:
            id = orig_item['id']
            stage = normalize_stage_to_estimate(orig_item['stage_to_estimate'])
            progress_score = orig_item.get('progress_score', '')

            # Try to find matching CoT response
            # Method 1: Match by id + progress_score
            match_key = f"{id}|{progress_score}"
            cot_item = cot_by_id_score.get(match_key)

            # Method 2: If not found, try matching by id + closest_idx
            if not cot_item:
                closest_idx = orig_item.get('closest_idx')
                # Search through CoT list for this id
                if id in cot_lookup:
                    for cot in cot_lookup[id]:
                        if cot.get('closest_idx') == closest_idx:
                            cot_item = cot
                            break

            if not cot_item:
                stats['unmatched'] += 1
                if verbose and idx < 10:  # Show first few warnings
                    print(f"  Warning: No CoT match for {id} | {stage}")
                continue

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
            converted = convert_text_demo_item(orig_item, cot_item)

            # Validate
            # messages[0] = user, messages[1] = assistant
            user_content = converted['messages'][0]['content']
            images = converted['images']
            if validate_image_tag_count(user_content, images):
                stats['validation_passed'] += 1
            else:
                print(f"  Warning: Image tag count mismatch for {id}")
                continue

            # Validate assistant response
            assistant_content = converted['messages'][1]['content']
            if not validate_xml_tags(assistant_content):
                print(f"  Warning: Invalid XML tags in assistant response for {id}")
                continue

            converted_samples.append(converted)
            stats['output_samples'] += 1

        except Exception as e:
            print(f"  Error processing item {idx}: {e}")
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
        description='Convert Text Demo data to LLaMA-Factory ShareGPT format'
    )
    parser.add_argument(
        '--original-data',
        required=True,
        help='Path to original Text Demo JSONL file'
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
    convert_text_demo_dataset(
        original_data_path=args.original_data,
        cot_responses_path=args.cot_responses,
        output_file=args.output_file,
        filter_success=args.filter_success,
        verbose=args.verbose
    )

    print("\n✅ Text Demo conversion completed successfully!")


if __name__ == '__main__':
    main()
