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
VISUAL_DEMO_SYSTEM_PROMPT = """You are a progress estimator specializing in evaluating the progress of an ongoing task based on visual evidence. The demonstration consists of a sequence of video frames (images) showing how the task evolves from 0% (start) to 100% (completion). Your goal is to produce a human-like reasoning chain that logically supports the given progress score."""


# Task instruction text (from original prompt, excluding ground_truth section)
VISUAL_DEMO_INSTRUCTION = """**Abnormal Situation Handling:**
If you detect any of the following abnormal situations:
- The current state does not match the task goal or any visual demon images
- The operation appears to have failed or resulted in an error state
- You must output "n/a" for both `<ref>` and `<score>`. In your reasoning sections, clearly explain why the situation is abnormal and why no valid progress estimation can be made.

Your task:
1. Analyze the demonstration images to understand how the task visually progresses from start to completion.
2. Identify the frame (or frames) from the demonstration that are visually most similar to the current state image.
3. Compare the current state to that reference frame and determine whether it shows more or less progress.
4. Finally, provide a numeric progress estimation between 0% and 100%, or both `<ref>` and `<score>` be "n/a" while encontering abnormal situation.

Your response must strictly follow this format:
<ref_think>Your reasoning for choosing the closest demonstration frame as the reference, OR explanation of why the situation is abnormal and no reference can be identified</ref_think>
<ref>The progress score of your chosen reference frame, OR "n/a" if abnormal situation detected</ref>
<score_think>Your reasoning for comparing the current state image with the reference frame, OR explanation of why no valid progress score can be assigned</score_think>
<score>Your final estimated progress score, OR "n/a" if abnormal situation detected</score>"""


def build_user_message(item: Dict[str, Any]) -> str:
    """
    Build user message content for Visual Demo task.

    Args:
        item: Original data item with task_goal, visual_demo, total_steps, etc.

    Returns:
        Formatted user message string with N+1 <image> tags
    """
    parts = []

    # 1. Task goal
    parts.append(f"Our goal is {item['task_goal']}.")

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

    # Construct ShareGPT format with system prompt
    converted = {
        "messages": [
            {
                "role": "system",
                "content": VISUAL_DEMO_SYSTEM_PROMPT
            },
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
        # Extract id from meta_data
        meta = cot_item.get('meta_data', {})
        id = meta.get('id')

        if not id:
            print(f"Warning: CoT response missing 'id' in meta_data, skipping")
            continue

        # Store by id (we'll match more precisely during conversion)
        if id not in cot_lookup:
            cot_lookup[id] = []
        cot_lookup[id].append(cot_item)

    print(f"Loaded {len(cot_data)} CoT responses covering {len(cot_lookup)} unique IDs")
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

    # Build a more precise lookup by (id, progress_score) or (id, closest_idx)
    cot_by_id_score = {}
    for id, cot_list in cot_lookup.items():
        for cot_item in cot_list:
            # Use ground_truth_score or closest_idx for matching
            score = cot_item.get('ground_truth_score', '')
            closest_idx = cot_item.get('closest_idx', '')

            # Try multiple keys for flexible matching
            key1 = f"{id}|{score}"
            key2 = f"{id}|{closest_idx}"

            cot_by_id_score[key1] = cot_item
            if key2 != key1:
                cot_by_id_score[key2] = cot_item

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

    print(f"\nProcessing {len(original_data)} original samples...")

    for idx, orig_item in enumerate(original_data):
        try:
            id = orig_item['id']
            progress_score = orig_item.get('progress_score', '')
            closest_idx = orig_item.get('closest_idx', '')

            # Try to find matching CoT response
            # Method 1: Match by id + progress_score
            match_key = f"{id}|{progress_score}"
            cot_item = cot_by_id_score.get(match_key)

            # Method 2: Match by id + closest_idx
            if not cot_item:
                match_key = f"{id}|{closest_idx}"
                cot_item = cot_by_id_score.get(match_key)

            # Method 3: Try any CoT for this ID
            if not cot_item and id in cot_lookup and len(cot_lookup[id]) > 0:
                # If only one CoT for this ID, use it
                if len(cot_lookup[id]) == 1:
                    cot_item = cot_lookup[id][0]
                else:
                    # Try to match by closest_idx
                    for cot in cot_lookup[id]:
                        if str(cot.get('closest_idx', '')) == str(closest_idx):
                            cot_item = cot
                            break

            if not cot_item:
                stats['unmatched'] += 1
                if verbose and idx < 10:  # Show first few warnings
                    stage = normalize_stage_to_estimate(orig_item['stage_to_estimate'])
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
            converted = convert_visual_demo_item(orig_item, cot_item)

            # Validate: number of <image> tags should equal number of images
            # messages[0] = system, messages[1] = user, messages[2] = assistant
            user_content = converted['messages'][1]['content']
            images = converted['images']

            image_tag_count = count_image_tags(user_content)
            if image_tag_count != len(images):
                print(f"  Warning: Image tag count mismatch for {id}: "
                      f"tags={image_tag_count}, images={len(images)}")
                continue

            stats['validation_passed'] += 1

            # Validate assistant response
            assistant_content = converted['messages'][2]['content']
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
