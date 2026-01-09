#!/usr/bin/env python3
"""
Convert Visual Demo data to LLaMA-Factory ShareGPT format (Simple Version).

This script converts visual demo data using ground truth progress_score
as the assistant response, without requiring CoT responses.
"""
import argparse
import os
import sys
from typing import Dict, Any, List
from utils import (
    load_jsonl,
    save_json,
    build_image_path,
    format_visual_demo_progress_shifts,
    normalize_stage_to_estimate,
    normalize_total_steps,
    count_image_tags,
    print_conversion_stats
)


# System prompt for inference mode
VISUAL_DEMO_SYSTEM_PROMPT = """You are a progress estimator that evaluates the progress of the current state during an ongoing task based on a visual demonstration. The demonstration consists of a sequence of vision-based states and their corresponding progress value (ranging from 0% to 100%), showing how the task evolves from start to completion."""


# Task instruction text
VISUAL_DEMO_INSTRUCTION = """1. Check the current state image carefully.
2. Analyze the overall task goal and visual demonstration to understand how the task progresses from start to completion.
3. Identify the reference states from the visual demonstration that are most related to the current state image.
4. Compare the current state image with the chosen reference state, determining whether the image is behind or after the reference state.
5. Estimate the progress numerically as a floating-point value between 0% and 100%, or directly output the "n/a" if you really cannot match the current state image to any of the states from demonstration.

Your answer only needs to output the final progress score you estimated, no other words needed."""


def format_progress_score(score: Any) -> str:
    """
    Format progress_score to ensure proper format.

    Rules:
    - n/a -> "n/a"
    - Numeric (50) -> "50%"
    - String with % ("50%") -> "50%"
    - String without % ("50") -> "50%"

    Args:
        score: Progress score value

    Returns:
        Formatted progress score string
    """
    # Handle n/a
    if score == 'n/a':
        return 'n/a'

    # Handle numeric types
    if isinstance(score, (int, float)):
        return f"{score}%"

    # Handle string types
    if isinstance(score, str):
        # Already has %
        if score.endswith('%'):
            return score
        # Try to convert to number and add %
        try:
            float(score)
            return f"{score}%"
        except ValueError:
            # Not a number, return as-is
            return score

    # Fallback
    return str(score)


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

    # 5. Task instructions
    parts.append("\n\n" + VISUAL_DEMO_INSTRUCTION)

    return "\n".join(parts)


def convert_visual_demo_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single Visual Demo item to LLaMA-Factory format.

    Args:
        item: Original data item

    Returns:
        Converted item in ShareGPT format with messages and images
    """
    # Build user message
    user_content = build_user_message(item)

    # Build assistant response from progress_score
    progress_score = item.get('progress_score', 'n/a')
    assistant_content = format_progress_score(progress_score)

    # Build image paths
    id = item['id']

    # Visual demo images (N images: 0% to 100%)
    visual_demo_paths = [
        build_image_path(id, filename)
        for filename in item['visual_demo']
    ]

    # Stage to estimate image (1 image)
    stage_filename = normalize_stage_to_estimate(item['stage_to_estimate'])
    stage_path = build_image_path(id, stage_filename)

    # Combine: visual_demo images + stage_to_estimate
    all_image_paths = visual_demo_paths + [stage_path]

    # Construct ShareGPT format (system prompt in user message)
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


def convert_visual_demo_dataset(
    input_file: str,
    output_file: str,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Convert entire Visual Demo dataset.

    Args:
        input_file: Path to input JSONL data
        output_file: Path to output JSON file
        verbose: Print detailed logs

    Returns:
        Statistics dictionary
    """
    print(f"Loading data from: {input_file}")
    input_data = load_jsonl(input_file)
    print(f"Loaded {len(input_data)} samples")

    # Convert samples
    converted_samples = []
    stats = {
        'dataset_name': os.path.basename(input_file).replace('.jsonl', ''),
        'total_input': len(input_data),
        'converted': 0,
        'validation_passed': 0,
        'output_samples': 0,
        'output_file': output_file
    }

    print(f"\nProcessing {len(input_data)} samples...")

    for idx, item in enumerate(input_data):
        try:
            # Convert item
            converted = convert_visual_demo_item(item)
            stats['converted'] += 1

            # Validate: number of <image> tags should equal number of images
            user_content = converted['messages'][0]['content']
            images = converted['images']

            image_tag_count = count_image_tags(user_content)
            if image_tag_count != len(images):
                if verbose and stats['validation_passed'] <= 5:
                    print(f"  Warning: Image tag count mismatch for sample {idx}: "
                          f"tags={image_tag_count}, images={len(images)}")
                continue

            stats['validation_passed'] += 1
            converted_samples.append(converted)
            stats['output_samples'] += 1

        except Exception as e:
            if verbose and idx < 10:
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
        description='Convert Visual Demo data to LLaMA-Factory ShareGPT format (Simple Version)'
    )
    parser.add_argument(
        '--input-file',
        required=True,
        help='Path to input Visual Demo JSONL file'
    )
    parser.add_argument(
        '--output-file',
        required=True,
        help='Path to output JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed logs'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # Run conversion
    convert_visual_demo_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        verbose=args.verbose
    )

    print("\n✅ Visual Demo conversion completed successfully!")


if __name__ == '__main__':
    main()
