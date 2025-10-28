import os
import sys
import json
import argparse
import re
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from image_edit_quality_dataset import load_image_edit_quality_dataset, validate_edited_image_path
from image_edit_quality_prompt import build_image_edit_quality_prompt_from_item, IMAGE_EDIT_QUALITY_SYSTEM_PROMPT
from qwen2_vl.model import Qwen2VLChat


def parse_quality_response(response: str) -> Dict[str, Any]:
    """
    Parse the model's response to extract yes/no quality judgment.

    Expected format: Simple "yes" or "no" answer

    Args:
        response: Model output string

    Returns:
        Dictionary with parsed fields:
        {
            'quality_judgment': str or None ('yes', 'no', or None if parse error),
            'parse_error': bool
        }
    """
    result = {
        'quality_judgment': None,
        'parse_error': False
    }

    try:
        # Normalize response: lowercase and strip whitespace
        response_normalized = response.strip().lower()

        # Try to extract yes/no from the response
        # Look for standalone "yes" or "no" or as part of common patterns

        # Pattern 1: Standalone yes/no
        if re.match(r'^\s*yes\s*$', response_normalized):
            result['quality_judgment'] = 'yes'
            return result

        if re.match(r'^\s*no\s*$', response_normalized):
            result['quality_judgment'] = 'no'
            return result

        # Pattern 2: "your answer: yes" or "answer: no"
        answer_match = re.search(r'(?:your\s+)?answer\s*[:\-]?\s*(yes|no)', response_normalized)
        if answer_match:
            result['quality_judgment'] = answer_match.group(1)
            return result

        # Pattern 3: Starts with yes/no
        if response_normalized.startswith('yes'):
            result['quality_judgment'] = 'yes'
            return result

        if response_normalized.startswith('no'):
            result['quality_judgment'] = 'no'
            return result

        # Pattern 4: Contains "yes" or "no" as the first clear indicator
        # Look for yes/no preceded by common phrases
        yes_pattern = re.search(r'\b(yes)\b', response_normalized)
        no_pattern = re.search(r'\b(no)\b', response_normalized)

        # If both found, take the first one
        if yes_pattern and no_pattern:
            if yes_pattern.start() < no_pattern.start():
                result['quality_judgment'] = 'yes'
            else:
                result['quality_judgment'] = 'no'
            return result
        elif yes_pattern:
            result['quality_judgment'] = 'yes'
            return result
        elif no_pattern:
            result['quality_judgment'] = 'no'
            return result

        # If no clear yes/no found, mark as parse error
        result['parse_error'] = True

    except Exception as e:
        result['parse_error'] = True

    return result


def run_image_edit_quality_eval(args):
    """
    Run image edit quality evaluation with single-process batch inference.
    Optimized for 72B models using model parallelism across multiple GPUs.
    """

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")
    image_root = args.image_root if hasattr(args, 'image_root') and args.image_root else None

    from io import StringIO
    if not args.verbose:
        old_stdout = sys.stdout
        sys.stdout = StringIO()

    data = load_image_edit_quality_dataset(
        args.dataset_path,
        image_root=image_root
    )

    if not args.verbose:
        sys.stdout = old_stdout

    if args.limit > 0:
        data = data[:args.limit]
        print(f"Limited to first {args.limit} samples")

    # Get GPU configuration
    gpu_ids = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')]
    num_gpus = len(gpu_ids)

    print(f"\n{'='*70}")
    print(f"IMAGE EDIT QUALITY EVALUATION - SINGLE PROCESS MODE (72B Optimized)")
    print(f"{'='*70}")
    print(f"GPUs available: {num_gpus} ({gpu_ids})")
    print(f"Model parallelism: {'ENABLED' if num_gpus > 1 else 'DISABLED'}")
    print(f"Total samples: {len(data)}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*70}\n")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Initialize model once (will automatically distribute across all visible GPUs)
    print("Loading model... (this may take a few minutes for 72B models)")
    model = Qwen2VLChat(
        model_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        use_custom_prompt=False,
        system_prompt=IMAGE_EDIT_QUALITY_SYSTEM_PROMPT,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        verbose=args.verbose
    )
    print("Model loaded successfully!\n")

    # Process data in batches
    batch_size = args.batch_size
    results = []

    # Statistics tracking
    valid_edit_count = 0
    invalid_edit_count = 0
    parse_error_count = 0

    # Progress bar
    pbar = tqdm(total=len(data), desc="Processing", ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    i = 0
    while i < len(data):
        batch_end = min(i + batch_size, len(data))
        batch_items = data[i:batch_end]

        try:
            # Build batch prompts
            batch_messages = []
            valid_batch_items = []

            for item in batch_items:
                # Validate image path
                is_valid, error_msg = validate_edited_image_path(item)
                if not is_valid:
                    # Skip this item, record error
                    result = {
                        "model_response": f"Validation error: {error_msg}",
                        "quality_judgment": None,
                        "id": item.get('id', ''),
                        "task_goal": item.get('task_goal', ''),
                        "raw_demo": item.get('raw_demo', ''),
                        "editing_prompt": item.get('prompt', ''),
                        "editing_strategy": item.get('editing_strategy', ''),
                        "edited_image": item.get('edited_image', ''),
                        "data_source": item.get('data_source', ''),
                        "status": "failed"
                    }
                    results.append(result)
                    parse_error_count += 1
                    pbar.update(1)
                    continue

                messages = build_image_edit_quality_prompt_from_item(
                    item,
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels
                )
                batch_messages.append(messages)
                valid_batch_items.append(item)

            if not batch_messages:
                # All items in batch failed validation
                i = batch_end
                continue

            # Batch inference
            batch_responses = model.generate(batch_messages)

            # Process responses
            for item, response in zip(valid_batch_items, batch_responses):
                try:
                    # Parse response
                    parsed = parse_quality_response(response)
                    quality_judgment = parsed['quality_judgment']
                    has_error = parsed['parse_error']

                    # Update statistics
                    if has_error:
                        parse_error_count += 1
                    elif quality_judgment == 'yes':
                        valid_edit_count += 1
                    elif quality_judgment == 'no':
                        invalid_edit_count += 1

                    result = {
                        "model_response": response,
                        "quality_judgment": quality_judgment,
                        "id": item.get('id', ''),
                        "task_goal": item.get('task_goal', ''),
                        "raw_demo": item.get('raw_demo', ''),
                        "editing_prompt": item.get('prompt', ''),
                        "editing_strategy": item.get('editing_strategy', ''),
                        "edited_image": item.get('edited_image', ''),
                        "data_source": item.get('data_source', ''),
                        "status": "failed" if has_error else "success"
                    }

                    results.append(result)
                    pbar.update(1)

                    # Update progress bar stats
                    total_processed = len(results)
                    valid_rate = valid_edit_count / total_processed * 100 if total_processed > 0 else 0.0
                    invalid_rate = invalid_edit_count / total_processed * 100 if total_processed > 0 else 0.0
                    error_rate = parse_error_count / total_processed * 100 if total_processed > 0 else 0.0
                    pbar.set_postfix_str(f"Valid={valid_rate:.1f}% Invalid={invalid_rate:.1f}% Error={error_rate:.1f}%")

                except Exception as e:
                    # Parse error for this specific item
                    result = {
                        "model_response": f"Processing error: {str(e)}\nResponse: {response if 'response' in locals() else ''}",
                        "quality_judgment": None,
                        "id": item.get('id', ''),
                        "task_goal": item.get('task_goal', ''),
                        "raw_demo": item.get('raw_demo', ''),
                        "editing_prompt": item.get('prompt', ''),
                        "editing_strategy": item.get('editing_strategy', ''),
                        "edited_image": item.get('edited_image', ''),
                        "data_source": item.get('data_source', ''),
                        "status": "failed"
                    }
                    results.append(result)
                    parse_error_count += 1
                    pbar.update(1)

        except Exception as e:
            # Batch error - mark all items in batch as errors
            for item in batch_items:
                result = {
                    "model_response": f"Batch error: {str(e)}",
                    "quality_judgment": None,
                    "id": item.get('id', ''),
                    "task_goal": item.get('task_goal', ''),
                    "raw_demo": item.get('raw_demo', ''),
                    "editing_prompt": item.get('prompt', ''),
                    "editing_strategy": item.get('editing_strategy', ''),
                    "edited_image": item.get('edited_image', ''),
                    "data_source": item.get('data_source', ''),
                    "status": "failed"
                }
                results.append(result)
                parse_error_count += 1
            pbar.update(len(batch_items))

        # Save results periodically (every 10 batches)
        if (i // batch_size) % 10 == 0:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')

        i = batch_end

    pbar.close()

    # Final save
    print("\nSaving final results...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    # Calculate final statistics
    total_processed = len(results)
    valid_edit_rate = valid_edit_count / total_processed if total_processed > 0 else 0.0
    invalid_edit_rate = invalid_edit_count / total_processed if total_processed > 0 else 0.0
    parse_error_rate = parse_error_count / total_processed if total_processed > 0 else 0.0

    # Print final summary
    print("\n" + "=" * 70)
    print("IMAGE EDIT QUALITY EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total samples: {len(data)}")
    print(f"Processed: {total_processed}")
    print(f"Valid edits (yes): {valid_edit_count} ({valid_edit_rate*100:.2f}%)")
    print(f"Invalid edits (no): {invalid_edit_count} ({invalid_edit_rate*100:.2f}%)")
    print(f"Parse errors: {parse_error_count} ({parse_error_rate*100:.2f}%)")
    print(f"Results saved to: {args.output_file}")
    print("=" * 70)

    # Save summary
    summary_file = args.output_file.replace('.jsonl', '_summary.json')
    summary = {
        "total_samples": len(data),
        "processed": total_processed,
        "valid_edits": valid_edit_count,
        "invalid_edits": invalid_edit_count,
        "parse_errors": parse_error_count,
        "valid_edit_rate": valid_edit_rate,
        "invalid_edit_rate": invalid_edit_rate,
        "parse_error_rate": parse_error_rate,
        "batch_size": args.batch_size,
        "num_gpus": num_gpus,
        "dataset_path": args.dataset_path,
        "model_path": args.model_path,
        "output_file": args.output_file,
        "image_root": args.image_root
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Image Edit Quality Evaluation - Single Process (72B Optimized)"
    )

    # Required arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the Qwen2.5-VL model")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the edited images dataset (JSONL format)")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output JSONL file path for results")
    parser.add_argument("--image-root", type=str, required=True,
                        help="Root directory to prepend to image paths")

    # Optional arguments
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference (default: 1 for 72B models)")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit number of samples to process (-1 for all)")

    # Model parameters
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (default: 0.1 for deterministic output)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling parameter (default: 0.9)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling parameter (default: 50)")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Maximum number of tokens to generate (default: 128)")

    # Image processing parameters
    parser.add_argument("--min-pixels", type=int, default=1280*28*28,
                        help="Minimum pixels for image processing")
    parser.add_argument("--max-pixels", type=int, default=5120*28*28,
                        help="Maximum pixels for image processing")

    # Misc
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file not found: {args.dataset_path}")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        sys.exit(1)

    if args.image_root and not os.path.exists(args.image_root):
        print(f"Error: Image root directory not found: {args.image_root}")
        sys.exit(1)

    # Run evaluation
    run_image_edit_quality_eval(args)


if __name__ == "__main__":
    main()
