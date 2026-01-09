import os
import sys
import json
import argparse
import time
import re
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import torch

# Local imports
from datasets.text_object_replacement_dataset import load_text_object_replacement_dataset, validate_image_path
from prompts.text_object_replacement_prompt import build_text_object_replacement_prompt_from_item, TEXT_OBJECT_REPLACEMENT_SYSTEM_PROMPT
from core.model import Qwen2VLChat


def parse_text_object_replacement_response(response: str) -> Dict[str, Any]:
    """
    Parse the model's response to extract XML tags.

    Expected format:
    <think>
    - Original object identified: [object name]
    - Objects visible in current image: [list other objects if any]
    - Replacement strategy used: [Priority 1 or Priority 2]
    - Replacement object chosen: [new object name]
    - Reasoning: [why this creates a valid negative case]
    </think>

    <edited_goal> "put your edited task goal here" </edited_goal>

    <edited_demo>
    ["your edited step 1", "your edited step 2", "your edited step 3", ..., "your edited step n"]
    </edited_demo>

    Args:
        response: Model output string

    Returns:
        Dictionary with parsed fields:
        {
            'original_object': str,
            'replacement_object': str,
            'replacement_strategy': str,
            'reasoning': str,
            'edited_goal': str,
            'edited_demo': List[str],
            'parse_error': bool
        }
    """
    result = {
        'original_object': '',
        'replacement_object': '',
        'replacement_strategy': '',
        'reasoning': '',
        'edited_goal': '',
        'edited_demo': [],
        'parse_error': False
    }

    try:
        # Extract think section
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()

            # Parse think content for structured information
            original_obj_match = re.search(r'Original object identified:\s*(.+?)(?:\n|$)', think_content, re.IGNORECASE)
            if original_obj_match:
                result['original_object'] = original_obj_match.group(1).strip()

            replacement_obj_match = re.search(r'Replacement object chosen:\s*(.+?)(?:\n|$)', think_content, re.IGNORECASE)
            if replacement_obj_match:
                result['replacement_object'] = replacement_obj_match.group(1).strip()

            strategy_match = re.search(r'Replacement strategy used:\s*(.+?)(?:\n|$)', think_content, re.IGNORECASE)
            if strategy_match:
                result['replacement_strategy'] = strategy_match.group(1).strip()

            reasoning_match = re.search(r'Reasoning:\s*(.+?)$', think_content, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                result['reasoning'] = reasoning_match.group(1).strip()

        # Extract edited_goal
        edited_goal_match = re.search(r'<edited_goal>\s*"?([^"<]+)"?\s*</edited_goal>', response, re.DOTALL)
        if edited_goal_match:
            result['edited_goal'] = edited_goal_match.group(1).strip()
        else:
            result['parse_error'] = True

        # Extract edited_demo (JSON array)
        edited_demo_match = re.search(r'<edited_demo>\s*(\[.*?\])\s*</edited_demo>', response, re.DOTALL)
        if edited_demo_match:
            try:
                demo_json = edited_demo_match.group(1).strip()
                result['edited_demo'] = json.loads(demo_json)

                # Validate it's a list
                if not isinstance(result['edited_demo'], list):
                    result['parse_error'] = True
                    result['edited_demo'] = []
            except json.JSONDecodeError:
                result['parse_error'] = True
                result['edited_demo'] = []
        else:
            result['parse_error'] = True

    except Exception as e:
        result['parse_error'] = True

    return result


def run_text_object_replacement_inference_single(args):
    """
    Run text object replacement with single-process batch inference.
    Optimized for 72B/32B models using model parallelism across multiple GPUs.
    """

    # Load dataset (already expanded N times)
    print(f"Loading dataset from {args.dataset_path}")
    image_root = args.image_root if hasattr(args, 'image_root') and args.image_root else None

    from io import StringIO
    if not args.verbose:
        old_stdout = sys.stdout
        sys.stdout = StringIO()

    data = load_text_object_replacement_dataset(
        args.dataset_path,
        num_inferences=args.num_inferences,
        image_root=image_root
    )

    if not args.verbose:
        sys.stdout = old_stdout

    if args.limit > 0:
        data = data[:args.limit]
        print(f"Limited to first {args.limit} samples (after expansion)")

    # Get GPU configuration
    gpu_ids = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')]
    num_gpus = len(gpu_ids)

    print(f"\n{'='*70}")
    print(f"TEXT OBJECT REPLACEMENT INFERENCE - SINGLE PROCESS MODE (72B Optimized)")
    print(f"{'='*70}")
    print(f"GPUs available: {num_gpus} ({gpu_ids})")
    print(f"Model parallelism: {'ENABLED' if num_gpus > 1 else 'DISABLED'}")
    print(f"Total samples (expanded): {len(data)}")
    print(f"Original samples: {len(data) // args.num_inferences}")
    print(f"Inferences per sample: {args.num_inferences}")
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
        system_prompt=TEXT_OBJECT_REPLACEMENT_SYSTEM_PROMPT,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        verbose=args.verbose
    )
    print("Model loaded successfully!\n")

    # Process data in batches
    batch_size = args.batch_size
    results = []

    # Statistics tracking
    error_count = 0

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
                # Validate image path if provided
                is_valid, error_msg = validate_image_path(item)
                if not is_valid:
                    # Skip this item, record error
                    result = {
                        "original_object": None,
                        "replacement_object": None,
                        "reasoning": None,
                        "edited_goal": None,
                        "edited_demo": None,
                        "raw_response": f"Validation error: {error_msg}",
                        "meta_data": dict(item, status="failed")
                    }
                    results.append(result)
                    error_count += 1
                    pbar.update(1)
                    continue

                messages = build_text_object_replacement_prompt_from_item(item)
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
                    parsed = parse_text_object_replacement_response(response)
                    has_error = parsed['parse_error']

                    # Update statistics
                    if has_error:
                        error_count += 1

                    # Preserve all original fields in meta_data
                    meta_data = dict(item)
                    meta_data['status'] = "failed" if has_error else "success"

                    result = {
                        "original_object": parsed['original_object'],
                        "replacement_object": parsed['replacement_object'],
                        "replacement_strategy": parsed['replacement_strategy'],
                        "reasoning": parsed['reasoning'],
                        "edited_goal": parsed['edited_goal'],
                        "edited_demo": parsed['edited_demo'],
                        "raw_response": response,
                        "meta_data": meta_data
                    }

                    results.append(result)
                    pbar.update(1)

                    # Update progress bar stats
                    error_rate = error_count / len(results) * 100 if results else 0.0
                    pbar.set_postfix_str(f"ErrorRate={error_rate:.1f}%")

                except Exception as e:
                    # Parse error for this specific item
                    meta_data = dict(item)
                    meta_data['status'] = "failed"

                    result = {
                        "original_object": None,
                        "replacement_object": None,
                        "reasoning": None,
                        "edited_goal": None,
                        "edited_demo": None,
                        "raw_response": f"Processing error: {str(e)}\nResponse: {response if 'response' in locals() else ''}",
                        "meta_data": meta_data
                    }
                    results.append(result)
                    error_count += 1
                    pbar.update(1)

        except Exception as e:
            # Batch error - mark all items in batch as errors
            for item in batch_items:
                meta_data = dict(item)
                meta_data['status'] = "failed"

                result = {
                    "original_object": None,
                    "replacement_object": None,
                    "reasoning": None,
                    "edited_goal": None,
                    "edited_demo": None,
                    "raw_response": f"Batch error: {str(e)}",
                    "meta_data": meta_data
                }
                results.append(result)
                error_count += 1
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
    valid_results = [r for r in results if r['meta_data']['status'] == 'success']
    error_rate = error_count / len(results) if results else 0.0

    # Print final summary
    print("\n" + "=" * 70)
    print("TEXT OBJECT REPLACEMENT SUMMARY")
    print("=" * 70)
    print(f"Total samples (expanded): {len(data)}")
    print(f"Original samples: {len(data) // args.num_inferences}")
    print(f"Inferences per sample: {args.num_inferences}")
    print(f"Processed: {len(results)}")
    print(f"Errors: {error_count} ({error_rate*100:.2f}%)")
    print(f"Success: {len(valid_results)} ({len(valid_results)/len(results)*100:.2f}%)")
    print(f"Results saved to: {args.output_file}")
    print("=" * 70)

    # Save summary
    summary_file = args.output_file.replace('.jsonl', '_summary.json')
    summary = {
        "total_samples_expanded": len(data),
        "original_samples": len(data) // args.num_inferences,
        "num_inferences_per_sample": args.num_inferences,
        "processed": len(results),
        "errors": error_count,
        "error_rate": error_rate,
        "success_count": len(valid_results),
        "success_rate": len(valid_results) / len(results) if results else 0.0,
        "batch_size": args.batch_size,
        "num_gpus": num_gpus,
        "dataset_path": args.dataset_path,
        "model_path": args.model_path,
        "output_file": args.output_file,
        "image_root": args.image_root if image_root else "N/A"
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Text Object Replacement - Single Process (72B Optimized)"
    )

    # Required arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the Qwen2.5-VL model")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the dataset (JSONL format)")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output JSONL file path for results")

    # Optional arguments
    parser.add_argument("--image-root", type=str, default=None,
                        help="Root directory to prepend to image paths (optional, for image-based analysis)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference (default: 1 for 72B models)")
    parser.add_argument("--num-inferences", type=int, default=1,
                        help="Number of inferences per sample (data expansion factor, default: 1)")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit number of samples to process after expansion (-1 for all)")

    # Model parameters
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling parameter (default: 0.9)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling parameter (default: 50)")
    parser.add_argument("--max-new-tokens", type=int, default=20000,
                        help="Maximum number of tokens to generate (default: 20000)")

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

    # Run inference
    run_text_object_replacement_inference_single(args)


if __name__ == "__main__":
    main()
