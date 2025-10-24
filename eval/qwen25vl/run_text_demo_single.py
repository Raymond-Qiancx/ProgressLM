import os
import sys
import json
import argparse
import time
import re
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from text_demo_dataset import load_text_demo_dataset, validate_image_path
from text_demo_prompt import build_text_demo_prompt_from_item, TEXT_DEMO_SYSTEM_PROMPT
from qwen2_vl.model import Qwen2VLChat


def parse_text_demo_response(response: str) -> Dict[str, Any]:
    """
    Parse the model's response to extract XML tags.

    Expected format:
    <ref_think>reasoning...</ref_think>
    <ref>2</ref>  (now expects 1-based integer index)
    <score_think>reasoning...</score_think>
    <score>33%</score>  (supports both "33%" and "0.33" formats)

    Args:
        response: Model output string

    Returns:
        Dictionary with parsed fields:
        {
            'ref_think': str,
            'ref': str or int (1-based step number),
            'score_think': str,
            'score': float or None (in 0.0-1.0 range),
            'parse_error': bool
        }
    """
    result = {
        'ref_think': '',
        'ref': '',
        'score_think': '',
        'score': None,
        'parse_error': False
    }

    try:
        # Extract ref_think
        ref_think_match = re.search(r'<ref_think>(.*?)</ref_think>', response, re.DOTALL)
        if ref_think_match:
            result['ref_think'] = ref_think_match.group(1).strip()

        # Extract ref (now expects integer 1-based index)
        ref_match = re.search(r'<ref>(.*?)</ref>', response, re.DOTALL)
        if ref_match:
            ref_str = ref_match.group(1).strip()
            try:
                # Extract just the number (handle "No. 2", "2", "step 2", etc.)
                ref_num = re.search(r'\d+', ref_str)
                if ref_num:
                    result['ref'] = int(ref_num.group())
                else:
                    result['ref'] = ref_str  # Keep original if no number found
            except (ValueError, AttributeError):
                result['ref'] = ref_str

        # Extract score_think
        score_think_match = re.search(r'<score_think>(.*?)</score_think>', response, re.DOTALL)
        if score_think_match:
            result['score_think'] = score_think_match.group(1).strip()

        # Extract score (supports both "33%" and "0.33")
        score_match = re.search(r'<score>(.*?)</score>', response, re.DOTALL)
        if score_match:
            score_str = score_match.group(1).strip()
            try:
                # Remove % sign if present
                if score_str.endswith('%'):
                    score_value = float(score_str[:-1]) / 100.0
                else:
                    score_value = float(score_str)
                    # If > 1.0, assume it's percentage without % sign
                    if score_value > 1.0:
                        score_value = score_value / 100.0

                # Clamp to [0, 1]
                result['score'] = max(0.0, min(1.0, score_value))
            except ValueError:
                result['parse_error'] = True
        else:
            result['parse_error'] = True

    except Exception as e:
        result['parse_error'] = True

    return result


def calculate_evaluation_score(predicted: Optional[float], ground_truth: float) -> float:
    """
    Calculate evaluation score: |ground_truth - predicted| / ground_truth

    Uses pure relative error metric. Lower is better (0.0 = perfect prediction).
    This is more suitable for progress estimation as it considers the magnitude
    of the true value.

    Args:
        predicted: Predicted progress score (0-1) or None if parsing failed
        ground_truth: Ground truth progress score (0-1)

    Returns:
        Relative error (0.0 = perfect, higher = worse), or inf if predicted is None or ground_truth is 0
    """
    if predicted is None:
        return float('inf')

    # Avoid division by zero
    if ground_truth == 0.0:
        # If ground_truth is 0, only perfect prediction gets 0.0
        return 0.0 if predicted == 0.0 else float('inf')

    relative_error = abs(ground_truth - predicted) / ground_truth
    return relative_error


def calculate_ref_error(predicted_ref: Optional[int], ground_truth_ref: int) -> float:
    """
    Calculate reference index error: |ground_truth_ref - predicted_ref|

    Measures absolute difference between predicted and ground truth step indices.
    Lower is better (0.0 = perfect match).

    Args:
        predicted_ref: Predicted reference step index (1-based) or None if parsing failed
        ground_truth_ref: Ground truth closest step index (1-based)

    Returns:
        Absolute error (0.0 = perfect, higher = worse), or inf if predicted_ref is None or not an integer
    """
    if predicted_ref is None:
        return float('inf')

    # Ensure predicted_ref is an integer
    if not isinstance(predicted_ref, int):
        return float('inf')

    absolute_error = abs(ground_truth_ref - predicted_ref)
    return float(absolute_error)


def run_text_demo_inference_single(args):
    """
    Run text demo progress estimation with single-process batch inference.
    Optimized for 72B models using model parallelism across multiple GPUs.
    """

    # Load dataset (already expanded N times)
    print(f"Loading dataset from {args.dataset_path}")
    image_root = args.image_root if hasattr(args, 'image_root') and args.image_root else None

    from io import StringIO
    if not args.verbose:
        old_stdout = sys.stdout
        sys.stdout = StringIO()

    data = load_text_demo_dataset(
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
    print(f"TEXT DEMO INFERENCE - SINGLE PROCESS MODE (72B Optimized)")
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
        system_prompt=TEXT_DEMO_SYSTEM_PROMPT,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        verbose=args.verbose
    )
    print("Model loaded successfully!\n")

    # Process data in batches
    batch_size = args.batch_size
    results = []

    # Statistics tracking
    total_score_sum = 0.0
    total_ref_error_sum = 0.0
    valid_count = 0
    error_count = 0

    # Progress bar
    pbar = tqdm(total=len(data), desc="Processing", ncols=140,
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
                # Validate image paths
                is_valid, error_msg = validate_image_path(item)
                if not is_valid:
                    # Skip this item, record error
                    ground_truth_score_str = f"{int(item['progress_score'] * 100)}%"
                    result = {
                        "ref": None,
                        "score": None,
                        "closest_idx": item['closest_idx'],
                        "ground_truth_score": ground_truth_score_str,
                        "ref_score": float('inf'),
                        "pred_score": float('inf'),
                        "response": f"Validation error: {error_msg}",
                        "meta_data": {
                            "id": item['id'],
                            "task_goal": item.get('task_goal', ''),
                            "stage_to_estimate": item.get('stage_to_estimate', ''),
                            "status": "failed"
                        }
                    }
                    results.append(result)
                    error_count += 1
                    pbar.update(1)
                    continue

                messages = build_text_demo_prompt_from_item(
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
                    parsed = parse_text_demo_response(response)
                    predicted_score = parsed['score']
                    predicted_ref = parsed['ref']
                    has_error = parsed['parse_error']

                    # Calculate evaluation score for progress
                    evaluation_score = calculate_evaluation_score(
                        predicted_score,
                        item['progress_score']
                    )

                    # Calculate reference index error
                    ref_error = calculate_ref_error(
                        predicted_ref,
                        item['closest_idx']
                    )

                    # Update statistics
                    if not has_error:
                        if evaluation_score != float('inf'):
                            total_score_sum += evaluation_score
                        if ref_error != float('inf'):
                            total_ref_error_sum += ref_error
                        valid_count += 1
                    else:
                        error_count += 1

                    # Convert scores back to percentage strings
                    predicted_score_str = f"{int(predicted_score * 100)}%" if predicted_score is not None else None
                    ground_truth_score_str = f"{int(item['progress_score'] * 100)}%"
                    predicted_ref_str = str(predicted_ref) if predicted_ref is not None else None

                    result = {
                        "ref": predicted_ref_str,
                        "score": predicted_score_str,
                        "closest_idx": item['closest_idx'],
                        "ground_truth_score": ground_truth_score_str,
                        "ref_score": evaluation_score,
                        "pred_score": ref_error,
                        "response": response,
                        "meta_data": {
                            "id": item['id'],
                            "task_goal": item.get('task_goal', ''),
                            "stage_to_estimate": item.get('stage_to_estimate', ''),
                            "status": "failed" if has_error else "success"
                        }
                    }

                    results.append(result)
                    pbar.update(1)

                    # Update progress bar stats
                    mean_score = total_score_sum / valid_count if valid_count > 0 else 0.0
                    mean_ref_error = total_ref_error_sum / valid_count if valid_count > 0 else 0.0
                    error_rate = error_count / len(results) * 100 if results else 0.0
                    pbar.set_postfix_str(f"MeanScore={mean_score:.3f}, MeanRef={mean_ref_error:.2f}, ErrorRate={error_rate:.1f}%")

                except Exception as e:
                    # Parse error for this specific item
                    ground_truth_score_str = f"{int(item['progress_score'] * 100)}%"
                    result = {
                        "ref": None,
                        "score": None,
                        "closest_idx": item['closest_idx'],
                        "ground_truth_score": ground_truth_score_str,
                        "ref_score": float('inf'),
                        "pred_score": float('inf'),
                        "response": f"Processing error: {str(e)}\nResponse: {response if 'response' in locals() else ''}",
                        "meta_data": {
                            "id": item['id'],
                            "task_goal": item.get('task_goal', ''),
                            "stage_to_estimate": item.get('stage_to_estimate', ''),
                            "status": "failed"
                        }
                    }
                    results.append(result)
                    error_count += 1
                    pbar.update(1)

        except Exception as e:
            # Batch error - mark all items in batch as errors
            for item in batch_items:
                ground_truth_score_str = f"{int(item.get('progress_score', 0.0) * 100)}%"
                result = {
                    "ref": None,
                    "score": None,
                    "closest_idx": item.get('closest_idx', 0),
                    "ground_truth_score": ground_truth_score_str,
                    "ref_score": float('inf'),
                    "pred_score": float('inf'),
                    "response": f"Batch error: {str(e)}",
                    "meta_data": {
                        "id": item['id'],
                        "task_goal": item.get('task_goal', ''),
                        "stage_to_estimate": item.get('stage_to_estimate', ''),
                        "status": "failed"
                    }
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
    finite_scores_all = [r['ref_score'] for r in results if r['ref_score'] != float('inf')]
    finite_scores_valid = [r['ref_score'] for r in valid_results if r['ref_score'] != float('inf')]
    finite_ref_errors_all = [r['pred_score'] for r in results if r['pred_score'] != float('inf')]
    finite_ref_errors_valid = [r['pred_score'] for r in valid_results if r['pred_score'] != float('inf')]

    mean_score = sum(finite_scores_all) / len(finite_scores_all) if finite_scores_all else float('inf')
    mean_score_valid = sum(finite_scores_valid) / len(finite_scores_valid) if finite_scores_valid else float('inf')
    mean_ref_error = sum(finite_ref_errors_all) / len(finite_ref_errors_all) if finite_ref_errors_all else float('inf')
    mean_ref_error_valid = sum(finite_ref_errors_valid) / len(finite_ref_errors_valid) if finite_ref_errors_valid else float('inf')
    error_rate = error_count / len(results) if results else 0.0

    # Print final summary
    print("\n" + "=" * 70)
    print("TEXT DEMO PROGRESS ESTIMATION SUMMARY")
    print("=" * 70)
    print(f"Total samples (expanded): {len(data)}")
    print(f"Original samples: {len(data) // args.num_inferences}")
    print(f"Inferences per sample: {args.num_inferences}")
    print(f"Processed: {len(results)}")
    print(f"Errors: {error_count} ({error_rate*100:.2f}%)")
    print(f"Mean evaluation score (all): {mean_score:.4f}")
    print(f"Mean evaluation score (valid only): {mean_score_valid:.4f}")
    print(f"Mean ref error (all): {mean_ref_error:.4f}")
    print(f"Mean ref error (valid only): {mean_ref_error_valid:.4f}")
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
        "mean_evaluation_score_all": mean_score,
        "mean_evaluation_score_valid": mean_score_valid,
        "mean_ref_error_all": mean_ref_error,
        "mean_ref_error_valid": mean_ref_error_valid,
        "batch_size": args.batch_size,
        "num_gpus": num_gpus,
        "dataset_path": args.dataset_path,
        "model_path": args.model_path,
        "output_file": args.output_file
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Text Demo Progress Estimation - Single Process (72B Optimized)"
    )

    # Required arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the Qwen2-VL model")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the Text Demo dataset (JSONL format)")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output JSONL file path for results")

    # Optional arguments
    parser.add_argument("--image-root", type=str, default=None,
                        help="Root directory to prepend to relative image paths")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference (default: 1 for 72B models)")
    parser.add_argument("--num-inferences", type=int, default=4,
                        help="Number of inferences per sample (data expansion factor, default: 4)")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit number of samples to process after expansion (-1 for all)")

    # Model parameters
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7 for diversity)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling parameter (default: 0.9)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling parameter (default: 50)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum number of tokens to generate (default: 512)")

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

    # Run inference
    run_text_demo_inference_single(args)


if __name__ == "__main__":
    main()
