import os
import sys
import json
import argparse
import time
import re
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union
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

        # Extract ref (now expects integer 1-based index or "n/a")
        ref_match = re.search(r'<ref>(.*?)</ref>', response, re.DOTALL)
        if ref_match:
            ref_str = ref_match.group(1).strip()
            # Check if it's "n/a" (case-insensitive)
            if ref_str.lower() in ['n/a', 'na']:
                result['ref'] = "n/a"
            else:
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

        # Extract score (supports "33%", "0.33", or "n/a")
        score_match = re.search(r'<score>(.*?)</score>', response, re.DOTALL)
        if score_match:
            score_str = score_match.group(1).strip()
            # Check if it's "n/a" (case-insensitive)
            if score_str.lower() in ['n/a', 'na']:
                result['score'] = "n/a"
            else:
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


def calculate_evaluation_score(predicted: Union[float, str, None], ground_truth: Union[float, str]) -> float:
    """
    Calculate evaluation score: |ground_truth - predicted| / ground_truth

    Uses pure relative error metric. Lower is better (0.0 = perfect prediction).
    For "n/a" ground truth, checks if predicted is also "n/a" (Pass/Fail).

    Args:
        predicted: Predicted progress score (0-1), "n/a", or None if parsing failed
        ground_truth: Ground truth progress score (0-1) or "n/a"

    Returns:
        Relative error (0.0 = perfect, higher = worse), or inf if prediction failed
    """
    # Handle "n/a" ground truth: only "n/a" prediction is correct
    if isinstance(ground_truth, str) and ground_truth.lower() == "n/a":
        if isinstance(predicted, str) and predicted.lower() == "n/a":
            return 0.0  # Perfect match: both "n/a"
        else:
            return float('inf')  # Wrong: should predict "n/a" but didn't

    # Handle "n/a" prediction with numeric ground truth
    if isinstance(predicted, str) and predicted.lower() == "n/a":
        return float('inf')  # Wrong: predicted "n/a" but ground truth is numeric

    if predicted is None:
        return float('inf')

    # Avoid division by zero
    if ground_truth == 0.0:
        # If ground_truth is 0, only perfect prediction gets 0.0
        return 0.0 if predicted == 0.0 else float('inf')

    relative_error = abs(ground_truth - predicted) / ground_truth
    return relative_error


def calculate_ref_error(predicted_ref: Union[int, str, None], ground_truth_ref: Union[int, str]) -> float:
    """
    Calculate reference index error: |ground_truth_ref - predicted_ref|

    Measures absolute difference between predicted and ground truth step indices.
    For "n/a" ground truth, checks if predicted is also "n/a" (Pass/Fail).

    Args:
        predicted_ref: Predicted reference step index (1-based), "n/a", or None if parsing failed
        ground_truth_ref: Ground truth closest step index (1-based) or "n/a"

    Returns:
        Absolute error (0.0 = perfect, higher = worse), or inf if prediction failed
    """
    # Handle "n/a" ground truth: only "n/a" prediction is correct
    if isinstance(ground_truth_ref, str) and ground_truth_ref.lower() == "n/a":
        if isinstance(predicted_ref, str) and predicted_ref.lower() == "n/a":
            return 0.0  # Perfect match: both "n/a"
        else:
            return float('inf')  # Wrong: should predict "n/a" but didn't

    # Handle "n/a" prediction with numeric ground truth
    if isinstance(predicted_ref, str) and predicted_ref.lower() == "n/a":
        return float('inf')  # Wrong: predicted "n/a" but ground truth is numeric

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
    Uses FRM's cheat prompt system with ground-truth.
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
    print(f"Using FRM cheat prompt: True")
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

    # n/a sample statistics
    na_total = 0  # Total "n/a" samples
    na_pass = 0   # "n/a" samples where model correctly predicted "n/a"

    # Numeric sample statistics
    numeric_total = 0
    numeric_score_sum = 0.0
    numeric_ref_error_sum = 0.0
    numeric_valid = 0

    # Progress bar
    pbar = tqdm(total=len(data), desc="Processing", ncols=160,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    i = 0
    batch_counter = 0  # Track number of batches processed
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
                    # Format ground truth score string (handle "n/a")
                    if isinstance(item['progress_score'], str) and item['progress_score'].lower() == "n/a":
                        ground_truth_score_str = "n/a"
                    else:
                        ground_truth_score_str = f"{int(item['progress_score'] * 100)}%"

                    # Format closest_idx (handle "n/a")
                    closest_idx_str = str(item['closest_idx']) if isinstance(item['closest_idx'], int) else item['closest_idx']

                    result = {
                        "ref": None,
                        "score": None,
                        "closest_idx": closest_idx_str,
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
                    max_pixels=args.max_pixels,
                    use_ground_truth=True  # FRM cheat mode
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

                    # Determine if this is an "n/a" sample
                    is_na_sample = isinstance(item['progress_score'], str) and item['progress_score'].lower() == "n/a"

                    # Update statistics
                    if not has_error:
                        valid_count += 1

                        if is_na_sample:
                            na_total += 1
                            if evaluation_score == 0.0:  # Both ground truth and prediction are "n/a"
                                na_pass += 1
                        else:
                            numeric_total += 1
                            if evaluation_score != float('inf'):
                                numeric_score_sum += evaluation_score
                            if ref_error != float('inf'):
                                numeric_ref_error_sum += ref_error
                            numeric_valid += 1

                        # Overall statistics (legacy)
                        if evaluation_score != float('inf'):
                            total_score_sum += evaluation_score
                        if ref_error != float('inf'):
                            total_ref_error_sum += ref_error
                    else:
                        error_count += 1

                    # Convert scores back to percentage strings (handle "n/a")
                    if isinstance(predicted_score, str) and predicted_score.lower() == "n/a":
                        predicted_score_str = "n/a"
                    elif predicted_score is not None:
                        predicted_score_str = f"{int(predicted_score * 100)}%"
                    else:
                        predicted_score_str = None

                    if isinstance(item['progress_score'], str) and item['progress_score'].lower() == "n/a":
                        ground_truth_score_str = "n/a"
                    else:
                        ground_truth_score_str = f"{int(item['progress_score'] * 100)}%"

                    # Format predicted_ref and closest_idx (handle "n/a")
                    if isinstance(predicted_ref, str) and predicted_ref.lower() == "n/a":
                        predicted_ref_str = "n/a"
                    elif predicted_ref is not None:
                        predicted_ref_str = str(predicted_ref)
                    else:
                        predicted_ref_str = None

                    closest_idx_str = str(item['closest_idx']) if isinstance(item['closest_idx'], int) else item['closest_idx']

                    result = {
                        "ref": predicted_ref_str,
                        "score": predicted_score_str,
                        "closest_idx": closest_idx_str,
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
                    postfix_parts = []

                    # Numeric sample statistics
                    if numeric_total > 0:
                        mean_score = numeric_score_sum / numeric_valid if numeric_valid > 0 else 0.0
                        mean_ref_error = numeric_ref_error_sum / numeric_valid if numeric_valid > 0 else 0.0
                        postfix_parts.append(f"score_err={mean_score:.3f}, ref_err={mean_ref_error:.2f}")

                    # n/a sample statistics (Pass Rate)
                    if na_total > 0:
                        na_pass_rate = na_pass / na_total * 100
                        postfix_parts.append(f"na_pass={na_pass_rate:.1f}%")

                    # Error rate
                    error_rate_pct = error_count / len(results) * 100 if results else 0.0
                    postfix_parts.append(f"err={error_rate_pct:.1f}%")

                    pbar.set_postfix_str(", ".join(postfix_parts))

                except Exception as e:
                    # Parse error for this specific item
                    if isinstance(item['progress_score'], str) and item['progress_score'].lower() == "n/a":
                        ground_truth_score_str = "n/a"
                    else:
                        ground_truth_score_str = f"{int(item['progress_score'] * 100)}%"

                    closest_idx_str = str(item['closest_idx']) if isinstance(item['closest_idx'], int) else item['closest_idx']

                    result = {
                        "ref": None,
                        "score": None,
                        "closest_idx": closest_idx_str,
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
                progress_score = item.get('progress_score', 0.0)
                if isinstance(progress_score, str) and progress_score.lower() == "n/a":
                    ground_truth_score_str = "n/a"
                else:
                    ground_truth_score_str = f"{int(progress_score * 100)}%"

                closest_idx = item.get('closest_idx', 0)
                closest_idx_str = str(closest_idx) if isinstance(closest_idx, int) else closest_idx

                result = {
                    "ref": None,
                    "score": None,
                    "closest_idx": closest_idx_str,
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

        # Increment batch counter
        batch_counter += 1

        # Save results after every batch
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

    # Calculate n/a pass rate
    na_pass_rate = na_pass / na_total if na_total > 0 else 0.0

    # Calculate numeric sample statistics
    numeric_mean_score = numeric_score_sum / numeric_valid if numeric_valid > 0 else float('inf')
    numeric_mean_ref_error = numeric_ref_error_sum / numeric_valid if numeric_valid > 0 else float('inf')

    # Print final summary
    print("\n" + "=" * 70)
    print("TEXT DEMO PROGRESS ESTIMATION SUMMARY (72B + FRM Cheat Prompt)")
    print("=" * 70)
    print(f"Total samples (expanded): {len(data)}")
    print(f"Original samples: {len(data) // args.num_inferences}")
    print(f"Inferences per sample: {args.num_inferences}")
    print(f"Processed: {len(results)}")
    print(f"Errors: {error_count} ({error_rate*100:.2f}%)")
    print()
    print("Numeric Samples:")
    print(f"  Total: {numeric_total}")
    print(f"  Mean evaluation score: {numeric_mean_score:.4f}")
    print(f"  Mean ref error: {numeric_mean_ref_error:.4f}")
    print()
    print("N/A Samples:")
    print(f"  Total: {na_total}")
    print(f"  Pass (correctly predicted n/a): {na_pass}")
    print(f"  Fail (incorrectly predicted): {na_total - na_pass}")
    print(f"  Pass Rate: {na_pass_rate*100:.2f}%")
    print()
    print(f"Overall (legacy):")
    print(f"  Mean evaluation score (all): {mean_score:.4f}")
    print(f"  Mean evaluation score (valid only): {mean_score_valid:.4f}")
    print(f"  Mean ref error (all): {mean_ref_error:.4f}")
    print(f"  Mean ref error (valid only): {mean_ref_error_valid:.4f}")
    print()
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

        "numeric_samples": {
            "total": numeric_total,
            "valid": numeric_valid,
            "mean_evaluation_score": numeric_mean_score,
            "mean_ref_error": numeric_mean_ref_error
        },

        "na_samples": {
            "total": na_total,
            "pass": na_pass,
            "fail": na_total - na_pass,
            "pass_rate": na_pass_rate
        },

        "overall_legacy": {
            "mean_evaluation_score_all": mean_score,
            "mean_evaluation_score_valid": mean_score_valid,
            "mean_ref_error_all": mean_ref_error,
            "mean_ref_error_valid": mean_ref_error_valid
        },

        "config": {
            "batch_size": args.batch_size,
            "num_gpus": num_gpus,
            "mode": "single_process_72b_frm_cheat",
            "dataset_path": args.dataset_path,
            "model_path": args.model_path,
            "output_file": args.output_file
        }
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Text Demo Progress Estimation - Single Process (72B Optimized, FRM Cheat Prompt)"
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
