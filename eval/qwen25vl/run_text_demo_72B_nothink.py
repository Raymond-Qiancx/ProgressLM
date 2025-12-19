import os
import sys
import json
import argparse
import time
import re
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from text_demo_dataset import load_text_demo_dataset, validate_image_path
from text_demo_prompt_nothink import build_text_demo_prompt_from_item, TEXT_DEMO_SYSTEM_PROMPT
from qwen2_vl.model import Qwen2VLChat


def parse_text_demo_response(response: str) -> Dict[str, Any]:
    """
    Parse the model's response to extract the progress score directly.

    This version expects a simple numeric output without XML tags,
    as the prompt asks: "Your answer only needs to output the final progress score you estimated."

    Supports formats like:
    - "33%" or "33"
    - "0.33"
    - "The progress is 33%"

    Args:
        response: Model output string

    Returns:
        Dictionary with parsed fields:
        {
            'score': float or None (in 0.0-1.0 range),
            'parse_error': bool
        }
    """
    result = {
        'score': None,
        'parse_error': False
    }

    try:
        # Check for "n/a" first
        response_lower = response.lower().strip()
        if response_lower in ["n/a", "na"] or "n/a" in response_lower or "na" in response_lower:
            result['score'] = "n/a"
            return result

        # Try to extract any percentage or decimal number from the response
        # Look for patterns like "33%", "0.33", "33", etc.

        # First, try to find percentage format (e.g., "33%")
        percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', response)
        if percent_match:
            score_value = float(percent_match.group(1)) / 100.0
            result['score'] = max(0.0, min(1.0, score_value))
            return result

        # Then, try to find decimal format (e.g., "0.33" or just "33")
        # Look for numbers that might be standalone or after common phrases
        number_match = re.search(r'(?:progress|score|estimate)[:\s]+(\d+(?:\.\d+)?)|^(\d+(?:\.\d+)?)$|(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        if number_match:
            # Get the first non-None group
            score_str = number_match.group(1) or number_match.group(2) or number_match.group(3)
            score_value = float(score_str)

            # If > 1.0, assume it's percentage without % sign
            if score_value > 1.0:
                score_value = score_value / 100.0

            # Clamp to [0, 1]
            result['score'] = max(0.0, min(1.0, score_value))
            return result

        # If no number found, mark as parse error
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


def calculate_score_false_positive(predicted_score, gt_score) -> bool:
    """
    Calculate false positive for score prediction.

    False positive occurs when:
    - GT is numeric but prediction is "n/a"
    - GT is "n/a" but prediction is numeric

    Args:
        predicted_score: Predicted score (float, "n/a", or invalid)
        gt_score: Ground truth score (float or None)

    Returns:
        is_score_false_positive (bool)
    """
    gt_score_is_na = (gt_score is None)
    pred_score_is_na = (predicted_score == "n/a" or predicted_score is None or not isinstance(predicted_score, (int, float)))

    # False positive: mismatch between GT and prediction
    score_fp = gt_score_is_na != pred_score_is_na

    return score_fp


def calculate_voc_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate VOC (trajectory order consistency) using Spearman correlation.

    Process:
    1. Group samples by complete trajectory ID
    2. Filter: only keep trajectories where GT closest_idx and progress_score are both numeric
    3. For each trajectory:
       - Extract GT scores and predicted scores
       - Calculate Spearman correlation directly between scores
       - scipy handles ties by assigning average ranks to equal values
       - Single-sample trajectories → VOC = None
    4. Return mean and std of all valid VOCs

    Args:
        results: List of result dictionaries with meta_data containing id, closest_idx, progress_score

    Returns:
        Dictionary with VOC statistics:
        {
            'voc_mean': float or None,
            'voc_std': float or None,
            'voc_count': int,  # number of trajectories with VOC
            'voc_values': List[float]  # individual VOC values
        }
    """
    from collections import defaultdict

    # Group by trajectory ID
    trajectories = defaultdict(list)
    for res in results:
        meta = res.get('meta_data', {})
        traj_id = meta.get('id', '')

        # Only include if GT has numeric values
        gt_ref = meta.get('closest_idx')
        gt_score = meta.get('progress_score')

        if gt_ref is not None and gt_score is not None:
            # GT is numeric
            pred_score = res.get('score')
            # Convert n/a to 0.0 for ranking
            if pred_score == "n/a" or pred_score is None:
                pred_score_numeric = 0.0
            else:
                pred_score_numeric = float(pred_score) if isinstance(pred_score, (int, float)) else 0.0

            trajectories[traj_id].append({
                'gt_score': gt_score,
                'pred_score': pred_score_numeric
            })

    # Calculate VOC for each trajectory
    voc_values = []
    for traj_id, samples in trajectories.items():
        if len(samples) <= 1:
            # Cannot calculate correlation for single sample
            continue

        # Extract scores directly
        gt_scores = [s['gt_score'] for s in samples]
        pred_scores = [s['pred_score'] for s in samples]

        # Calculate Spearman correlation directly on scores
        # scipy.stats.spearmanr handles ties by assigning average ranks
        if len(set(gt_scores)) > 1:  # Need variation in GT scores
            correlation, _ = spearmanr(gt_scores, pred_scores)
            if not np.isnan(correlation):
                voc_values.append(correlation)

    # Calculate statistics
    if len(voc_values) > 0:
        return {
            'voc_mean': float(np.mean(voc_values)),
            'voc_std': float(np.std(voc_values)),
            'voc_count': len(voc_values),
            'voc_values': voc_values
        }
    else:
        return {
            'voc_mean': None,
            'voc_std': None,
            'voc_count': 0,
            'voc_values': []
        }


def run_text_demo_inference_single(args):
    """
    Run text demo progress estimation with single-process batch inference.
    Optimized for 72B models using model parallelism across multiple GPUs.

    This "no-think" version uses a simplified prompt that asks for direct score output
    without intermediate reasoning steps.
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
    print(f"TEXT DEMO INFERENCE - 72B NO-THINK MODE")
    print(f"{'='*70}")
    print(f"GPUs available: {num_gpus} ({gpu_ids})")
    print(f"Model parallelism: {'ENABLED' if num_gpus > 1 else 'DISABLED'}")
    print(f"Total samples (expanded): {len(data)}")
    print(f"Original samples: {len(data) // args.num_inferences}")
    print(f"Inferences per sample: {args.num_inferences}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mode: Direct score output (no reasoning steps)")
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
    valid_count = 0
    error_count = 0
    score_fp_count = 0  # Count of score false positives

    # Progress bar
    pbar = tqdm(total=len(data), desc="Processing", ncols=160,
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
                    gt_score = item.get('progress_score')
                    if gt_score is not None:
                        ground_truth_score_str = f"{int(gt_score * 100)}%"
                    else:
                        ground_truth_score_str = "n/a"

                    result = {
                        "predicted_score": None,
                        "ground_truth_score": ground_truth_score_str,
                        "evaluation_score": float('inf'),
                        "score_false_positive": False,
                        "response": f"Validation error: {error_msg}",
                        "meta_data": {
                            **item,  # Include all original data
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
                    has_error = parsed['parse_error']

                    # Get ground truth value
                    gt_score = item['progress_score']  # Can be float or None

                    # Calculate false positive
                    score_fp = calculate_score_false_positive(predicted_score, gt_score)
                    if score_fp:
                        score_fp_count += 1

                    # Calculate evaluation score for progress (only for numeric pairs)
                    if gt_score is not None and isinstance(predicted_score, (int, float)):
                        evaluation_score = calculate_evaluation_score(predicted_score, gt_score)
                    else:
                        evaluation_score = float('inf')

                    # Update statistics
                    if not has_error:
                        if evaluation_score != float('inf'):
                            total_score_sum += evaluation_score
                        valid_count += 1
                    else:
                        error_count += 1

                    # Convert scores back to strings for output
                    if predicted_score == "n/a":
                        predicted_score_str = "n/a"
                    elif isinstance(predicted_score, (int, float)):
                        predicted_score_str = f"{int(predicted_score * 100)}%"
                    else:
                        predicted_score_str = None

                    if gt_score is not None:
                        ground_truth_score_str = f"{int(gt_score * 100)}%"
                    else:
                        ground_truth_score_str = "n/a"

                    result = {
                        "predicted_score": predicted_score_str,
                        "ground_truth_score": ground_truth_score_str,
                        "evaluation_score": evaluation_score,
                        "score_false_positive": score_fp,
                        "response": response,
                        "meta_data": {
                            **item,  # Include all original data
                            "status": "failed" if has_error else "success"
                        }
                    }

                    results.append(result)
                    pbar.update(1)

                    # Update progress bar stats
                    mean_score = total_score_sum / valid_count if valid_count > 0 else 0.0
                    error_rate = error_count / len(results) * 100 if results else 0.0
                    score_fp_rate = score_fp_count / len(results) * 100 if results else 0.0
                    pbar.set_postfix_str(f"MeanErr={mean_score:.3f}, Err={error_rate:.1f}%, FP={score_fp_rate:.1f}%")

                except Exception as e:
                    # Parse error for this specific item
                    gt_score = item.get('progress_score')
                    ground_truth_score_str = f"{int(gt_score * 100)}%" if gt_score is not None else "n/a"

                    result = {
                        "predicted_score": None,
                        "ground_truth_score": ground_truth_score_str,
                        "evaluation_score": float('inf'),
                        "score_false_positive": False,
                        "response": f"Processing error: {str(e)}\nResponse: {response if 'response' in locals() else ''}",
                        "meta_data": {
                            **item,  # Include all original data
                            "status": "failed"
                        }
                    }
                    results.append(result)
                    error_count += 1
                    pbar.update(1)

        except Exception as e:
            # Batch error - mark all items in batch as errors
            for item in batch_items:
                gt_score = item.get('progress_score')
                ground_truth_score_str = f"{int(gt_score * 100)}%" if gt_score is not None else "n/a"

                result = {
                    "predicted_score": None,
                    "ground_truth_score": ground_truth_score_str,
                    "evaluation_score": float('inf'),
                    "score_false_positive": False,
                    "response": f"Batch error: {str(e)}",
                    "meta_data": {
                        **item,  # Include all original data
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

    # Filter out inf values for mean calculation
    finite_scores_all = [r['evaluation_score'] for r in results if r.get('evaluation_score') != float('inf')]
    finite_scores_valid = [r['evaluation_score'] for r in valid_results if r.get('evaluation_score') != float('inf')]

    mean_score = sum(finite_scores_all) / len(finite_scores_all) if finite_scores_all else float('inf')
    mean_score_valid = sum(finite_scores_valid) / len(finite_scores_valid) if finite_scores_valid else float('inf')
    error_rate = error_count / len(results) if results else 0.0

    # Calculate false positive rate
    score_fp_total = sum(1 for r in results if r.get('score_false_positive', False))
    score_fp_rate = score_fp_total / len(results) if results else 0.0

    # Calculate VOC metrics
    print("\nCalculating VOC (trajectory order consistency) metrics...")
    voc_metrics = calculate_voc_metrics(results)

    # Count GT type distribution
    gt_numeric_count = sum(1 for r in results if r['meta_data'].get('progress_score') is not None)
    gt_na_count = len(results) - gt_numeric_count

    # Print final summary
    print("\n" + "=" * 70)
    print("TEXT DEMO PROGRESS ESTIMATION SUMMARY (NO-THINK MODE)")
    print("=" * 70)
    print(f"Total samples (expanded): {len(data)}")
    print(f"Original samples: {len(data) // args.num_inferences}")
    print(f"Inferences per sample: {args.num_inferences}")
    print(f"Processed: {len(results)}")
    print(f"Errors: {error_count} ({error_rate*100:.2f}%)")
    print(f"\nError Metrics:")
    print(f"  Mean evaluation score (all): {mean_score:.4f}")
    print(f"  Mean evaluation score (valid only): {mean_score_valid:.4f}")
    print(f"\nFalse Positive Rate:")
    print(f"  Score false positive rate: {score_fp_rate*100:.2f}% ({score_fp_total}/{len(results)})")
    print(f"\nVOC (Trajectory Order Consistency):")
    if voc_metrics['voc_mean'] is not None:
        print(f"  Mean VOC: {voc_metrics['voc_mean']:.4f}")
        print(f"  Std VOC: {voc_metrics['voc_std']:.4f}")
        print(f"  Trajectories evaluated: {voc_metrics['voc_count']}")
    else:
        print(f"  VOC: N/A (no valid trajectories)")
    print(f"\nGT Distribution:")
    print(f"  Numeric GT: {gt_numeric_count} ({gt_numeric_count/len(results)*100:.1f}%)")
    print(f"  N/A GT: {gt_na_count} ({gt_na_count/len(results)*100:.1f}%)")
    print(f"\nResults saved to: {args.output_file}")
    print("=" * 70)

    # Save summary
    summary_file = args.output_file.replace('.jsonl', '_summary.json')
    summary = {
        "mode": "no-think (direct score output)",
        "total_samples_expanded": len(data),
        "original_samples": len(data) // args.num_inferences,
        "num_inferences_per_sample": args.num_inferences,
        "processed": len(results),
        "errors": error_count,
        "error_rate": error_rate,
        "mean_evaluation_score_all": mean_score,
        "mean_evaluation_score_valid": mean_score_valid,
        "score_false_positive_count": score_fp_total,
        "score_false_positive_rate": score_fp_rate,
        "voc_mean": voc_metrics['voc_mean'],
        "voc_std": voc_metrics['voc_std'],
        "voc_trajectories_count": voc_metrics['voc_count'],
        "gt_numeric_count": gt_numeric_count,
        "gt_na_count": gt_na_count,
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
        description="Text Demo Progress Estimation - 72B No-Think Mode (Direct Score Output)"
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
