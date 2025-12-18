#!/usr/bin/env python3
"""
GPT-5 Visual Demo Progress Evaluation

Evaluates GPT-5 model on visual-based demonstration progress estimation task.
Uses ThreadPoolExecutor for concurrent API calls.
"""

import os
import sys
import json
import argparse
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from tqdm import tqdm

# Local imports
from visual_demo_dataset import load_visual_demo_dataset, validate_image_paths
from visual_demo_prompt import build_visual_demo_prompt_from_item, VISUAL_DEMO_SYSTEM_PROMPT
from openai_client import OpenAIVisionClient
from common_utils import (
    parse_response,
    calculate_evaluation_score,
    calculate_ref_error,
    calculate_false_positives,
    calculate_voc_metrics,
    format_score_string,
    format_ref_string,
    get_sample_unique_id
)

# Global lock for file writing
write_lock = threading.Lock()


def load_processed_ids(output_file: Path) -> set:
    """
    Load already processed sample IDs from output file.

    Args:
        output_file: Path to output JSONL file

    Returns:
        Set of processed unique IDs
    """
    processed_ids = set()

    if not output_file.exists():
        return processed_ids

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    if 'meta_data' in result:
                        sample_id = result['meta_data'].get('id', 'unknown')
                        progress_score = result.get('ground_truth_score', 'unknown')
                        unique_id = f"{sample_id}_{progress_score}"
                        processed_ids.add(unique_id)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Warning: Error reading processed file: {str(e)}")

    return processed_ids


def save_result(result: Dict, output_file: Path):
    """
    Save a single result to JSONL file (thread-safe).

    Args:
        result: Result dictionary
        output_file: Output file path
    """
    with write_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def process_single_sample(
    sample: Dict,
    client: OpenAIVisionClient,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process a single sample through the GPT-5 API.

    Args:
        sample: Dataset sample
        client: OpenAI Vision client
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing

    Returns:
        Result dictionary
    """
    try:
        # Validate image paths
        is_valid, error_msg = validate_image_paths(sample)
        if not is_valid:
            gt_score = sample.get('progress_score')
            gt_ref = sample.get('closest_idx')
            return {
                "ref": None,
                "score": None,
                "closest_idx": str(gt_ref) if gt_ref is not None else "n/a",
                "ground_truth_score": format_score_string(gt_score) if gt_score is not None else "n/a",
                "score_error": float('inf'),
                "ref_error": float('inf'),
                "ref_false_positive": False,
                "score_false_positive": False,
                "response": f"Validation error: {error_msg}",
                "meta_data": {
                    **sample,
                    "status": "failed",
                    "timestamp": datetime.now().isoformat()
                }
            }

        # Build prompt
        messages = build_visual_demo_prompt_from_item(
            sample,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )

        # Call API
        api_result = client.generate_with_retry(messages)

        if api_result["status"] == "error":
            gt_score = sample.get('progress_score')
            gt_ref = sample.get('closest_idx')
            return {
                "ref": None,
                "score": None,
                "closest_idx": str(gt_ref) if gt_ref is not None else "n/a",
                "ground_truth_score": format_score_string(gt_score) if gt_score is not None else "n/a",
                "score_error": float('inf'),
                "ref_error": float('inf'),
                "ref_false_positive": False,
                "score_false_positive": False,
                "response": f"API error: {api_result['error']}",
                "meta_data": {
                    **sample,
                    "status": "failed",
                    "error": api_result["error"],
                    "timestamp": datetime.now().isoformat()
                }
            }

        # Parse response
        response = api_result["response"]
        parsed = parse_response(response)
        predicted_score = parsed['score']
        predicted_ref = parsed['ref']
        has_error = parsed['parse_error']

        # Get ground truth values
        gt_score = sample['progress_score']
        gt_ref = sample['closest_idx']

        # Calculate false positives
        ref_fp, score_fp = calculate_false_positives(
            predicted_ref, predicted_score, gt_ref, gt_score
        )

        # Calculate evaluation score (only for numeric pairs)
        if gt_score is not None and isinstance(predicted_score, (int, float)):
            evaluation_score = calculate_evaluation_score(predicted_score, gt_score)
        else:
            evaluation_score = float('inf')

        # Calculate reference index error (only for numeric pairs)
        if gt_ref is not None and isinstance(predicted_ref, int):
            ref_error = calculate_ref_error(predicted_ref, gt_ref)
        else:
            ref_error = float('inf')

        # Format output strings
        predicted_score_str = format_score_string(predicted_score)
        predicted_ref_str = format_ref_string(predicted_ref)
        ground_truth_score_str = format_score_string(gt_score) if gt_score is not None else "n/a"

        return {
            "ref": predicted_ref_str,
            "score": predicted_score_str,
            "closest_idx": str(gt_ref) if gt_ref is not None else "n/a",
            "ground_truth_score": ground_truth_score_str,
            "score_error": evaluation_score,  # score 的相对误差
            "ref_error": ref_error,           # ref 的绝对误差
            "ref_false_positive": ref_fp,
            "score_false_positive": score_fp,
            "response": response,
            "meta_data": {
                **sample,
                "status": "failed" if has_error else "success",
                "tokens_used": api_result["tokens_used"],
                "model": client.model,
                "timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        gt_score = sample.get('progress_score')
        gt_ref = sample.get('closest_idx')
        return {
            "ref": None,
            "score": None,
            "closest_idx": str(gt_ref) if gt_ref is not None else "n/a",
            "ground_truth_score": format_score_string(gt_score) if gt_score is not None else "n/a",
            "score_error": float('inf'),
            "ref_error": float('inf'),
            "ref_false_positive": False,
            "score_false_positive": False,
            "response": f"Processing error: {str(e)}\n{traceback.format_exc()}",
            "meta_data": {
                **sample,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        }


def run_visual_demo_evaluation(args):
    """Run visual demo progress evaluation with GPT-5 API."""

    # Load dataset
    print(f"Loading dataset from {args.input}")
    image_root = args.image_dir if args.image_dir else None

    data = load_visual_demo_dataset(
        args.input,
        num_inferences=args.num_inferences,
        image_root=image_root
    )

    print(f"Total samples (expanded): {len(data)}")
    print(f"Original samples: {len(data) // args.num_inferences}")
    print(f"Inferences per sample: {args.num_inferences}")

    # Handle resume mode
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_ids = set()
    if args.resume:
        processed_ids = load_processed_ids(output_path)
        if processed_ids:
            print(f"Resume mode: Found {len(processed_ids)} already processed samples")
    else:
        # Non-resume mode: clear output file
        if output_path.exists():
            output_path.unlink()
            print("Cleared existing output file")

    # Filter out already processed samples
    samples_to_process = []
    skipped_count = 0

    for sample in data:
        unique_id = get_sample_unique_id(sample)
        if unique_id in processed_ids:
            skipped_count += 1
            continue
        samples_to_process.append(sample)

        if args.limit and len(samples_to_process) >= args.limit:
            break

    if skipped_count > 0:
        print(f"Skipped {skipped_count} already processed samples")

    if args.limit:
        samples_to_process = samples_to_process[:args.limit]
        print(f"Limited to {args.limit} samples")

    if not samples_to_process:
        print("No new samples to process")
        return

    print(f"Processing {len(samples_to_process)} samples with {args.max_workers} workers")

    # Initialize client
    client = OpenAIVisionClient(
        api_key=args.api_key,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature
    )

    # Statistics
    success_count = 0
    error_count = 0
    total_tokens = 0
    total_score_sum = 0.0
    total_ref_error_sum = 0.0
    valid_count = 0
    ref_fp_count = 0
    score_fp_count = 0

    # Process with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_sample = {
            executor.submit(
                process_single_sample,
                sample,
                client,
                args.min_pixels,
                args.max_pixels
            ): sample
            for sample in samples_to_process
        }

        desc = "Resume progress" if args.resume else "Processing"
        with tqdm(total=len(samples_to_process), desc=desc) as pbar:
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]

                try:
                    result = future.result(timeout=120)

                    # Save result
                    save_result(result, output_path)

                    # Update statistics
                    if result['meta_data']['status'] == 'success':
                        success_count += 1
                        total_tokens += result['meta_data'].get('tokens_used', 0)

                        # Update metrics
                        if result['score_error'] != float('inf'):
                            total_score_sum += result['score_error']
                            valid_count += 1
                        if result['ref_error'] != float('inf'):
                            total_ref_error_sum += result['ref_error']
                        if result['ref_false_positive']:
                            ref_fp_count += 1
                        if result['score_false_positive']:
                            score_fp_count += 1
                    else:
                        error_count += 1

                    # Update progress bar
                    mean_score = total_score_sum / valid_count if valid_count > 0 else 0.0
                    pbar.set_postfix({
                        'ok': success_count,
                        'err': error_count,
                        'score': f'{mean_score:.3f}',
                        'tokens': total_tokens
                    })

                except Exception as e:
                    error_count += 1
                    error_result = {
                        "ref": None,
                        "score": None,
                        "closest_idx": str(sample.get('closest_idx')) if sample.get('closest_idx') is not None else "n/a",
                        "ground_truth_score": format_score_string(sample.get('progress_score')) if sample.get('progress_score') is not None else "n/a",
                        "score_error": float('inf'),
                        "ref_error": float('inf'),
                        "ref_false_positive": False,
                        "score_false_positive": False,
                        "response": f"Execution error: {str(e)}",
                        "meta_data": {
                            **sample,
                            "status": "failed",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    save_result(error_result, output_path)

                pbar.update(1)

    # Load all results for final statistics
    print("\nLoading results for final statistics...")
    all_results = []
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))

    # Calculate final statistics
    valid_results = [r for r in all_results if r['meta_data']['status'] == 'success']
    finite_scores = [r['score_error'] for r in all_results if r.get('score_error') != float('inf')]
    finite_ref_errors = [r['ref_error'] for r in all_results if r.get('ref_error') != float('inf')]

    mean_score = sum(finite_scores) / len(finite_scores) if finite_scores else float('inf')
    mean_ref_error = sum(finite_ref_errors) / len(finite_ref_errors) if finite_ref_errors else float('inf')
    error_rate = error_count / len(samples_to_process) if samples_to_process else 0.0

    # Calculate false positive rates
    ref_fp_total = sum(1 for r in all_results if r.get('ref_false_positive', False))
    score_fp_total = sum(1 for r in all_results if r.get('score_false_positive', False))
    ref_fp_rate = ref_fp_total / len(all_results) if all_results else 0.0
    score_fp_rate = score_fp_total / len(all_results) if all_results else 0.0

    # Calculate VOC metrics
    print("Calculating VOC metrics...")
    voc_metrics = calculate_voc_metrics(all_results)

    # Count GT distribution
    gt_numeric_count = sum(1 for r in all_results if r['meta_data'].get('closest_idx') is not None and r['meta_data'].get('progress_score') is not None)
    gt_na_count = len(all_results) - gt_numeric_count

    # Print summary
    print("\n" + "=" * 70)
    print("VISUAL DEMO PROGRESS EVALUATION SUMMARY (GPT-5)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Total samples (expanded): {len(data)}")
    print(f"Processed this run: {len(samples_to_process)}")
    print(f"Total in output: {len(all_results)}")
    print(f"Errors: {error_count} ({error_rate*100:.2f}%)")
    print(f"\nError Metrics:")
    print(f"  Mean evaluation score: {mean_score:.4f}")
    print(f"  Mean ref error: {mean_ref_error:.4f}")
    print(f"\nFalse Positive Rates:")
    print(f"  Ref FP rate: {ref_fp_rate*100:.2f}% ({ref_fp_total}/{len(all_results)})")
    print(f"  Score FP rate: {score_fp_rate*100:.2f}% ({score_fp_total}/{len(all_results)})")
    print(f"\nVOC (Trajectory Order Consistency):")
    if voc_metrics['voc_mean'] is not None:
        print(f"  Mean VOC: {voc_metrics['voc_mean']:.4f}")
        print(f"  Std VOC: {voc_metrics['voc_std']:.4f}")
        print(f"  Trajectories evaluated: {voc_metrics['voc_count']}")
    else:
        print(f"  VOC: N/A (no valid trajectories)")
    print(f"\nGT Distribution:")
    print(f"  Numeric GT: {gt_numeric_count} ({gt_numeric_count/len(all_results)*100:.1f}%)")
    print(f"  N/A GT: {gt_na_count} ({gt_na_count/len(all_results)*100:.1f}%)")
    print(f"\nTotal tokens used: {total_tokens:,}")
    print(f"Results saved to: {output_path}")
    print("=" * 70)

    # Save summary
    summary_file = str(output_path).replace('.jsonl', '_summary.json')
    summary = {
        "model": args.model,
        "total_samples_expanded": len(data),
        "original_samples": len(data) // args.num_inferences,
        "num_inferences_per_sample": args.num_inferences,
        "processed_this_run": len(samples_to_process),
        "total_in_output": len(all_results),
        "errors": error_count,
        "error_rate": error_rate,
        "mean_evaluation_score": mean_score,
        "mean_ref_error": mean_ref_error,
        "ref_false_positive_count": ref_fp_total,
        "score_false_positive_count": score_fp_total,
        "ref_false_positive_rate": ref_fp_rate,
        "score_false_positive_rate": score_fp_rate,
        "voc_mean": voc_metrics['voc_mean'],
        "voc_std": voc_metrics['voc_std'],
        "voc_trajectories_count": voc_metrics['voc_count'],
        "gt_numeric_count": gt_numeric_count,
        "gt_na_count": gt_na_count,
        "total_tokens": total_tokens,
        "max_workers": args.max_workers,
        "temperature": args.temperature,
        "dataset_path": args.input,
        "output_file": str(output_path)
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="GPT-5 Visual Demo Progress Evaluation"
    )

    # Required arguments
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="OpenAI API key"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL dataset file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL results file"
    )

    # Optional arguments
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Image root directory (IMAGE_ROOT/{id}/{filename})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        choices=["gpt-5", "gpt-5-mini", "gpt-5-nano"],
        help="GPT-5 model version (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum concurrent workers (default: 5)"
    )
    parser.add_argument(
        "--num-inferences",
        type=int,
        default=4,
        help="Inferences per sample (default: 4)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=3000,
        help="Max completion tokens (default: 3000)"
    )

    # Image processing
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Minimum pixels for image processing"
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Maximum pixels for image processing"
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Print configuration
    print("\n" + "=" * 60)
    print("GPT-5 Visual Demo Progress Evaluation")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Image dir: {args.image_dir or 'Not specified'}")
    print(f"Model: {args.model}")
    print(f"Max workers: {args.max_workers}")
    print(f"Temperature: {args.temperature}")
    print(f"Resume: {'Yes' if args.resume else 'No'}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print("=" * 60 + "\n")

    start_time = time.time()

    try:
        run_visual_demo_evaluation(args)
        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed:.2f} seconds")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
