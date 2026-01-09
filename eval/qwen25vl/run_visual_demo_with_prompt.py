import os
import sys
import json
import argparse
import time
import re
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
import torch
import traceback
import multiprocessing as mp
from multiprocessing import Manager, Process, Queue
from scipy.stats import spearmanr
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from visual_demo_dataset import load_visual_demo_dataset, validate_image_paths
from visual_demo_prompt import build_visual_demo_prompt_from_item, VISUAL_DEMO_SYSTEM_PROMPT
from qwen2_vl.model import Qwen2VLChat


def get_prompt_output_path(base_path: str, gpu_id: int) -> str:
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".jsonl"
    return f"{root}.gpu{gpu_id}{ext}"


def parse_visual_demo_response(response: str) -> Dict[str, Any]:
    """
    Parse the model's response to extract XML tags.

    Expected format:
    <ref_think>reasoning...</ref_think>
    <ref>2</ref>  (now expects 1-based integer index)
    <score_think>reasoning...</score_think>
    <score>8%</score>  (supports both "8%" and "0.08" formats)

    Args:
        response: Model output string

    Returns:
        Dictionary with parsed fields:
        {
            'ref_think': str,
            'ref': str or int (1-based image number),
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
            # Check for "n/a" first
            if ref_str.lower() in ["n/a", "na"]:
                result['ref'] = "n/a"
            else:
                try:
                    # Extract just the number (handle "No. 2", "2", "image 2", etc.)
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

        # Extract score (supports "8%", "0.08", or "n/a")
        score_match = re.search(r'<score>(.*?)</score>', response, re.DOTALL)
        if score_match:
            score_str = score_match.group(1).strip()
            # Check for "n/a" first
            if score_str.lower() in ["n/a", "na"]:
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


def calculate_evaluation_score(predicted: Optional[float], ground_truth: float) -> float:
    """
    Calculate evaluation score: |ground_truth - predicted| / max(ground_truth, 1 - ground_truth)

    Uses normalized error metric. Lower is better (0.0 = perfect prediction).
    This normalization treats small and large GT values fairly.

    Args:
        predicted: Predicted progress score (0-1) or None if parsing failed
        ground_truth: Ground truth progress score (0-1)

    Returns:
        Normalized error (0.0 = perfect, 1.0 = max possible error), or inf if invalid
    """
    if predicted is None:
        return float('inf')

    # Calculate max possible error for normalization
    max_possible = max(ground_truth, 1.0 - ground_truth)

    # Avoid division by zero (only happens when gt = 0.5 exactly, but max_possible would be 0.5)
    if max_possible == 0.0:
        return 0.0 if predicted == ground_truth else float('inf')

    normalized_error = abs(ground_truth - predicted) / max_possible
    return normalized_error


def calculate_ref_error(predicted_ref: Optional[int], ground_truth_ref: int, num_demos: int = None) -> float:
    """
    Calculate normalized reference index error: |gt_ref - pred_ref| / max(gt_ref - 1, num_demos - gt_ref)

    Uses normalized error metric for fair comparison across different trajectory lengths.

    Args:
        predicted_ref: Predicted reference index (1-based) or None
        ground_truth_ref: Ground truth closest index (1-based)
        num_demos: Total number of demos in the trajectory (required for normalization)

    Returns:
        Normalized error (0.0 = perfect, 1.0 = max possible error), or inf if invalid
    """
    if predicted_ref is None:
        return float('inf')

    if not isinstance(predicted_ref, int):
        return float('inf')

    # If num_demos not provided, fall back to absolute error
    if num_demos is None:
        return float(abs(ground_truth_ref - predicted_ref))

    # Calculate max possible error for normalization
    max_possible = max(ground_truth_ref - 1, num_demos - ground_truth_ref)

    if max_possible == 0:
        return 0.0

    normalized_error = abs(ground_truth_ref - predicted_ref) / max_possible
    return normalized_error

def calculate_false_positives(predicted_ref, predicted_score, gt_ref, gt_score) -> Tuple[bool, bool]:
    """
    Calculate false positive rates for ref and score.

    False positive occurs when:
    - GT is numeric but prediction is "n/a"
    - GT is "n/a" but prediction is numeric

    Correct cases:
    - Both GT and prediction are "n/a"
    - Both GT and prediction are numeric (use error calculation instead)

    Args:
        predicted_ref: Predicted ref (int, "n/a", or invalid)
        predicted_score: Predicted score (float, "n/a", or invalid)
        gt_ref: Ground truth ref (int or None)
        gt_score: Ground truth score (float or None)

    Returns:
        (is_ref_false_positive, is_score_false_positive)
    """
    # Check ref false positive
    gt_ref_is_na = (gt_ref is None)
    pred_ref_is_na = (predicted_ref == "n/a" or predicted_ref == "" or not isinstance(predicted_ref, int))

    # False positive: mismatch between GT and prediction
    ref_fp = gt_ref_is_na != pred_ref_is_na

    # Check score false positive
    gt_score_is_na = (gt_score is None)
    pred_score_is_na = (predicted_score == "n/a" or predicted_score is None or not isinstance(predicted_score, (int, float)))

    # False positive: mismatch between GT and prediction
    score_fp = gt_score_is_na != pred_score_is_na

    return ref_fp, score_fp


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


def worker_process(gpu_id: int, data_slice: List, args, progress_queue: Queue, result_queue: Queue):
    """Worker process for one GPU with batch inference."""

    # Set this process to use only one GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Create GPU-specific output file
    gpu_output_file = args.output_file.replace('.jsonl', f'_gpu{gpu_id}.jsonl')

    prompt_file = None
    try:
        if args.prompt_output:
            prompt_output_path = get_prompt_output_path(args.prompt_output, gpu_id)
            prompt_dir = os.path.dirname(prompt_output_path)
            if prompt_dir:
                os.makedirs(prompt_dir, exist_ok=True)
            prompt_file = open(prompt_output_path, "a", encoding="utf-8")

        # Initialize model on this GPU
        model = Qwen2VLChat(
            model_path=args.model_path,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            use_custom_prompt=False,
            system_prompt=VISUAL_DEMO_SYSTEM_PROMPT,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            verbose=False
        )

        # Process data in batches
        batch_size = args.batch_size
        results = []
        processed_count = 0

        i = 0
        while i < len(data_slice):
            batch_end = min(i + batch_size, len(data_slice))
            batch_items = data_slice[i:batch_end]

            try:
                # Build batch prompts
                batch_messages = []
                valid_batch_items = []

                for item in batch_items:
                    # Validate image paths
                    is_valid, error_msg = validate_image_paths(item)
                    if not is_valid:
                        # Skip this item, record error
                        gt_score = item.get('progress_score')
                        if gt_score is not None:
                            ground_truth_score_str = f"{int(gt_score * 100)}%"
                        else:
                            ground_truth_score_str = "n/a"

                        gt_ref = item.get('closest_idx')
                        result = {
                            "ref": None,
                            "score": None,
                            "closest_idx": str(gt_ref) if gt_ref is not None else "n/a",
                            "ground_truth_score": ground_truth_score_str,
                            "ref_score": float('inf'),
                            "pred_score": float('inf'),
                            "ref_false_positive": False,
                            "score_false_positive": False,
                            "response": f"Validation error: {error_msg}",
                            "meta_data": {
                                **item,  # Include all original data
                                "status": "failed"
                            }
                        }
                        results.append(result)
                        progress_queue.put((1, float('inf'), float('inf'), 0, 0, 1))  # (processed, score_error, ref_error, ref_fp, score_fp, parse_error)
                        continue

                    messages = build_visual_demo_prompt_from_item(
                        item,
                        min_pixels=args.min_pixels,
                        max_pixels=args.max_pixels
                    )
                    if prompt_file is not None:
                        prompt_record = {
                            "id": item.get("id"),
                            "messages": messages
                        }
                        prompt_file.write(json.dumps(prompt_record, ensure_ascii=False) + "\n")
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
                        parsed = parse_visual_demo_response(response)
                        predicted_score = parsed['score']
                        predicted_ref = parsed['ref']
                        has_error = parsed['parse_error']

                        # Get ground truth values
                        gt_score = item['progress_score']  # Can be float or None
                        gt_ref = item['closest_idx']  # Can be int or None

                        # Calculate false positives
                        ref_fp, score_fp = calculate_false_positives(
                            predicted_ref, predicted_score, gt_ref, gt_score
                        )

                        # Calculate evaluation score for progress (only for numeric pairs)
                        if gt_score is not None and isinstance(predicted_score, (int, float)):
                            evaluation_score = calculate_evaluation_score(predicted_score, gt_score)
                        else:
                            evaluation_score = float('inf')

                        # Calculate reference index error (only for numeric pairs)
                        if gt_ref is not None and isinstance(predicted_ref, int):
                            ref_error = calculate_ref_error(predicted_ref, gt_ref)
                        else:
                            ref_error = float('inf')

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

                        if predicted_ref == "n/a":
                            predicted_ref_str = "n/a"
                        elif isinstance(predicted_ref, int):
                            predicted_ref_str = str(predicted_ref)
                        else:
                            predicted_ref_str = None

                        result = {
                            "ref": predicted_ref_str,
                            "score": predicted_score_str,
                            "closest_idx": str(gt_ref) if gt_ref is not None else "n/a",
                            "ground_truth_score": ground_truth_score_str,
                            "ref_score": evaluation_score,
                            "pred_score": ref_error,
                            "ref_false_positive": ref_fp,
                            "score_false_positive": score_fp,
                            "response": response,
                            "meta_data": {
                                **item,  # Include all original data
                                "status": "failed" if has_error else "success"
                            }
                        }

                        results.append(result)

                        # Report progress: (processed_count, score_error, ref_error, ref_fp, score_fp, parse_error)
                        progress_queue.put((1, evaluation_score, ref_error, 1 if ref_fp else 0, 1 if score_fp else 0, 1 if has_error else 0))

                    except Exception as e:
                        # Parse error for this specific item
                        gt_score = item.get('progress_score')
                        if gt_score is not None:
                            ground_truth_score_str = f"{int(gt_score * 100)}%"
                        else:
                            ground_truth_score_str = "n/a"

                        gt_ref = item.get('closest_idx')
                        result = {
                            "ref": None,
                            "score": None,
                            "closest_idx": str(gt_ref) if gt_ref is not None else "n/a",
                            "ground_truth_score": ground_truth_score_str,
                            "ref_score": float('inf'),
                            "pred_score": float('inf'),
                            "ref_false_positive": False,
                            "score_false_positive": False,
                            "response": f"Processing error: {str(e)}\nResponse: {response if response else ''}",
                            "meta_data": {
                                **item,  # Include all original data
                                "status": "failed"
                            }
                        }
                        results.append(result)
                        progress_queue.put((1, float('inf'), float('inf'), 0, 0, 1))

            except Exception as e:
                # Batch error - mark all items in batch as errors
                for item in batch_items:
                    gt_score = item.get('progress_score')
                    if gt_score is not None:
                        ground_truth_score_str = f"{int(gt_score * 100)}%"
                    else:
                        ground_truth_score_str = "n/a"

                    gt_ref = item.get('closest_idx')
                    result = {
                        "ref": None,
                        "score": None,
                        "closest_idx": str(gt_ref) if gt_ref is not None else "n/a",
                        "ground_truth_score": ground_truth_score_str,
                        "ref_score": float('inf'),
                        "pred_score": float('inf'),
                        "ref_false_positive": False,
                        "score_false_positive": False,
                        "response": f"Batch error: {str(e)}",
                        "meta_data": {
                            **item,  # Include all original data
                            "status": "failed"
                        }
                    }
                    results.append(result)
                    progress_queue.put((1, float('inf'), float('inf'), 0, 0, 1))

            # Update processed count
            processed_count += len(batch_items)

            # Save results immediately after each batch
            with open(gpu_output_file, 'w', encoding='utf-8') as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')

            i = batch_end

        # Final save
        with open(gpu_output_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

        # Explicitly clean up model and CUDA resources before sending results
        try:
            del model
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            print(f"GPU {gpu_id}: Error during cleanup: {e}")

        # Send results back
        result_queue.put((gpu_id, results))

    except Exception as e:
        print(f"GPU {gpu_id} worker failed: {e}")
        traceback.print_exc()
        result_queue.put((gpu_id, []))
    finally:
        if prompt_file is not None:
            prompt_file.close()


def run_visual_demo_inference(args):
    """Run visual demo progress estimation with multi-GPU batch inference."""

    # Load dataset (already expanded N times)
    print(f"Loading dataset from {args.dataset_path}")
    image_root = args.image_root if hasattr(args, 'image_root') and args.image_root else None

    import sys
    from io import StringIO

    if not args.verbose:
        old_stdout = sys.stdout
        sys.stdout = StringIO()

    data = load_visual_demo_dataset(
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

    print(f"Using {num_gpus} GPUs: {gpu_ids}")
    print(f"Total samples (expanded): {len(data)}")
    print(f"Original samples: {len(data) // args.num_inferences}")
    print(f"Inferences per sample: {args.num_inferences}")
    print(f"Batch size per GPU: {args.batch_size}")

    # Split data across GPUs
    samples_per_gpu = len(data) // num_gpus
    data_slices = []
    for i in range(num_gpus):
        start_idx = i * samples_per_gpu
        if i == num_gpus - 1:
            end_idx = len(data)
        else:
            end_idx = start_idx + samples_per_gpu
        data_slices.append(data[start_idx:end_idx])
        print(f"GPU {gpu_ids[i]}: processing samples {start_idx}-{end_idx-1} ({len(data_slices[i])} samples)")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    output_file = args.output_file
    if args.prompt_output:
        prompt_dir = os.path.dirname(args.prompt_output)
        if prompt_dir:
            os.makedirs(prompt_dir, exist_ok=True)

    # Create queues for progress and results
    progress_queue = Queue()
    result_queue = Queue()

    # Start worker processes
    print(f"\nStarting {num_gpus} worker processes...")
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = Process(target=worker_process, args=(gpu_id, data_slices[i], args, progress_queue, result_queue))
        p.start()
        processes.append(p)
        print(f"  Started GPU {gpu_id} worker (PID: {p.pid})")

    print(f"\nProcessing {len(data)} samples with {num_gpus} GPUs (batch_size={args.batch_size} per GPU)...\n")

    # Monitor progress with unified tqdm
    total_processed = 0
    total_score_sum = 0.0
    total_ref_error_sum = 0.0
    valid_count = 0  # Count of non-error samples
    error_count = 0
    ref_fp_count = 0  # Count of ref false positives
    score_fp_count = 0  # Count of score false positives

    # Use tqdm with fixed width - dynamic single-line update only
    pbar = tqdm(total=len(data), desc="Progress", ncols=160,
                miniters=10, mininterval=2.0, smoothing=0.3, dynamic_ncols=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    # Monitor progress from all workers
    last_update_time = time.time()
    update_interval = 2.0  # Update every 2 seconds (less frequent)
    accumulated_updates = 0
    interrupted = False
    last_progress_time = time.time()
    no_progress_timeout = 60  # 60 seconds without progress triggers timeout check

    try:
        while total_processed < len(data):
            # Check if all workers are done
            all_workers_done = all(not p.is_alive() for p in processes)
            if all_workers_done:
                # All workers finished, break out of loop
                break

            # Collect all pending progress updates
            batch_proc_count = 0
            batch_score_sum = 0.0
            batch_ref_error_sum = 0.0
            batch_valid_count = 0
            batch_errors = 0
            batch_ref_fp = 0
            batch_score_fp = 0

            while not progress_queue.empty():
                # Each progress update: (processed_count, score, ref_error, ref_fp, score_fp, error)
                proc_count, score, ref_error, ref_fp, score_fp, error = progress_queue.get_nowait()
                batch_proc_count += proc_count
                # Only sum finite values
                if score != float('inf'):
                    batch_score_sum += score
                if ref_error != float('inf'):
                    batch_ref_error_sum += ref_error
                batch_errors += error
                batch_ref_fp += ref_fp
                batch_score_fp += score_fp
                # Only count non-error samples for mean calculation
                if error == 0:
                    batch_valid_count += proc_count

            # Update counters
            if batch_proc_count > 0:
                total_processed += batch_proc_count
                total_score_sum += batch_score_sum
                total_ref_error_sum += batch_ref_error_sum
                valid_count += batch_valid_count
                error_count += batch_errors
                ref_fp_count += batch_ref_fp
                score_fp_count += batch_score_fp
                accumulated_updates += batch_proc_count
                last_progress_time = time.time()  # Reset timeout timer

            # Check for timeout (no progress for too long)
            if time.time() - last_progress_time > no_progress_timeout:
                all_workers_done = all(not p.is_alive() for p in processes)
                if all_workers_done:
                    pbar.write(f"\nNo progress for {no_progress_timeout}s and all workers finished. Exiting loop.")
                    break

            # Update tqdm periodically
            current_time = time.time()
            if accumulated_updates > 0 and (current_time - last_update_time >= update_interval or total_processed >= len(data)):
                pbar.update(accumulated_updates)
                # Only calculate mean from valid (non-error) samples
                mean_score = total_score_sum / valid_count if valid_count > 0 else 0.0
                mean_ref_error = total_ref_error_sum / valid_count if valid_count > 0 else 0.0
                ref_fp_rate = ref_fp_count / total_processed * 100 if total_processed > 0 else 0.0
                score_fp_rate = score_fp_count / total_processed * 100 if total_processed > 0 else 0.0
                pbar.set_postfix_str(f"MeanScore={mean_score:.3f}, MeanRef={mean_ref_error:.2f}, RefFP={ref_fp_rate:.1f}%, ScoreFP={score_fp_rate:.1f}%")
                accumulated_updates = 0
                last_update_time = current_time

            # Small sleep to avoid busy waiting
            time.sleep(0.2)

    except KeyboardInterrupt:
        interrupted = True
        pbar.close()

        print("\n\n" + "=" * 70)
        print("WARNING: User interrupt (Ctrl+C) - Stopping all GPU processes...")
        print("=" * 70)

        # Terminate all processes
        print("Sending termination signal to all worker processes...")
        for i, p in enumerate(processes):
            if p.is_alive():
                print(f"  Stopping GPU {gpu_ids[i]} worker (PID: {p.pid})")
                p.terminate()

        # Wait for graceful shutdown (max 5 seconds)
        print("Waiting for processes to exit gracefully (max 5 seconds)...")
        start_wait = time.time()
        all_terminated = False
        while time.time() - start_wait < 5:
            if all(not p.is_alive() for p in processes):
                all_terminated = True
                break
            time.sleep(0.1)

        # Force kill if still running
        if not all_terminated:
            print("Force killing unresponsive processes...")
            for i, p in enumerate(processes):
                if p.is_alive():
                    print(f"  Force killing GPU {gpu_ids[i]} worker (PID: {p.pid})")
                    p.kill()
                    p.join(timeout=1)

        print("\nAll worker processes stopped")

        # Collect partial results
        print("Collecting partial results...")
        all_results = []
        while not result_queue.empty():
            try:
                gpu_id, results = result_queue.get(timeout=0.5)
                print(f"  Collected {len(results)} results from GPU {gpu_id}")
                all_results.extend(results)
            except:
                break

        if all_results:
            all_results.sort(key=lambda x: x.get('meta_data', {}).get('id', ''))
            partial_file = output_file.replace('.jsonl', '_partial.jsonl')
            print(f"\nSaving partial results to: {partial_file}")
            with open(partial_file, 'w', encoding='utf-8') as f:
                for res in all_results:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')

            print(f"\nSaved {len(all_results)} partial results")
            print("=" * 70)

        # Exit cleanly
        sys.exit(130)

    pbar.close()

    # Wait for all workers to complete
    if not interrupted:
        # First, drain any remaining progress updates before waiting for workers
        print("\nDraining remaining progress updates...")
        drain_count = 0
        while not progress_queue.empty():
            try:
                proc_count, score, ref_error, ref_fp, score_fp, error = progress_queue.get_nowait()
                total_processed += proc_count
                if score != float('inf'):
                    total_score_sum += score
                if ref_error != float('inf'):
                    total_ref_error_sum += ref_error
                error_count += error
                ref_fp_count += ref_fp
                score_fp_count += score_fp
                drain_count += proc_count
                # Only count non-error samples for mean calculation
                if error == 0:
                    valid_count += proc_count
            except:
                break

        if drain_count > 0:
            print(f"Drained {drain_count} remaining progress updates")
            mean_score = total_score_sum / valid_count if valid_count > 0 else 0.0
            mean_ref_error = total_ref_error_sum / valid_count if valid_count > 0 else 0.0
            ref_fp_rate = ref_fp_count / total_processed * 100 if total_processed > 0 else 0.0
            score_fp_rate = score_fp_count / total_processed * 100 if total_processed > 0 else 0.0
            print(f"Final count: {total_processed}/{len(data)}, MeanScore={mean_score:.3f}, MeanRef={mean_ref_error:.2f}, RefFP={ref_fp_rate:.1f}%, ScoreFP={score_fp_rate:.1f}%")

        print("\nWaiting for all workers to finish...")

        # Wait for workers while draining result_queue to prevent blocking
        all_done = False
        wait_start = time.time()
        max_wait = 300  # Max 5 minutes wait

        while not all_done and (time.time() - wait_start < max_wait):
            all_done = all(not p.is_alive() for p in processes)

            # Drain result_queue while waiting to prevent queue full blocking
            while not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except:
                    break

            if not all_done:
                time.sleep(0.5)

        if not all_done:
            print(f"\nWarning: Some workers didn't finish after {max_wait}s, forcing termination...")
            for i, p in enumerate(processes):
                if p.is_alive():
                    print(f"  Terminating GPU {gpu_ids[i]} worker (PID: {p.pid})")
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        print(f"  Force killing GPU {gpu_ids[i]} worker")
                        p.kill()

        # Final join to clean up
        for p in processes:
            if p.is_alive():
                p.join(timeout=1)

    # Sleep 10 seconds before merging results
    print("\nAll GPU workers finished. Sleeping for 10 seconds before merging results...")
    time.sleep(10)

    # Merge results from all GPU files
    print("Merging results from all GPU files...")
    all_results = []
    for gpu_id in gpu_ids:
        gpu_file = output_file.replace('.jsonl', f'_gpu{gpu_id}.jsonl')
        if os.path.exists(gpu_file):
            with open(gpu_file, 'r', encoding='utf-8') as f:
                gpu_results = []
                for line in f:
                    if line.strip():
                        gpu_results.append(json.loads(line))
                print(f"  Loaded {len(gpu_results)} results from GPU {gpu_id}")
                all_results.extend(gpu_results)
        else:
            print(f"  Warning: GPU {gpu_id} file not found: {gpu_file}")

    # Sort results by id
    all_results.sort(key=lambda x: x.get('meta_data', {}).get('id', ''))

    # Write merged results to final output file
    print(f"\nWriting {len(all_results)} merged results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in all_results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    print(f"Merged results saved. Individual GPU files are preserved.")

    # Calculate final statistics
    valid_results = [r for r in all_results if r['meta_data']['status'] == 'success']
    # Filter out inf values for mean calculation
    finite_scores_all = [r['ref_score'] for r in all_results if r.get('ref_score') != float('inf')]
    finite_scores_valid = [r['ref_score'] for r in valid_results if r.get('ref_score') != float('inf')]
    finite_ref_errors_all = [r['pred_score'] for r in all_results if r.get('pred_score') != float('inf')]
    finite_ref_errors_valid = [r['pred_score'] for r in valid_results if r.get('pred_score') != float('inf')]

    mean_score = sum(finite_scores_all) / len(finite_scores_all) if finite_scores_all else float('inf')
    mean_score_valid = sum(finite_scores_valid) / len(finite_scores_valid) if finite_scores_valid else float('inf')
    mean_ref_error = sum(finite_ref_errors_all) / len(finite_ref_errors_all) if finite_ref_errors_all else float('inf')
    mean_ref_error_valid = sum(finite_ref_errors_valid) / len(finite_ref_errors_valid) if finite_ref_errors_valid else float('inf')
    error_rate = error_count / len(all_results) if all_results else 0.0

    # Calculate false positive rates
    ref_fp_total = sum(1 for r in all_results if r.get('ref_false_positive', False))
    score_fp_total = sum(1 for r in all_results if r.get('score_false_positive', False))
    ref_fp_rate = ref_fp_total / len(all_results) if all_results else 0.0
    score_fp_rate = score_fp_total / len(all_results) if all_results else 0.0

    # Calculate VOC metrics
    print("\nCalculating VOC (trajectory order consistency) metrics...")
    voc_metrics = calculate_voc_metrics(all_results)

    # Count GT type distribution
    gt_numeric_count = sum(1 for r in all_results if r['meta_data'].get('closest_idx') is not None and r['meta_data'].get('progress_score') is not None)
    gt_na_count = len(all_results) - gt_numeric_count

    # Print final summary
    print("\n" + "=" * 70)
    print("VISUAL DEMO PROGRESS ESTIMATION SUMMARY")
    print("=" * 70)
    print(f"Total samples (expanded): {len(data)}")
    print(f"Original samples: {len(data) // args.num_inferences}")
    print(f"Inferences per sample: {args.num_inferences}")
    print(f"Processed: {len(all_results)}")
    print(f"Errors: {error_count} ({error_rate*100:.2f}%)")
    print(f"\nError Metrics:")
    print(f"  Mean evaluation score (all): {mean_score:.4f}")
    print(f"  Mean evaluation score (valid only): {mean_score_valid:.4f}")
    print(f"  Mean ref error (all): {mean_ref_error:.4f}")
    print(f"  Mean ref error (valid only): {mean_ref_error_valid:.4f}")
    print(f"\nFalse Positive Rates:")
    print(f"  Ref false positive rate: {ref_fp_rate*100:.2f}% ({ref_fp_total}/{len(all_results)})")
    print(f"  Score false positive rate: {score_fp_rate*100:.2f}% ({score_fp_total}/{len(all_results)})")
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
    print(f"\nResults saved to: {output_file}")
    print("=" * 70)

    # Save summary
    summary_file = output_file.replace('.jsonl', '_summary.json')
    summary = {
        "total_samples_expanded": len(data),
        "original_samples": len(data) // args.num_inferences,
        "num_inferences_per_sample": args.num_inferences,
        "processed": len(all_results),
        "errors": error_count,
        "error_rate": error_rate,
        "mean_evaluation_score_all": mean_score,
        "mean_evaluation_score_valid": mean_score_valid,
        "mean_ref_error_all": mean_ref_error,
        "mean_ref_error_valid": mean_ref_error_valid,
        "ref_false_positive_count": ref_fp_total,
        "score_false_positive_count": score_fp_total,
        "ref_false_positive_rate": ref_fp_rate,
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
    parser = argparse.ArgumentParser(description="Visual Demo Progress Estimation - Batch Inference")

    # Required arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the Qwen2-VL model")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the Visual Demo dataset (JSONL format)")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output JSONL file path for results")
    parser.add_argument("--prompt-output", type=str, default=None,
                        help="Optional JSONL file to dump model input messages per sample")

    # Optional arguments
    parser.add_argument("--image-root", type=str, default=None,
                        help="Root directory to prepend to relative image paths")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for inference per GPU (default: 16)")
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
    run_visual_demo_inference(args)


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
