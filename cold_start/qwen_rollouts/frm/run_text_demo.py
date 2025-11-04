import os
import sys
import json
import argparse
import time
import re
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union
import torch
import traceback
import multiprocessing as mp
from multiprocessing import Manager, Process, Queue

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


def worker_process(gpu_id: int, data_slice: List, args, progress_queue: Queue, result_queue: Queue):
    """Worker process for one GPU with batch inference."""

    # Set this process to use only one GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Create GPU-specific output file
    gpu_output_file = args.output_file.replace('.jsonl', f'_gpu{gpu_id}.jsonl')

    try:
        # Initialize model on this GPU
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
                                "status": "failed"
                            }
                        }
                        results.append(result)
                        # Determine if this is an "n/a" sample (error case)
                        is_na_sample = isinstance(item['progress_score'], str) and item['progress_score'].lower() == "n/a"
                        progress_queue.put((1, float('inf'), float('inf'), 1, is_na_sample, False))  # (processed, score, ref_error, error, is_na_sample, is_na_correct)
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
                                "status": "failed" if has_error else "success"
                            }
                        }

                        results.append(result)

                        # Determine if this is an "n/a" sample
                        is_na_sample = isinstance(item['progress_score'], str) and item['progress_score'].lower() == "n/a"

                        # Report progress: (processed_count, score, ref_error, error, is_na_sample, is_na_correct)
                        # is_na_correct: True if both ground truth and prediction are "n/a"
                        is_na_correct = is_na_sample and evaluation_score == 0.0
                        progress_queue.put((1, evaluation_score, ref_error, 1 if has_error else 0, is_na_sample, is_na_correct))

                    except Exception as e:
                        # Parse error for this specific item
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
                            "response": f"Processing error: {str(e)}\nResponse: {response if response else ''}",
                            "meta_data": {
                                "id": item['id'],
                                "task_goal": item.get('task_goal', ''),
                                "status": "failed"
                            }
                        }
                        results.append(result)
                        # Determine if this is an "n/a" sample (error case)
                        is_na_sample = isinstance(item['progress_score'], str) and item['progress_score'].lower() == "n/a"
                        progress_queue.put((1, float('inf'), float('inf'), 1, is_na_sample, False))

            except Exception as e:
                # Batch error - mark all items in batch as errors
                for item in batch_items:
                    # Format ground truth score string (handle "n/a")
                    progress_score = item.get('progress_score', 0.0)
                    if isinstance(progress_score, str) and progress_score.lower() == "n/a":
                        ground_truth_score_str = "n/a"
                    else:
                        ground_truth_score_str = f"{int(progress_score * 100)}%"

                    # Format closest_idx (handle "n/a")
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
                            "status": "failed"
                        }
                    }
                    results.append(result)
                    # Determine if this is an "n/a" sample (batch error case)
                    is_na_sample = isinstance(progress_score, str) and progress_score.lower() == "n/a"
                    progress_queue.put((1, float('inf'), float('inf'), 1, is_na_sample, False))

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


def run_text_demo_inference(args):
    """Run text demo progress estimation with multi-GPU batch inference."""

    # Load dataset (already expanded N times)
    print(f"Loading dataset from {args.dataset_path}")
    image_root = args.image_root if hasattr(args, 'image_root') and args.image_root else None

    import sys
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

    # n/a sample statistics
    na_total = 0  # Total "n/a" samples
    na_pass = 0   # "n/a" samples where model correctly predicted "n/a"

    # Numeric sample statistics
    numeric_total = 0
    numeric_score_sum = 0.0
    numeric_ref_error_sum = 0.0
    numeric_valid = 0

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
            batch_na_total = 0
            batch_na_pass = 0
            batch_numeric_total = 0
            batch_numeric_score_sum = 0.0
            batch_numeric_ref_error_sum = 0.0
            batch_numeric_valid = 0

            while not progress_queue.empty():
                # Each progress update: (processed_count, score, ref_error, error, is_na_sample, is_na_correct)
                proc_count, score, ref_error, error, is_na_sample, is_na_correct = progress_queue.get_nowait()
                batch_proc_count += proc_count
                batch_errors += error

                # Count non-error samples for overall mean calculation
                if error == 0:
                    batch_valid_count += proc_count

                # Separate statistics for "n/a" samples vs numeric samples
                if is_na_sample:
                    batch_na_total += proc_count
                    if is_na_correct:
                        batch_na_pass += proc_count
                else:
                    batch_numeric_total += proc_count
                    # Only sum finite values for numeric samples
                    if score != float('inf'):
                        batch_numeric_score_sum += score
                    if ref_error != float('inf'):
                        batch_numeric_ref_error_sum += ref_error
                    if error == 0:
                        batch_numeric_valid += proc_count

                # For overall statistics (legacy)
                if score != float('inf'):
                    batch_score_sum += score
                if ref_error != float('inf'):
                    batch_ref_error_sum += ref_error

            # Update counters
            if batch_proc_count > 0:
                total_processed += batch_proc_count
                total_score_sum += batch_score_sum
                total_ref_error_sum += batch_ref_error_sum
                valid_count += batch_valid_count
                error_count += batch_errors
                na_total += batch_na_total
                na_pass += batch_na_pass
                numeric_total += batch_numeric_total
                numeric_score_sum += batch_numeric_score_sum
                numeric_ref_error_sum += batch_numeric_ref_error_sum
                numeric_valid += batch_numeric_valid
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

                # Build postfix string
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
                error_rate = error_count / total_processed * 100 if total_processed > 0 else 0.0
                postfix_parts.append(f"err={error_rate:.1f}%")

                pbar.set_postfix_str(", ".join(postfix_parts))
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
            all_results.sort(key=lambda x: x.get('id', ''))
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
                proc_count, score, ref_error, error = progress_queue.get_nowait()
                total_processed += proc_count
                if score != float('inf'):
                    total_score_sum += score
                if ref_error != float('inf'):
                    total_ref_error_sum += ref_error
                error_count += error
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
            print(f"Final count: {total_processed}/{len(data)}, MeanScore={mean_score:.3f}, MeanRef={mean_ref_error:.2f}")

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
    all_results.sort(key=lambda x: x.get('id', ''))

    # Write merged results to final output file
    print(f"\nWriting {len(all_results)} merged results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in all_results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    print(f"Merged results saved. Individual GPU files are preserved.")

    # Calculate final statistics
    valid_results = [r for r in all_results if r['meta_data']['status'] == 'success']
    # Filter out inf values for mean calculation
    finite_scores_all = [r['ref_score'] for r in all_results if r['ref_score'] != float('inf')]
    finite_scores_valid = [r['ref_score'] for r in valid_results if r['ref_score'] != float('inf')]
    finite_ref_errors_all = [r['pred_score'] for r in all_results if r['pred_score'] != float('inf')]
    finite_ref_errors_valid = [r['pred_score'] for r in valid_results if r['pred_score'] != float('inf')]

    mean_score = sum(finite_scores_all) / len(finite_scores_all) if finite_scores_all else float('inf')
    mean_score_valid = sum(finite_scores_valid) / len(finite_scores_valid) if finite_scores_valid else float('inf')
    mean_ref_error = sum(finite_ref_errors_all) / len(finite_ref_errors_all) if finite_ref_errors_all else float('inf')
    mean_ref_error_valid = sum(finite_ref_errors_valid) / len(finite_ref_errors_valid) if finite_ref_errors_valid else float('inf')
    error_rate = error_count / len(all_results) if all_results else 0.0

    # Calculate n/a pass rate
    na_pass_rate = na_pass / na_total if na_total > 0 else 0.0

    # Calculate numeric sample statistics
    numeric_mean_score = numeric_score_sum / numeric_valid if numeric_valid > 0 else float('inf')
    numeric_mean_ref_error = numeric_ref_error_sum / numeric_valid if numeric_valid > 0 else float('inf')

    # Print final summary
    print("\n" + "=" * 70)
    print("TEXT DEMO PROGRESS ESTIMATION SUMMARY")
    print("=" * 70)
    print(f"Total samples (expanded): {len(data)}")
    print(f"Original samples: {len(data) // args.num_inferences}")
    print(f"Inferences per sample: {args.num_inferences}")
    print(f"Processed: {len(all_results)}")
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
    print(f"Results saved to: {output_file}")
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
            "dataset_path": args.dataset_path,
            "model_path": args.model_path,
            "output_file": args.output_file
        }
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Text Demo Progress Estimation - Batch Inference")

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
    run_text_demo_inference(args)


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
