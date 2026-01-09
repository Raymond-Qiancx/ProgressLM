"""
Visual Demo progress estimation inference script for InternVL.
Multi-GPU data parallel version.
"""

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
from multiprocessing import Process, Queue
from scipy.stats import spearmanr
import numpy as np

# Local imports
from visual_demo_dataset import load_visual_demo_dataset, validate_image_paths
from visual_demo_prompt import build_visual_demo_prompt_from_item, VISUAL_DEMO_SYSTEM_PROMPT
from core.model import InternVLChat


def parse_visual_demo_response(response: str) -> Dict[str, Any]:
    """Parse the model's response to extract XML tags."""
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

        # Extract ref
        ref_match = re.search(r'<ref>(.*?)</ref>', response, re.DOTALL)
        if ref_match:
            ref_str = ref_match.group(1).strip()
            if ref_str.lower() in ["n/a", "na"]:
                result['ref'] = "n/a"
            else:
                try:
                    ref_num = re.search(r'\d+', ref_str)
                    if ref_num:
                        result['ref'] = int(ref_num.group())
                    else:
                        result['ref'] = ref_str
                except (ValueError, AttributeError):
                    result['ref'] = ref_str
        else:
            # Fallback 1: look for number between </ref_think> and <score_think>
            ref_fallback = re.search(r'</ref_think>\s*(\d+)\s*<score_think>', response, re.DOTALL)
            if ref_fallback:
                result['ref'] = int(ref_fallback.group(1))
            else:
                # Fallback 2: number right after </ref_think>
                ref_fallback2 = re.search(r'</ref_think>\s*(\d+)', response, re.DOTALL)
                if ref_fallback2:
                    result['ref'] = int(ref_fallback2.group(1))
                else:
                    # Fallback 3: extract "Step N" from <ref_think> content
                    ref_think_content = result.get('ref_think', '')
                    step_match = re.search(r'(?:Step|step)\s*(\d+)', ref_think_content)
                    if step_match:
                        result['ref'] = int(step_match.group(1))
                    else:
                        # Fallback 4: extract first "Step N" from entire response
                        step_match_global = re.search(r'(?:Step|step)\s*(\d+)', response)
                        if step_match_global:
                            result['ref'] = int(step_match_global.group(1))

        # Extract score_think
        score_think_match = re.search(r'<score_think>(.*?)</score_think>', response, re.DOTALL)
        if score_think_match:
            result['score_think'] = score_think_match.group(1).strip()

        # Extract score
        score_match = re.search(r'<score>(.*?)</score>', response, re.DOTALL)
        if score_match:
            score_str = score_match.group(1).strip()
            if score_str.lower() in ["n/a", "na"]:
                result['score'] = "n/a"
            else:
                try:
                    if score_str.endswith('%'):
                        score_value = float(score_str[:-1]) / 100.0
                    else:
                        score_value = float(score_str)
                        if score_value > 1.0:
                            score_value = score_value / 100.0
                    result['score'] = max(0.0, min(1.0, score_value))
                except ValueError:
                    result['parse_error'] = True
        else:
            result['parse_error'] = True

    except Exception as e:
        result['parse_error'] = True

    return result


def calculate_evaluation_score(predicted: Optional[float], ground_truth: float) -> float:
    """Calculate normalized error: |ground_truth - predicted| / max(ground_truth, 1 - ground_truth)"""
    if predicted is None:
        return float('inf')
    max_possible = max(ground_truth, 1.0 - ground_truth)
    if max_possible == 0.0:
        return 0.0 if predicted == ground_truth else float('inf')
    return abs(ground_truth - predicted) / max_possible


def calculate_ref_error(predicted_ref: Optional[int], ground_truth_ref: int, num_demos: int = None) -> float:
    """Calculate normalized error for reference index."""
    if predicted_ref is None or not isinstance(predicted_ref, int):
        return float('inf')
    if num_demos is None:
        return float(abs(ground_truth_ref - predicted_ref))
    max_possible = max(ground_truth_ref - 1, num_demos - ground_truth_ref)
    if max_possible == 0:
        return 0.0
    return abs(ground_truth_ref - predicted_ref) / max_possible


def calculate_false_positives(predicted_ref, predicted_score, gt_ref, gt_score) -> Tuple[bool, bool]:
    """Calculate false positive rates for ref and score."""
    gt_ref_is_na = (gt_ref is None)
    pred_ref_is_na = (predicted_ref == "n/a" or predicted_ref == "" or not isinstance(predicted_ref, int))
    ref_fp = gt_ref_is_na != pred_ref_is_na

    gt_score_is_na = (gt_score is None)
    pred_score_is_na = (predicted_score == "n/a" or predicted_score is None or not isinstance(predicted_score, (int, float)))
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
    """Worker process for one GPU with batch inference support."""
    # Must set CUDA_VISIBLE_DEVICES before any CUDA operations
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    gpu_output_file = args.output_file.replace('.jsonl', f'_gpu{gpu_id}.jsonl')

    try:
        # Import torch here to ensure CUDA_VISIBLE_DEVICES is set before torch initializes CUDA
        import torch as torch_local

        # Force CUDA initialization
        if torch_local.cuda.is_available():
            torch_local.cuda.set_device(0)
            _ = torch_local.zeros(1).cuda()
            print(f"GPU {gpu_id} worker: CUDA initialized, device count: {torch_local.cuda.device_count()}")
        else:
            print(f"GPU {gpu_id} worker: CUDA not available!")

        # Initialize InternVL model
        model = InternVLChat(
            model_path=args.model_path,
            max_num_tiles=args.max_num_tiles,
            input_size=args.input_size,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            system_prompt=VISUAL_DEMO_SYSTEM_PROMPT,
            verbose=args.verbose
        )

        results = []
        batch_size = args.batch_size

        # Process in batches
        for batch_start in range(0, len(data_slice), batch_size):
            batch_items = data_slice[batch_start:batch_start + batch_size]

            # 1. Validate and prepare batch data
            valid_items = []
            valid_data = []  # List of (image_paths, prompt_text)

            for item in batch_items:
                is_valid, error_msg = validate_image_paths(item)
                if not is_valid:
                    # Record error result immediately
                    gt_score = item.get('progress_score')
                    gt_ref = item.get('closest_idx')
                    result = {
                        "ref": None,
                        "score": None,
                        "closest_idx": str(gt_ref) if gt_ref is not None else "n/a",
                        "ground_truth_score": f"{int(gt_score * 100)}%" if gt_score is not None else "n/a",
                        "ref_score": float('inf'),
                        "pred_score": float('inf'),
                        "ref_false_positive": False,
                        "score_false_positive": False,
                        "response": f"Validation error: {error_msg}",
                        "meta_data": {**item, "status": "failed"}
                    }
                    results.append(result)
                    progress_queue.put((1, float('inf'), float('inf'), 0, 0, 1))
                else:
                    image_paths, prompt_text = build_visual_demo_prompt_from_item(item)
                    valid_items.append(item)
                    valid_data.append((image_paths, prompt_text))

            if not valid_data:
                continue

            # 2. Batch inference
            try:
                responses = model.batch_generate(valid_data)

                # 3. Process each response
                for item, response in zip(valid_items, responses):
                    try:
                        parsed = parse_visual_demo_response(response)
                        predicted_score = parsed['score']
                        predicted_ref = parsed['ref']
                        has_error = parsed['parse_error']

                        gt_score = item['progress_score']
                        gt_ref = item['closest_idx']

                        ref_fp, score_fp = calculate_false_positives(predicted_ref, predicted_score, gt_ref, gt_score)

                        if gt_score is not None and isinstance(predicted_score, (int, float)):
                            evaluation_score = calculate_evaluation_score(predicted_score, gt_score)
                        else:
                            evaluation_score = float('inf')

                        if gt_ref is not None and isinstance(predicted_ref, int):
                            ref_error = calculate_ref_error(predicted_ref, gt_ref)
                        else:
                            ref_error = float('inf')

                        if predicted_score == "n/a":
                            predicted_score_str = "n/a"
                        elif isinstance(predicted_score, (int, float)):
                            predicted_score_str = f"{int(predicted_score * 100)}%"
                        else:
                            predicted_score_str = None

                        ground_truth_score_str = f"{int(gt_score * 100)}%" if gt_score is not None else "n/a"
                        predicted_ref_str = "n/a" if predicted_ref == "n/a" else str(predicted_ref) if isinstance(predicted_ref, int) else None

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
                            "meta_data": {**item, "status": "failed" if has_error else "success"}
                        }
                        results.append(result)
                        progress_queue.put((1, evaluation_score, ref_error, 1 if ref_fp else 0, 1 if score_fp else 0, 1 if has_error else 0))

                    except Exception as e:
                        gt_score = item.get('progress_score')
                        gt_ref = item.get('closest_idx')
                        result = {
                            "ref": None,
                            "score": None,
                            "closest_idx": str(gt_ref) if gt_ref is not None else "n/a",
                            "ground_truth_score": f"{int(gt_score * 100)}%" if gt_score is not None else "n/a",
                            "ref_score": float('inf'),
                            "pred_score": float('inf'),
                            "ref_false_positive": False,
                            "score_false_positive": False,
                            "response": f"Error processing response: {str(e)}",
                            "meta_data": {**item, "status": "failed"}
                        }
                        results.append(result)
                        progress_queue.put((1, float('inf'), float('inf'), 0, 0, 1))

            except Exception as e:
                # Batch inference failed, mark all items as error
                for item in valid_items:
                    gt_score = item.get('progress_score')
                    gt_ref = item.get('closest_idx')
                    result = {
                        "ref": None,
                        "score": None,
                        "closest_idx": str(gt_ref) if gt_ref is not None else "n/a",
                        "ground_truth_score": f"{int(gt_score * 100)}%" if gt_score is not None else "n/a",
                        "ref_score": float('inf'),
                        "pred_score": float('inf'),
                        "ref_false_positive": False,
                        "score_false_positive": False,
                        "response": f"Batch inference error: {str(e)}",
                        "meta_data": {**item, "status": "failed"}
                    }
                    results.append(result)
                    progress_queue.put((1, float('inf'), float('inf'), 0, 0, 1))

            # Save periodically
            if len(results) % 50 == 0:
                with open(gpu_output_file, 'w', encoding='utf-8') as f:
                    for res in results:
                        f.write(json.dumps(res, ensure_ascii=False) + '\n')

        # Final save
        with open(gpu_output_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

        # Cleanup
        del model
        torch.cuda.empty_cache()
        result_queue.put((gpu_id, results))

    except Exception as e:
        print(f"GPU {gpu_id} worker failed: {e}")
        traceback.print_exc()
        result_queue.put((gpu_id, []))


def run_visual_demo_inference(args):
    """Run visual demo progress estimation with multi-GPU inference."""
    print(f"Loading dataset from {args.dataset_path}")
    image_root = args.image_root if hasattr(args, 'image_root') and args.image_root else None

    data = load_visual_demo_dataset(
        args.dataset_path,
        num_inferences=args.num_inferences,
        image_root=image_root
    )

    if args.limit > 0:
        data = data[:args.limit]
        print(f"Limited to first {args.limit} samples")

    # Get GPU configuration
    gpu_ids = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')]
    num_gpus = len(gpu_ids)

    print(f"Using {num_gpus} GPUs: {gpu_ids}")
    print(f"Total samples: {len(data)}")

    # Split data across GPUs
    samples_per_gpu = len(data) // num_gpus
    data_slices = []
    for i in range(num_gpus):
        start_idx = i * samples_per_gpu
        end_idx = len(data) if i == num_gpus - 1 else start_idx + samples_per_gpu
        data_slices.append(data[start_idx:end_idx])
        print(f"GPU {gpu_ids[i]}: {len(data_slices[i])} samples")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Create queues
    progress_queue = Queue()
    result_queue = Queue()

    # Start workers
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = Process(target=worker_process, args=(gpu_id, data_slices[i], args, progress_queue, result_queue))
        p.start()
        processes.append(p)
        print(f"Started GPU {gpu_id} worker (PID: {p.pid})")

    # Monitor progress
    total_processed = 0
    total_score_sum = 0.0
    total_ref_error_sum = 0.0
    valid_count = 0
    error_count = 0

    pbar = tqdm(total=len(data), desc="Progress")

    while total_processed < len(data):
        all_done = all(not p.is_alive() for p in processes)
        if all_done:
            break

        while not progress_queue.empty():
            proc_count, score, ref_error, ref_fp, score_fp, error = progress_queue.get_nowait()
            total_processed += proc_count
            if score != float('inf'):
                total_score_sum += score
            if ref_error != float('inf'):
                total_ref_error_sum += ref_error
            if error == 0:
                valid_count += proc_count
            error_count += error
            pbar.update(proc_count)

        time.sleep(0.5)

    pbar.close()

    # Drain queues to prevent blocking
    while not progress_queue.empty():
        try:
            progress_queue.get_nowait()
        except:
            break

    while not result_queue.empty():
        try:
            result_queue.get_nowait()
        except:
            break

    # Wait for all processes
    for p in processes:
        p.join(timeout=60)
        if p.is_alive():
            print(f"Force terminating process {p.pid}")
            p.terminate()
            p.join(timeout=5)

    # Collect results
    all_results = []
    for gpu_id in gpu_ids:
        gpu_file = args.output_file.replace('.jsonl', f'_gpu{gpu_id}.jsonl')
        if os.path.exists(gpu_file):
            with open(gpu_file, 'r', encoding='utf-8') as f:
                for line in f:
                    all_results.append(json.loads(line))

    # Sort and save
    all_results.sort(key=lambda x: x.get('meta_data', {}).get('id', ''))
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for res in all_results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    # Calculate statistics
    voc_metrics = calculate_voc_metrics(all_results)
    mean_score = total_score_sum / valid_count if valid_count > 0 else 0.0
    mean_ref_error = total_ref_error_sum / valid_count if valid_count > 0 else 0.0

    # Save summary
    summary = {
        "total_samples": len(all_results),
        "valid_samples": valid_count,
        "error_count": error_count,
        "mean_evaluation_score": mean_score,
        "mean_ref_error": mean_ref_error,
        "voc_mean": voc_metrics['voc_mean'],
        "voc_std": voc_metrics['voc_std'],
        "voc_count": voc_metrics['voc_count'],
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
    }

    summary_file = args.output_file.replace('.jsonl', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {args.output_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"\nFinal Statistics:")
    print(f"  Mean Score Error: {mean_score:.4f}")
    print(f"  Mean Ref Error: {mean_ref_error:.4f}")
    print(f"  VOC Mean: {voc_metrics['voc_mean']}")


def main():
    parser = argparse.ArgumentParser(description="Visual Demo Progress Estimation - InternVL")
    parser.add_argument("--model-path", type=str, required=True, help="Path to InternVL model")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset JSONL")
    parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    parser.add_argument("--image-root", type=str, default=None, help="Image root directory")
    parser.add_argument("--max-num-tiles", type=int, default=12, help="Max tiles per image")
    parser.add_argument("--input-size", type=int, default=448, help="Input tile size")
    parser.add_argument("--num-inferences", type=int, default=1, help="Inferences per sample")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Max new tokens")
    parser.add_argument("--limit", type=int, default=-1, help="Limit samples (-1 for all)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset not found: {args.dataset_path}")
        sys.exit(1)
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found: {args.model_path}")
        sys.exit(1)

    run_visual_demo_inference(args)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
