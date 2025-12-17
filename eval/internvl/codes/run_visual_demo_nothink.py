"""
Visual Demo progress estimation - nothink version for InternVL.
Simplified output parsing (score only).
"""

import os
import sys
import json
import argparse
import time
import re
from tqdm import tqdm
from typing import List, Dict, Any
import torch
import traceback
import multiprocessing as mp
from multiprocessing import Process, Queue
import numpy as np

from visual_demo_dataset import load_visual_demo_dataset, validate_image_paths
from visual_demo_prompt_nothink import build_visual_demo_prompt_from_item, VISUAL_DEMO_SYSTEM_PROMPT
from core.model import InternVLChat


def parse_score_only(response: str) -> Dict[str, Any]:
    """Parse response expecting just a score value."""
    result = {'score': None, 'parse_error': False}
    try:
        response = response.strip()
        # Try to find percentage
        match = re.search(r'(\d+(?:\.\d+)?)\s*%', response)
        if match:
            score_value = float(match.group(1)) / 100.0
            result['score'] = max(0.0, min(1.0, score_value))
        else:
            # Try to find decimal
            match = re.search(r'(\d+(?:\.\d+)?)', response)
            if match:
                score_value = float(match.group(1))
                if score_value > 1.0:
                    score_value = score_value / 100.0
                result['score'] = max(0.0, min(1.0, score_value))
            else:
                if response.lower() in ["n/a", "na"]:
                    result['score'] = "n/a"
                else:
                    result['parse_error'] = True
    except:
        result['parse_error'] = True
    return result


def calculate_evaluation_score(predicted, ground_truth):
    if predicted is None:
        return float('inf')
    if ground_truth == 0.0:
        return 0.0 if predicted == 0.0 else float('inf')
    return abs(ground_truth - predicted) / ground_truth


def worker_process(gpu_id: int, data_slice: List, args, progress_queue: Queue, result_queue: Queue):
    """Worker process with batch inference support."""
    # Must set CUDA_VISIBLE_DEVICES before any CUDA operations
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    gpu_output_file = args.output_file.replace('.jsonl', f'_gpu{gpu_id}.jsonl')

    try:
        # Import torch locally to ensure CUDA_VISIBLE_DEVICES is set before CUDA init
        import torch as torch_local

        # Force CUDA initialization
        if torch_local.cuda.is_available():
            torch_local.cuda.set_device(0)
            _ = torch_local.zeros(1).cuda()
            print(f"GPU {gpu_id} worker: CUDA initialized")

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

            # 1. Validate and prepare batch
            valid_items = []
            valid_data = []

            for item in batch_items:
                is_valid, error_msg = validate_image_paths(item)
                if not is_valid:
                    gt_score = item.get('progress_score')
                    result = {
                        "score": None,
                        "ground_truth_score": f"{int(gt_score * 100)}%" if gt_score else "n/a",
                        "score_error": float('inf'),
                        "response": f"Validation error: {error_msg}",
                        "meta_data": {**item, "status": "failed"}
                    }
                    results.append(result)
                    progress_queue.put((1, float('inf'), 1))
                else:
                    image_paths, prompt_text = build_visual_demo_prompt_from_item(item)
                    valid_items.append(item)
                    valid_data.append((image_paths, prompt_text))

            if not valid_data:
                continue

            # 2. Batch inference
            try:
                responses = model.batch_generate(valid_data)

                # 3. Process responses
                for item, response in zip(valid_items, responses):
                    try:
                        parsed = parse_score_only(response)
                        predicted_score = parsed['score']
                        has_error = parsed['parse_error']

                        gt_score = item['progress_score']

                        if gt_score is not None and isinstance(predicted_score, (int, float)):
                            evaluation_score = calculate_evaluation_score(predicted_score, gt_score)
                        else:
                            evaluation_score = float('inf')

                        result = {
                            "score": "n/a" if predicted_score == "n/a" else f"{int(predicted_score * 100)}%" if isinstance(predicted_score, (int, float)) else None,
                            "ground_truth_score": f"{int(gt_score * 100)}%" if gt_score else "n/a",
                            "score_error": evaluation_score,
                            "response": response,
                            "meta_data": {**item, "status": "failed" if has_error else "success"}
                        }
                        results.append(result)
                        progress_queue.put((1, evaluation_score, 1 if has_error else 0))

                    except Exception as e:
                        gt_score = item.get('progress_score')
                        result = {
                            "score": None,
                            "ground_truth_score": f"{int(gt_score * 100)}%" if gt_score else "n/a",
                            "score_error": float('inf'),
                            "response": f"Error: {str(e)}",
                            "meta_data": {**item, "status": "failed"}
                        }
                        results.append(result)
                        progress_queue.put((1, float('inf'), 1))

            except Exception as e:
                # Batch failed, mark all items as error
                for item in valid_items:
                    gt_score = item.get('progress_score')
                    result = {
                        "score": None,
                        "ground_truth_score": f"{int(gt_score * 100)}%" if gt_score else "n/a",
                        "score_error": float('inf'),
                        "response": f"Batch error: {str(e)}",
                        "meta_data": {**item, "status": "failed"}
                    }
                    results.append(result)
                    progress_queue.put((1, float('inf'), 1))

            # Save periodically
            if len(results) % 50 == 0:
                with open(gpu_output_file, 'w', encoding='utf-8') as f:
                    for res in results:
                        f.write(json.dumps(res, ensure_ascii=False) + '\n')

        with open(gpu_output_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

        del model
        torch.cuda.empty_cache()
        result_queue.put((gpu_id, results))

    except Exception as e:
        print(f"GPU {gpu_id} worker failed: {e}")
        traceback.print_exc()
        result_queue.put((gpu_id, []))


def main():
    parser = argparse.ArgumentParser(description="Visual Demo Nothink - InternVL")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--max-num-tiles", type=int, default=12)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--num-inferences", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    data = load_visual_demo_dataset(args.dataset_path, num_inferences=args.num_inferences, image_root=args.image_root)
    if args.limit > 0:
        data = data[:args.limit]

    gpu_ids = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')]
    num_gpus = len(gpu_ids)

    samples_per_gpu = len(data) // num_gpus
    data_slices = []
    for i in range(num_gpus):
        start_idx = i * samples_per_gpu
        end_idx = len(data) if i == num_gpus - 1 else start_idx + samples_per_gpu
        data_slices.append(data[start_idx:end_idx])

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    progress_queue = Queue()
    result_queue = Queue()

    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = Process(target=worker_process, args=(gpu_id, data_slices[i], args, progress_queue, result_queue))
        p.start()
        processes.append(p)

    total_processed = 0
    total_score_sum = 0.0
    valid_count = 0

    pbar = tqdm(total=len(data), desc="Progress")

    while total_processed < len(data):
        if all(not p.is_alive() for p in processes):
            break
        while not progress_queue.empty():
            proc_count, score, error = progress_queue.get_nowait()
            total_processed += proc_count
            if score != float('inf'):
                total_score_sum += score
                valid_count += 1
            pbar.update(proc_count)
        time.sleep(0.5)

    pbar.close()
    for p in processes:
        p.join(timeout=60)

    all_results = []
    for gpu_id in gpu_ids:
        gpu_file = args.output_file.replace('.jsonl', f'_gpu{gpu_id}.jsonl')
        if os.path.exists(gpu_file):
            with open(gpu_file, 'r', encoding='utf-8') as f:
                for line in f:
                    all_results.append(json.loads(line))

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for res in all_results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    mean_score = total_score_sum / valid_count if valid_count > 0 else 0.0
    summary = {"total_samples": len(all_results), "valid_samples": valid_count, "mean_score_error": mean_score}

    summary_file = args.output_file.replace('.jsonl', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {args.output_file}")
    print(f"Mean Score Error: {mean_score:.4f}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
