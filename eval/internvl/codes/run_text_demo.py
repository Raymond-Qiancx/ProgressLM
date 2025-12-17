"""
Text Demo progress estimation inference script for InternVL.
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
from text_demo_dataset import load_text_demo_dataset, validate_image_path
from text_demo_prompt import build_text_demo_prompt_from_item, TEXT_DEMO_SYSTEM_PROMPT
from core.model import InternVLChat


def parse_text_demo_response(response: str) -> Dict[str, Any]:
    """Parse the model's response to extract XML tags."""
    result = {'ref_think': '', 'ref': '', 'score_think': '', 'score': None, 'parse_error': False}

    try:
        ref_think_match = re.search(r'<ref_think>(.*?)</ref_think>', response, re.DOTALL)
        if ref_think_match:
            result['ref_think'] = ref_think_match.group(1).strip()

        ref_match = re.search(r'<ref>(.*?)</ref>', response, re.DOTALL)
        if ref_match:
            ref_str = ref_match.group(1).strip()
            if ref_str.lower() in ["n/a", "na"]:
                result['ref'] = "n/a"
            else:
                ref_num = re.search(r'\d+', ref_str)
                if ref_num:
                    result['ref'] = int(ref_num.group())
                else:
                    result['ref'] = ref_str
        else:
            # Fallback: look for number between </ref_think> and <score_think>
            ref_fallback = re.search(r'</ref_think>\s*(\d+)\s*<score_think>', response, re.DOTALL)
            if ref_fallback:
                result['ref'] = int(ref_fallback.group(1))
            else:
                # Also try: number right after </ref_think>
                ref_fallback2 = re.search(r'</ref_think>\s*(\d+)', response, re.DOTALL)
                if ref_fallback2:
                    result['ref'] = int(ref_fallback2.group(1))

        score_think_match = re.search(r'<score_think>(.*?)</score_think>', response, re.DOTALL)
        if score_think_match:
            result['score_think'] = score_think_match.group(1).strip()

        score_match = re.search(r'<score>(.*?)</score>', response, re.DOTALL)
        if score_match:
            score_str = score_match.group(1).strip()
            if score_str.lower() in ["n/a", "na"]:
                result['score'] = "n/a"
            else:
                if score_str.endswith('%'):
                    score_value = float(score_str[:-1]) / 100.0
                else:
                    score_value = float(score_str)
                    if score_value > 1.0:
                        score_value = score_value / 100.0
                result['score'] = max(0.0, min(1.0, score_value))
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


def calculate_ref_error(predicted_ref, ground_truth_ref):
    if predicted_ref is None or not isinstance(predicted_ref, int):
        return float('inf')
    return float(abs(ground_truth_ref - predicted_ref))


def calculate_false_positives(predicted_ref, predicted_score, gt_ref, gt_score):
    gt_ref_is_na = (gt_ref is None)
    pred_ref_is_na = (predicted_ref == "n/a" or predicted_ref == "" or not isinstance(predicted_ref, int))
    ref_fp = gt_ref_is_na != pred_ref_is_na

    gt_score_is_na = (gt_score is None)
    pred_score_is_na = (predicted_score == "n/a" or predicted_score is None or not isinstance(predicted_score, (int, float)))
    score_fp = gt_score_is_na != pred_score_is_na

    return ref_fp, score_fp


def calculate_voc_metrics(results):
    from collections import defaultdict
    trajectories = defaultdict(list)

    for res in results:
        meta = res.get('meta_data', {})
        traj_id = meta.get('id', '')
        gt_ref = meta.get('closest_idx')
        gt_score = meta.get('progress_score')

        if gt_ref is not None and gt_score is not None:
            pred_score = res.get('score')
            pred_score_numeric = 0.0 if pred_score == "n/a" or pred_score is None else float(pred_score) if isinstance(pred_score, (int, float)) else 0.0
            trajectories[traj_id].append({'gt_score': gt_score, 'pred_score': pred_score_numeric, 'result': res})

    voc_values = []
    for traj_id, samples in trajectories.items():
        if len(samples) <= 1:
            continue
        samples_sorted_by_gt = sorted(samples, key=lambda x: x['gt_score'])
        true_order = list(range(len(samples_sorted_by_gt)))
        samples_sorted_by_pred = sorted(samples, key=lambda x: x['pred_score'])
        pred_rank_map = {id(s['result']): rank for rank, s in enumerate(samples_sorted_by_pred)}
        pred_order = [pred_rank_map[id(s['result'])] for s in samples_sorted_by_gt]

        if len(set(true_order)) > 1 and len(set(pred_order)) > 1:
            correlation, _ = spearmanr(true_order, pred_order)
            if not np.isnan(correlation):
                voc_values.append(correlation)

    if voc_values:
        return {'voc_mean': float(np.mean(voc_values)), 'voc_std': float(np.std(voc_values)), 'voc_count': len(voc_values)}
    return {'voc_mean': None, 'voc_std': None, 'voc_count': 0}


def worker_process(gpu_id: int, data_slice: List, args, progress_queue: Queue, result_queue: Queue):
    """Worker process for one GPU with batch inference support."""
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
            system_prompt=TEXT_DEMO_SYSTEM_PROMPT,
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
                is_valid, error_msg = validate_image_path(item)
                if not is_valid:
                    gt_score = item.get('progress_score')
                    gt_ref = item.get('closest_idx')
                    result = {
                        "ref": None, "score": None,
                        "closest_idx": str(gt_ref) if gt_ref else "n/a",
                        "ground_truth_score": f"{int(gt_score * 100)}%" if gt_score else "n/a",
                        "ref_score": float('inf'), "pred_score": float('inf'),
                        "response": f"Validation error: {error_msg}",
                        "meta_data": {**item, "status": "failed"}
                    }
                    results.append(result)
                    progress_queue.put((1, float('inf'), float('inf'), 0, 0, 1))
                else:
                    image_paths, prompt_text = build_text_demo_prompt_from_item(item)
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
                        parsed = parse_text_demo_response(response)
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

                        ref_error = calculate_ref_error(predicted_ref, gt_ref) if gt_ref and isinstance(predicted_ref, int) else float('inf')

                        result = {
                            "ref": "n/a" if predicted_ref == "n/a" else str(predicted_ref) if isinstance(predicted_ref, int) else None,
                            "score": "n/a" if predicted_score == "n/a" else f"{int(predicted_score * 100)}%" if isinstance(predicted_score, (int, float)) else None,
                            "closest_idx": str(gt_ref) if gt_ref else "n/a",
                            "ground_truth_score": f"{int(gt_score * 100)}%" if gt_score else "n/a",
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
                            "ref": None, "score": None,
                            "closest_idx": str(gt_ref) if gt_ref else "n/a",
                            "ground_truth_score": f"{int(gt_score * 100)}%" if gt_score else "n/a",
                            "ref_score": float('inf'), "pred_score": float('inf'),
                            "response": f"Error: {str(e)}",
                            "meta_data": {**item, "status": "failed"}
                        }
                        results.append(result)
                        progress_queue.put((1, float('inf'), float('inf'), 0, 0, 1))

            except Exception as e:
                # Batch failed, mark all items as error
                for item in valid_items:
                    gt_score = item.get('progress_score')
                    gt_ref = item.get('closest_idx')
                    result = {
                        "ref": None, "score": None,
                        "closest_idx": str(gt_ref) if gt_ref else "n/a",
                        "ground_truth_score": f"{int(gt_score * 100)}%" if gt_score else "n/a",
                        "ref_score": float('inf'), "pred_score": float('inf'),
                        "response": f"Batch error: {str(e)}",
                        "meta_data": {**item, "status": "failed"}
                    }
                    results.append(result)
                    progress_queue.put((1, float('inf'), float('inf'), 0, 0, 1))

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


def run_text_demo_inference(args):
    """Run text demo progress estimation with multi-GPU inference."""
    print(f"Loading dataset from {args.dataset_path}")
    image_root = args.image_root if hasattr(args, 'image_root') and args.image_root else None

    data = load_text_demo_dataset(args.dataset_path, num_inferences=args.num_inferences, image_root=image_root)

    if args.limit > 0:
        data = data[:args.limit]

    gpu_ids = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')]
    num_gpus = len(gpu_ids)

    print(f"Using {num_gpus} GPUs: {gpu_ids}")
    print(f"Total samples: {len(data)}")

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
        all_done = all(not p.is_alive() for p in processes)
        if all_done:
            break

        while not progress_queue.empty():
            proc_count, score, ref_error, ref_fp, score_fp, error = progress_queue.get_nowait()
            total_processed += proc_count
            if score != float('inf'):
                total_score_sum += score
            if error == 0:
                valid_count += proc_count
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

    all_results.sort(key=lambda x: x.get('meta_data', {}).get('id', ''))
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for res in all_results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    voc_metrics = calculate_voc_metrics(all_results)
    mean_score = total_score_sum / valid_count if valid_count > 0 else 0.0

    summary = {
        "total_samples": len(all_results),
        "valid_samples": valid_count,
        "mean_evaluation_score": mean_score,
        "voc_mean": voc_metrics['voc_mean'],
    }

    summary_file = args.output_file.replace('.jsonl', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {args.output_file}")
    print(f"Mean Score Error: {mean_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Text Demo Progress Estimation - InternVL")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--max-num-tiles", type=int, default=12)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--num-inferences", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    run_text_demo_inference(args)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
