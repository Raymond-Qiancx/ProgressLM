"""
Text Demo progress estimation inference script for InternVL.
Single-process model parallel version (for large models).
"""

import os
import sys
import json
import argparse
import re
from tqdm import tqdm
from typing import Dict, Any
from scipy.stats import spearmanr
import numpy as np

from text_demo_dataset import load_text_demo_dataset, validate_image_path
from text_demo_prompt import build_text_demo_prompt_from_item, TEXT_DEMO_SYSTEM_PROMPT
from core.model import InternVLChat


def parse_text_demo_response(response: str) -> Dict[str, Any]:
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
    """Calculate normalized error: |ground_truth - predicted| / max(ground_truth, 1 - ground_truth)"""
    if predicted is None:
        return float('inf')
    max_possible = max(ground_truth, 1.0 - ground_truth)
    if max_possible == 0.0:
        return 0.0 if predicted == ground_truth else float('inf')
    return abs(ground_truth - predicted) / max_possible


def calculate_ref_error(predicted_ref, ground_truth_ref, num_demos=None):
    """Calculate normalized error for reference index."""
    if predicted_ref is None or not isinstance(predicted_ref, int):
        return float('inf')
    if num_demos is None:
        return float(abs(ground_truth_ref - predicted_ref))
    max_possible = max(ground_truth_ref - 1, num_demos - ground_truth_ref)
    if max_possible == 0:
        return 0.0
    return abs(ground_truth_ref - predicted_ref) / max_possible


def calculate_false_positives(predicted_ref, predicted_score, gt_ref, gt_score):
    gt_ref_is_na = (gt_ref is None)
    pred_ref_is_na = (predicted_ref == "n/a" or predicted_ref == "" or not isinstance(predicted_ref, int))
    ref_fp = gt_ref_is_na != pred_ref_is_na

    gt_score_is_na = (gt_score is None)
    pred_score_is_na = (predicted_score == "n/a" or predicted_score is None or not isinstance(predicted_score, (int, float)))
    score_fp = gt_score_is_na != pred_score_is_na

    return ref_fp, score_fp


def main():
    parser = argparse.ArgumentParser(description="Text Demo - InternVL (Single Process)")
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
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    data = load_text_demo_dataset(args.dataset_path, num_inferences=args.num_inferences, image_root=args.image_root)
    if args.limit > 0:
        data = data[:args.limit]

    print(f"Total samples: {len(data)}")

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

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    results = []
    total_score_sum = 0.0
    valid_count = 0
    batch_size = args.batch_size
    error_count = 0
    ref_fp_count = 0
    score_fp_count = 0

    pbar = tqdm(total=len(data), desc="Processing", ncols=160,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    for batch_start in range(0, len(data), batch_size):
        batch_items = data[batch_start:batch_start + batch_size]

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
                error_count += 1
                pbar.update(1)
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
                        total_score_sum += evaluation_score
                        valid_count += 1
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

                    # Update statistics
                    if has_error:
                        error_count += 1
                    if ref_fp:
                        ref_fp_count += 1
                    if score_fp:
                        score_fp_count += 1

                    # Update progress bar
                    mean_score = total_score_sum / valid_count if valid_count > 0 else 0.0
                    error_rate = error_count / len(results) * 100 if results else 0.0
                    pbar.set_postfix_str(f"Score={mean_score:.3f}, Err={error_rate:.1f}%")
                    pbar.update(1)

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
                    error_count += 1
                    pbar.update(1)

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
                error_count += 1
                pbar.update(1)

        # Periodic save (every 10 batches)
        if (batch_start // batch_size) % 10 == 0 and results:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')

    pbar.close()

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    mean_score = total_score_sum / valid_count if valid_count > 0 else 0.0
    summary = {"total_samples": len(results), "valid_samples": valid_count, "mean_evaluation_score": mean_score}

    summary_file = args.output_file.replace('.jsonl', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {args.output_file}")
    print(f"Mean Score Error: {mean_score:.4f}")


if __name__ == "__main__":
    main()
