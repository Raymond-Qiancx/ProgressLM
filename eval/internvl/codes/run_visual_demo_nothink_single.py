"""
Visual Demo progress estimation - nothink single-process version for InternVL.
For large models (38B+) that require device_map="auto" across multiple GPUs.
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


def main():
    parser = argparse.ArgumentParser(description="Visual Demo Nothink Single Process - InternVL")
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
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (usually 1 for large models)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Print GPU info
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load dataset
    data = load_visual_demo_dataset(args.dataset_path, num_inferences=args.num_inferences, image_root=args.image_root)
    if args.limit > 0:
        data = data[:args.limit]

    print(f"Dataset loaded: {len(data)} samples")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load model with device_map="auto" (will distribute across all visible GPUs)
    print("Loading model with device_map='auto' for multi-GPU distribution...")
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
    print("Model loaded successfully")

    # Process samples
    results = []
    total_score_sum = 0.0
    valid_count = 0
    gt_na_count = 0
    pred_na_correct_count = 0

    pbar = tqdm(data, desc="Progress")

    for item in pbar:
        try:
            # Validate image paths
            is_valid, error_msg = validate_image_paths(item)
            if not is_valid:
                gt_score = item.get('progress_score')
                gt_is_na = gt_score is None
                if gt_is_na:
                    gt_na_count += 1
                result = {
                    "score": None,
                    "ground_truth_score": f"{int(gt_score * 100)}%" if gt_score else "n/a",
                    "score_error": float('inf'),
                    "gt_is_na": gt_is_na,
                    "pred_na_correct": None,
                    "response": f"Validation error: {error_msg}",
                    "meta_data": {**item, "status": "failed"}
                }
                results.append(result)
            else:
                # Build prompt and run inference
                image_paths, prompt_text = build_visual_demo_prompt_from_item(item)
                response = model.generate_from_item(image_paths, prompt_text)

                # Parse response
                parsed = parse_score_only(response)
                predicted_score = parsed['score']
                has_error = parsed['parse_error']

                gt_score = item['progress_score']

                if gt_score is not None and isinstance(predicted_score, (int, float)):
                    evaluation_score = calculate_evaluation_score(predicted_score, gt_score)
                else:
                    evaluation_score = float('inf')

                # Check n/a accuracy (when gt is n/a, did model predict n/a?)
                pred_is_na = predicted_score == "n/a"
                gt_is_na = gt_score is None
                if gt_is_na:
                    gt_na_count += 1
                    if pred_is_na:
                        pred_na_correct_count += 1

                # Track valid scores
                if evaluation_score != float('inf'):
                    total_score_sum += evaluation_score
                    valid_count += 1

                result = {
                    "score": "n/a" if predicted_score == "n/a" else f"{int(predicted_score * 100)}%" if isinstance(predicted_score, (int, float)) else None,
                    "ground_truth_score": f"{int(gt_score * 100)}%" if gt_score else "n/a",
                    "score_error": evaluation_score,
                    "gt_is_na": gt_is_na,
                    "pred_na_correct": pred_is_na if gt_is_na else None,
                    "response": response,
                    "meta_data": {**item, "status": "failed" if has_error else "success"}
                }
                results.append(result)

        except Exception as e:
            gt_score = item.get('progress_score')
            gt_is_na = gt_score is None
            if gt_is_na:
                gt_na_count += 1
            result = {
                "score": None,
                "ground_truth_score": f"{int(gt_score * 100)}%" if gt_score else "n/a",
                "score_error": float('inf'),
                "gt_is_na": gt_is_na,
                "pred_na_correct": None,
                "response": f"Error: {str(e)}",
                "meta_data": {**item, "status": "failed"}
            }
            results.append(result)

        # Update tqdm with real-time metrics
        total_processed = len(results)
        if valid_count > 0:
            mean_err = total_score_sum / valid_count
            na_recall = pred_na_correct_count / gt_na_count * 100 if gt_na_count > 0 else 0
            pbar.set_postfix({"mean_err": f"{mean_err:.4f}", "na_recall": f"{na_recall:.1f}%", "valid": valid_count})

        # Save after each sample
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

    pbar.close()

    # Final save
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    # Save summary
    mean_score = total_score_sum / valid_count if valid_count > 0 else 0.0
    na_recall = pred_na_correct_count / gt_na_count * 100 if gt_na_count > 0 else 0.0
    summary = {
        "total_samples": len(results),
        "valid_samples": valid_count,
        "mean_score_error": mean_score,
        "gt_na_count": gt_na_count,
        "pred_na_correct_count": pred_na_correct_count,
        "na_recall": na_recall
    }

    summary_file = args.output_file.replace('.jsonl', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {args.output_file}")
    print(f"Mean Score Error: {mean_score:.4f}")
    print(f"N/A Recall (gt=n/a -> pred=n/a): {na_recall:.1f}% ({pred_na_correct_count}/{gt_na_count})")

    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
