#!/usr/bin/env python3
"""
LIV Model Evaluation for Visual Demo Progress Estimation Task.

This script uses the LIV (Language-Image representations and rewards for robotic control)
model to evaluate progress estimation based on visual demonstrations.

Usage:
    python run_liv_visual_demo.py \
        --model-path /path/to/LIV/model \
        --dataset-path /path/to/visual_demo.jsonl \
        --image-root /path/to/images \
        --output-dir ./outputs
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms as T
from PIL import Image
import omegaconf
import hydra

# Add LIV to path
LIV_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if LIV_PATH not in sys.path:
    sys.path.insert(0, LIV_PATH)

# Add qwen25vl to path for dataset loading
QWEN25VL_PATH = os.path.join(os.path.dirname(LIV_PATH), "..", "qwen25vl")
QWEN25VL_PATH = os.path.abspath(QWEN25VL_PATH)
if QWEN25VL_PATH not in sys.path:
    sys.path.insert(0, QWEN25VL_PATH)

from visual_demo_dataset import load_visual_demo_dataset, validate_image_paths


def load_liv_from_path(model_dir: str) -> torch.nn.Module:
    """
    Load LIV model from local path.

    Args:
        model_dir: Path to LIV model directory containing model.pt and config.yaml

    Returns:
        Loaded LIV model wrapped in DataParallel
    """
    from liv.models.model_liv import LIV

    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelpath = os.path.join(model_dir, "model.pt")
    configpath = os.path.join(model_dir, "config.yaml")

    if not os.path.exists(modelpath):
        raise FileNotFoundError(f"Model file not found: {modelpath}")
    if not os.path.exists(configpath):
        raise FileNotFoundError(f"Config file not found: {configpath}")

    cfg = omegaconf.OmegaConf.load(configpath)
    cfg.agent.device = device

    model = hydra.utils.instantiate(cfg.agent)
    model = torch.nn.DataParallel(model)
    state_dict = torch.load(modelpath, map_location=device)['liv']
    model.load_state_dict(state_dict)

    return model


def load_image(image_path: str) -> Image.Image:
    """
    Load image with consistent preprocessing (same as qwen25vl).

    Args:
        image_path: Path to image file

    Returns:
        PIL Image in RGB mode
    """
    image = Image.open(image_path)
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def evaluate_visual_demo_sample(
    liv: torch.nn.Module,
    item: Dict[str, Any],
    device: str,
    transform: T.Compose
) -> Dict[str, Any]:
    """
    Evaluate a single Visual Demo sample using LIV.

    Args:
        liv: LIV model
        item: Dataset item containing visual_demo and stage_to_estimate
        device: Device to run inference on
        transform: Image transform

    Returns:
        Dictionary with prediction results
    """
    try:
        # Load and preprocess current state image
        stage_path = item['stage_to_estimate']
        stage_image = load_image(stage_path)
        stage_tensor = transform(stage_image).unsqueeze(0).to(device)

        # Load and preprocess visual demo images
        visual_demo = item['visual_demo']
        demo_tensors = []
        for demo_path in visual_demo:
            demo_image = load_image(demo_path)
            demo_tensors.append(transform(demo_image))
        demo_batch = torch.stack(demo_tensors).to(device)  # [N, 3, H, W]

        with torch.no_grad():
            # Encode current state image
            stage_emb = liv(input=stage_tensor, modality="vision")  # [1, 1024]

            # Encode all demo images
            demo_embs = liv(input=demo_batch, modality="vision")  # [N, 1024]

            # Compute similarities
            similarities = liv.module.sim(
                stage_emb.expand(len(visual_demo), -1),
                demo_embs
            )

        # Generate predictions
        pred_ref = similarities.argmax().item() + 1  # 1-based
        total_steps = item['total_steps']
        pred_score = (pred_ref - 1) / total_steps  # ref=1 corresponds to 0%

        return {
            'pred_ref': pred_ref,
            'pred_score': pred_score,
            'similarities': similarities.cpu().tolist(),
            'status': 'success'
        }

    except Exception as e:
        return {
            'pred_ref': None,
            'pred_score': None,
            'similarities': None,
            'status': 'failed',
            'error': str(e)
        }


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate evaluation metrics from results.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with aggregated metrics
    """
    ref_errors = []
    score_errors = []
    ref_correct = 0

    for r in results:
        gt_ref = r.get('gt_ref')
        gt_score = r.get('gt_score')
        pred_ref = r.get('pred_ref')
        pred_score = r.get('pred_score')

        # Ref Error (only when both are numeric)
        if gt_ref is not None and pred_ref is not None:
            ref_error = abs(gt_ref - pred_ref)
            ref_errors.append(ref_error)
            if ref_error == 0:
                ref_correct += 1

        # Score Error (only when both are numeric)
        if gt_score is not None and pred_score is not None:
            score_errors.append(abs(gt_score - pred_score))

    return {
        'mean_ref_error': float(np.mean(ref_errors)) if ref_errors else float('inf'),
        'mean_score_error': float(np.mean(score_errors)) if score_errors else float('inf'),
        'ref_accuracy': ref_correct / len(ref_errors) if ref_errors else 0,
        'total_samples': len(results),
        'valid_ref_samples': len(ref_errors),
        'valid_score_samples': len(score_errors),
        'success_count': sum(1 for r in results if r.get('status') == 'success'),
        'failed_count': sum(1 for r in results if r.get('status') == 'failed')
    }


def get_timestamp_output_path(output_dir: str, task_name: str) -> str:
    """
    Generate output file path with timestamp.

    Args:
        output_dir: Output directory
        task_name: Task name for filename

    Returns:
        Full path to output file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"{timestamp}_{task_name}.jsonl")


def run_evaluation(args):
    """Main evaluation function."""

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"Loading LIV model from {args.model_path}...")
    liv = load_liv_from_path(args.model_path)
    liv.eval()
    liv.to(device)
    print("Model loaded successfully!")

    # Setup transform
    transform = T.Compose([T.ToTensor()])

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    data = load_visual_demo_dataset(
        args.dataset_path,
        num_inferences=1,  # No expansion for LIV
        image_root=args.image_root
    )

    if args.limit > 0:
        data = data[:args.limit]
        print(f"Limited to {args.limit} samples")

    print(f"Total samples to evaluate: {len(data)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = get_timestamp_output_path(args.output_dir, "visual_demo")
    print(f"Output will be saved to: {output_file}")

    # Run evaluation
    results = []

    for item in tqdm(data, desc="Evaluating"):
        # Validate image paths
        is_valid, error_msg = validate_image_paths(item)
        if not is_valid:
            result = {
                'id': item.get('id', ''),
                'pred_ref': None,
                'pred_score': None,
                'gt_ref': item.get('closest_idx'),
                'gt_score': item.get('progress_score'),
                'ref_error': float('inf'),
                'score_error': float('inf'),
                'similarities': None,
                'status': 'failed',
                'error': error_msg,
                'meta_data': {
                    'task_goal': item.get('task_goal', ''),
                    'visual_demo': item.get('visual_demo', []),
                    'total_steps': item.get('total_steps', 0),
                    'stage_to_estimate': item.get('stage_to_estimate', ''),
                }
            }
            results.append(result)
            continue

        # Evaluate sample
        eval_result = evaluate_visual_demo_sample(liv, item, device, transform)

        # Calculate errors
        gt_ref = item.get('closest_idx')
        gt_score = item.get('progress_score')

        ref_error = float('inf')
        score_error = float('inf')

        if gt_ref is not None and eval_result['pred_ref'] is not None:
            ref_error = abs(gt_ref - eval_result['pred_ref'])

        if gt_score is not None and eval_result['pred_score'] is not None:
            score_error = abs(gt_score - eval_result['pred_score'])

        result = {
            'id': item.get('id', ''),
            'pred_ref': eval_result['pred_ref'],
            'pred_score': eval_result['pred_score'],
            'gt_ref': gt_ref,
            'gt_score': gt_score,
            'ref_error': ref_error,
            'score_error': score_error,
            'similarities': eval_result['similarities'],
            'status': eval_result['status'],
            'meta_data': {
                'task_goal': item.get('task_goal', ''),
                'visual_demo': item.get('visual_demo', []),
                'total_steps': item.get('total_steps', 0),
                'stage_to_estimate': item.get('stage_to_estimate', ''),
            }
        }

        if 'error' in eval_result:
            result['error'] = eval_result['error']

        results.append(result)

    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Calculate and print metrics
    metrics = calculate_metrics(results)

    print("\n" + "=" * 60)
    print("LIV VISUAL DEMO EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Success: {metrics['success_count']}")
    print(f"Failed: {metrics['failed_count']}")
    print(f"Valid ref samples: {metrics['valid_ref_samples']}")
    print(f"Valid score samples: {metrics['valid_score_samples']}")
    print(f"\nMetrics:")
    print(f"  Mean Ref Error: {metrics['mean_ref_error']:.4f}")
    print(f"  Mean Score Error: {metrics['mean_score_error']:.4f}")
    print(f"  Ref Accuracy (exact match): {metrics['ref_accuracy']*100:.2f}%")
    print("=" * 60)

    # Save summary
    summary_file = output_file.replace('.jsonl', '_summary.json')
    summary = {
        'model': 'LIV',
        'task': 'visual_demo',
        'timestamp': datetime.now().isoformat(),
        **metrics,
        'dataset_path': args.dataset_path,
        'model_path': args.model_path,
        'image_root': args.image_root,
        'output_file': output_file
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {summary_file}")

    return results, metrics


def main():
    parser = argparse.ArgumentParser(
        description="LIV Model Evaluation for Visual Demo Progress Estimation"
    )

    # Required arguments
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to LIV model directory"
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True,
        help="Path to visual demo dataset (JSONL format)"
    )

    # Optional arguments
    parser.add_argument(
        "--image-root", type=str, default=None,
        help="Root directory for image paths"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (not used currently, for future optimization)"
    )
    parser.add_argument(
        "--limit", type=int, default=-1,
        help="Limit number of samples to process (-1 for all)"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        sys.exit(1)

    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset not found: {args.dataset_path}")
        sys.exit(1)

    # Run evaluation
    run_evaluation(args)


if __name__ == "__main__":
    main()
