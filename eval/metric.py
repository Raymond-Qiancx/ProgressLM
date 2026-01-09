#!/usr/bin/env python3
"""
Batch Calculation Script for Model Evaluation Metrics

This script processes all model results from all_results_paths.txt and calculates:
- VOC (Trajectory Order Consistency) for all models
- SCORE error for all models
- REF error for think models only

Uses multiprocessing for faster execution.
"""

import json
import re
import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.stats import spearmanr
from multiprocessing import Pool, cpu_count
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ==================== Score and Ref Parsing ====================

def parse_score(score_str: Optional[str]) -> Optional[float]:
    """Parse score string to float [0, 1]. Returns None for N/A."""
    if score_str is None or score_str == '':
        return None

    score_str_lower = str(score_str).strip().lower()

    # Check for N/A variants
    if score_str_lower in ('n/a', 'na', 'none', 'null', '-'):
        return None

    # Handle numeric values
    if isinstance(score_str, (int, float)):
        val = float(score_str)
        return val if val <= 1.0 else val / 100.0

    score_str = str(score_str).strip()

    # Handle percentage format
    if score_str.endswith('%'):
        try:
            return float(score_str[:-1]) / 100.0
        except:
            return None

    # Try parsing as float
    try:
        val = float(score_str)
        return val if val <= 1.0 else val / 100.0
    except:
        return None


def parse_ref(ref_str: Optional[str]) -> Optional[int]:
    """Parse ref string to int."""
    if ref_str is None or ref_str == 'n/a' or ref_str == '':
        return None
    try:
        return int(ref_str)
    except (ValueError, TypeError):
        return None


# ==================== Error Calculations ====================

def calculate_score_error(gt: float, pred: float) -> float:
    """Normalized score error: |gt - pred| / max(gt, 1 - gt)"""
    max_possible = max(gt, 1.0 - gt)
    if max_possible == 0:
        return 0.0
    return abs(gt - pred) / max_possible


def calculate_ref_error(gt_ref: int, pred_ref: int, num_demos: int) -> float:
    """Normalized ref error: |gt_ref - pred_ref| / max(gt_ref - 1, num_demos - gt_ref)"""
    max_possible = max(gt_ref - 1, num_demos - gt_ref)
    if max_possible == 0:
        return 0.0
    return abs(gt_ref - pred_ref) / max_possible


# ==================== VOC Calculation ====================

def calculate_voc(results: List[Dict], pred_field: str) -> Dict:
    """Calculate VOC using Spearman correlation with sample-level N/A filtering."""
    # Group by trajectory ID
    trajectories = defaultdict(list)
    total_samples = 0
    total_na_samples = 0

    for res in results:
        meta = res.get('meta_data', {})
        traj_id = meta.get('id', '')
        gt_score = meta.get('progress_score')

        if gt_score is not None:
            total_samples += 1
            pred_str = res.get(pred_field, '')
            pred_score = parse_score(pred_str)

            if pred_score is None:
                total_na_samples += 1

            trajectories[traj_id].append({
                'gt_score': gt_score,
                'pred_score': pred_score
            })

    # Calculate VOC for each trajectory
    voc_values = []
    skipped_all_na = 0
    skipped_single = 0
    skipped_constant_pred = 0
    skipped_constant_gt = 0

    for traj_id, samples in trajectories.items():
        # Filter out samples with N/A predictions
        valid_samples = [s for s in samples if s['pred_score'] is not None]

        if len(valid_samples) == 0:
            skipped_all_na += 1
            continue

        if len(valid_samples) <= 1:
            skipped_single += 1
            continue

        gt_scores = [s['gt_score'] for s in valid_samples]
        pred_scores = [s['pred_score'] for s in valid_samples]

        # Skip if all predictions are the same
        if len(set(pred_scores)) == 1:
            skipped_constant_pred += 1
            continue

        # Skip if all GT scores are the same
        if len(set(gt_scores)) == 1:
            skipped_constant_gt += 1
            continue

        # Calculate Spearman correlation
        correlation, p_value = spearmanr(gt_scores, pred_scores)
        if not np.isnan(correlation):
            voc_values.append(correlation)

    valid_samples = total_samples - total_na_samples
    sample_coverage = valid_samples / total_samples if total_samples > 0 else 0.0

    result = {
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'sample_coverage': sample_coverage,
        'total_trajectories': len(trajectories),
        'valid_trajectories': len(voc_values),
    }

    if voc_values:
        result.update({
            'voc_mean': float(np.mean(voc_values)),
            'voc_std': float(np.std(voc_values)),
        })
    else:
        result.update({
            'voc_mean': None,
            'voc_std': None,
        })

    return result


# ==================== Error Calculations ====================

def calculate_errors(results: List[Dict], format_type: str) -> Dict:
    """Calculate score and ref errors."""
    score_errors = []
    ref_errors = []

    for r in results:
        meta = r.get('meta_data', {})
        gt_score = meta.get('progress_score')
        gt_ref = meta.get('closest_idx')
        num_demos = len(meta.get('visual_demo', meta.get('text_demo', [])))

        # Parse predicted values based on format
        if format_type == 'ref_score':
            pred_score = parse_score(r.get('score'))
            pred_ref = parse_ref(r.get('ref'))
        else:  # score_only
            pred_score = parse_score(r.get('predicted_score'))
            pred_ref = None

        # Calculate score errors
        if gt_score is not None and pred_score is not None:
            score_err = calculate_score_error(gt_score, pred_score)
            if score_err != float('inf'):
                score_errors.append(score_err)

        # Calculate ref errors (only for ref_score format)
        if format_type == 'ref_score' and gt_ref is not None and pred_ref is not None:
            ref_err = calculate_ref_error(gt_ref, pred_ref, num_demos)
            if ref_err != float('inf'):
                ref_errors.append(ref_err)

    result = {}

    if score_errors:
        result['score_error_mean'] = float(np.mean(score_errors))
        result['score_error_std'] = float(np.std(score_errors))
    else:
        result['score_error_mean'] = None
        result['score_error_std'] = None

    if ref_errors:
        result['ref_error_mean'] = float(np.mean(ref_errors))
        result['ref_error_std'] = float(np.std(ref_errors))
    else:
        result['ref_error_mean'] = None
        result['ref_error_std'] = None

    return result


# ==================== Format Detection ====================

def detect_format(first_record: Dict) -> Tuple[str, str]:
    """
    Detect JSONL format.
    Returns: (format_type, pred_field)
        format_type: 'ref_score' or 'score_only'
        pred_field: 'score' or 'predicted_score'
    """
    if 'ref' in first_record and 'score' in first_record:
        return 'ref_score', 'score'
    elif 'predicted_score' in first_record:
        return 'score_only', 'predicted_score'
    elif 'score' in first_record:
        return 'score_only', 'score'
    else:
        raise ValueError("Unknown JSONL format")


# ==================== Process Single Model ====================

def process_single_model(args: Tuple[Dict, int, int]) -> Dict:
    """Process a single model result file."""
    model_info, idx, total = args

    try:
        filepath = model_info['path']

        # Check if file exists
        if not os.path.exists(filepath):
            return {
                'model_info': model_info,
                'error': f'File not found: {filepath}',
                'idx': idx,
                'total': total
            }

        # Load results
        results = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        if not results:
            return {
                'model_info': model_info,
                'error': 'Empty results file',
                'idx': idx,
                'total': total
            }

        # Detect format
        format_type, pred_field = detect_format(results[0])

        # Calculate metrics
        voc_metrics = calculate_voc(results, pred_field)
        error_metrics = calculate_errors(results, format_type)

        # Combine results
        result = {
            'model_info': model_info,
            'format_type': format_type,
            'voc': voc_metrics,
            'errors': error_metrics,
            'idx': idx,
            'total': total,
            'success': True
        }

        print(f"[{idx+1}/{total}] Processed: {model_info['model_family']}/{model_info['task']}/{model_info['model_type']}_{model_info['model_size']}")

        return result

    except Exception as e:
        return {
            'model_info': model_info,
            'error': str(e),
            'idx': idx,
            'total': total
        }


# ==================== Parse Input File ====================

def parse_input_file(input_path: str) -> List[Dict]:
    """Parse all_results_paths.txt and extract model metadata."""
    models = []

    current_family = None
    current_task = None
    current_type = None
    current_size = None

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Parse model family
            if line.startswith('# Model Family:'):
                current_family = line.split(':', 1)[1].strip()
                continue

            # Parse task
            if line.startswith('# Task:'):
                task_match = re.search(r'# Task:\s*(\S+)', line)
                if task_match:
                    current_task = task_match.group(1)
                continue

            # Parse type and size
            if line.startswith('# type:'):
                type_match = re.search(r'# type:\s*(\S+),\s*size:\s*(\S+)', line)
                if type_match:
                    current_type = type_match.group(1)
                    current_size = type_match.group(2)
                continue

            # Skip comment lines
            if line.startswith('#'):
                continue

            # This should be a file path
            if line.startswith('/'):
                if current_family and current_task and current_type and current_size:
                    # Skip negative tasks (edit_nega, text_nega, etc.)
                    if 'nega' in current_task:
                        continue

                    # Rename new_pro_bench to Qwen2.5VL
                    family_name = 'Qwen2.5VL' if current_family == 'new_pro_bench' else current_family

                    models.append({
                        'model_family': family_name,
                        'task': current_task,
                        'model_type': current_type,
                        'model_size': current_size,
                        'path': line
                    })

    return models


# ==================== Write Results ====================

def write_results(results: List[Dict], output_path: str, summary_path: str):
    """Write results to TXT file."""

    # Separate successful and failed results
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("Batch Calculation Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Models Processed: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")

        # Summary table (tab-separated)
        f.write("=" * 80 + "\n")
        f.write("SUMMARY TABLE (Tab-Separated)\n")
        f.write("=" * 80 + "\n")
        f.write("ModelFamily\tTask\tType\tSize\tVOC_Mean\tVOC_Std\tScore_Mean\tScore_Std\tRef_Mean\tRef_Std\tSamples\n")

        for r in successful:
            mi = r['model_info']
            voc = r['voc']
            err = r['errors']

            voc_mean = f"{voc['voc_mean']:.4f}" if voc['voc_mean'] is not None else "N/A"
            voc_std = f"{voc['voc_std']:.4f}" if voc['voc_std'] is not None else "N/A"
            score_mean = f"{err['score_error_mean']:.4f}" if err['score_error_mean'] is not None else "N/A"
            score_std = f"{err['score_error_std']:.4f}" if err['score_error_std'] is not None else "N/A"
            ref_mean = f"{err['ref_error_mean']:.4f}" if err['ref_error_mean'] is not None else "N/A"
            ref_std = f"{err['ref_error_std']:.4f}" if err['ref_error_std'] is not None else "N/A"

            f.write(f"{mi['model_family']}\t{mi['task']}\t{mi['model_type']}\t{mi['model_size']}\t")
            f.write(f"{voc_mean}\t{voc_std}\t{score_mean}\t{score_std}\t{ref_mean}\t{ref_std}\t")
            f.write(f"{voc['valid_samples']}/{voc['total_samples']}\n")

        f.write("\n")

        # Detailed results
        f.write("=" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for r in successful:
            mi = r['model_info']
            voc = r['voc']
            err = r['errors']

            f.write("-" * 80 + "\n")
            f.write(f"Model: {mi['model_family']}/{mi['task']}/{mi['model_type']}_{mi['model_size']}\n")
            f.write("-" * 80 + "\n")

            # VOC
            f.write("VOC:\n")
            if voc['voc_mean'] is not None:
                f.write(f"  Mean: {voc['voc_mean']:.4f}  Std: {voc['voc_std']:.4f}\n")
            else:
                f.write(f"  N/A (no valid trajectories)\n")

            # Score Error
            f.write("Score Error:\n")
            if err['score_error_mean'] is not None:
                f.write(f"  Mean: {err['score_error_mean']:.4f}  Std: {err['score_error_std']:.4f}\n")
            else:
                f.write(f"  N/A\n")

            # Ref Error
            f.write("Ref Error:\n")
            if err['ref_error_mean'] is not None:
                f.write(f"  Mean: {err['ref_error_mean']:.4f}  Std: {err['ref_error_std']:.4f}\n")
            else:
                f.write(f"  N/A\n")

            # Samples
            coverage = voc['sample_coverage'] * 100
            f.write(f"Samples: {voc['valid_samples']}/{voc['total_samples']} ({coverage:.2f}%)\n")
            f.write(f"Trajectories: {voc['valid_trajectories']}/{voc['total_trajectories']}\n")
            f.write(f"Format: {r['format_type']}\n")
            f.write(f"Path: {mi['path']}\n")
            f.write("\n")

        # Failed results
        if failed:
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("FAILED RESULTS\n")
            f.write("=" * 80 + "\n\n")

            for r in failed:
                mi = r['model_info']
                f.write(f"Model: {mi['model_family']}/{mi['task']}/{mi['model_type']}_{mi['model_size']}\n")
                f.write(f"Error: {r.get('error', 'Unknown error')}\n")
                f.write(f"Path: {mi['path']}\n")
                f.write("\n")

    # Write summary to separate file
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("ModelFamily\tTask\tType\tSize\tVOC_Mean\tVOC_Std\tScore_Mean\tScore_Std\tRef_Mean\tRef_Std\tSamples\n")
        for r in successful:
            mi = r['model_info']
            voc = r['voc']
            err = r['errors']

            voc_mean = f"{voc['voc_mean']:.4f}" if voc['voc_mean'] is not None else "N/A"
            voc_std = f"{voc['voc_std']:.4f}" if voc['voc_std'] is not None else "N/A"
            score_mean = f"{err['score_error_mean']:.4f}" if err['score_error_mean'] is not None else "N/A"
            score_std = f"{err['score_error_std']:.4f}" if err['score_error_std'] is not None else "N/A"
            ref_mean = f"{err['ref_error_mean']:.4f}" if err['ref_error_mean'] is not None else "N/A"
            ref_std = f"{err['ref_error_std']:.4f}" if err['ref_error_std'] is not None else "N/A"

            f.write(f"{mi['model_family']}\t{mi['task']}\t{mi['model_type']}\t{mi['model_size']}\t")
            f.write(f"{voc_mean}\t{voc_std}\t{score_mean}\t{score_std}\t{ref_mean}\t{ref_std}\t")
            f.write(f"{voc['valid_samples']}/{voc['total_samples']}\n")


# ==================== Main ====================

def main():
    # Paths
    input_path = '/projects/p32958/chengxuan/ProgressLM/eval/eval_results/map/all_results_paths.txt'
    output_dir = '/projects/p32958/chengxuan/ProgressLM/eval/eval_results/batch'
    output_path = os.path.join(output_dir, 'batch_results.txt')
    summary_path = os.path.join(output_dir, 'summary_table.txt')

    print("=" * 80)
    print("Batch Calculation Script for Model Evaluation Metrics")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")
    print()

    # Parse input file
    print("Parsing input file...")
    models = parse_input_file(input_path)
    print(f"Found {len(models)} models to process\n")

    # Process models in parallel
    print("Processing models...")
    num_workers = min(cpu_count(), 16)  # Limit to 16 workers
    print(f"Using {num_workers} workers\n")

    args = [(model, idx, len(models)) for idx, model in enumerate(models)]

    with Pool(num_workers) as pool:
        results = pool.map(process_single_model, args)

    # Write results
    print("\nWriting results...")
    write_results(results, output_path, summary_path)

    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful

    print("\n" + "=" * 80)
    print("Batch Processing Complete")
    print("=" * 80)
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults written to: {output_path}")
    print(f"Summary table written to: {summary_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
