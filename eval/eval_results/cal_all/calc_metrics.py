#!/usr/bin/env python3
"""
Combined Metrics Calculator: VOC + Normalized Error

Calculates:
1. VOC (Trajectory Order Consistency) - Spearman correlation
2. Normalized Score Error: |gt - pred| / max(gt, 1 - gt)
3. Normalized Ref Error: |gt_ref - pred_ref| / max(gt_ref - 1, num_demos - gt_ref)

Supports multiple JSONL formats:
- ref_score: has 'ref' and 'score' fields
- score_only: has 'predicted_score' field
- score_gt: has 'score' and 'ground_truth_score' fields
- score_meta: has 'score' field with progress_score in meta_data

Usage:
    python calc_metrics.py <results.jsonl>
"""

import json
import argparse
import os
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


def detect_format(first_record):
    """Detect JSONL format based on field names."""
    if 'ref' in first_record and 'score' in first_record:
        return 'ref_score'
    elif 'predicted_score' in first_record:
        return 'score_only'
    elif 'score' in first_record and 'ground_truth_score' in first_record:
        return 'score_gt'
    elif 'score' in first_record and 'meta_data' in first_record:
        return 'score_meta'
    else:
        raise ValueError("Unknown JSONL format")


def parse_score(score_str):
    """Parse score string to float [0, 1]. Returns None for invalid values."""
    if score_str is None or score_str == '':
        return None

    score_str_lower = str(score_str).strip().lower()
    if score_str_lower in ('n/a', 'na', 'none', 'null', '-'):
        return None

    if isinstance(score_str, (int, float)):
        val = float(score_str)
        return val if val <= 1.0 else val / 100.0

    score_str = str(score_str).strip()
    if score_str.endswith('%'):
        try:
            return float(score_str[:-1]) / 100.0
        except:
            return None

    try:
        val = float(score_str)
        return val if val <= 1.0 else val / 100.0
    except:
        return None


def parse_ref(ref_str):
    """Parse ref string to int."""
    if ref_str is None or ref_str == 'n/a' or ref_str == '':
        return None
    try:
        return int(ref_str)
    except (ValueError, TypeError):
        return None


def calculate_new_score_error(gt, pred):
    """Normalized score error: |gt - pred| / max(gt, 1 - gt)"""
    max_possible = max(gt, 1.0 - gt)
    if max_possible == 0:
        return 0.0
    return abs(gt - pred) / max_possible


def calculate_new_ref_error(gt_ref, pred_ref, num_demos):
    """Normalized ref error: |gt_ref - pred_ref| / max(gt_ref - 1, num_demos - gt_ref)"""
    max_possible = max(gt_ref - 1, num_demos - gt_ref)
    if max_possible == 0:
        return 0.0
    return abs(gt_ref - pred_ref) / max_possible


def load_results(filepath):
    """Load and process results from JSONL file."""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        raise ValueError("Empty results file")

    format_type = detect_format(results[0])
    return format_type, results


def calculate_voc(results, format_type):
    """Calculate VOC using Spearman correlation."""
    # Determine prediction field
    if format_type in ('ref_score', 'score_gt', 'score_meta'):
        pred_field = 'score'
    else:
        pred_field = 'predicted_score'

    # Group by trajectory
    trajectories = defaultdict(list)

    for res in results:
        meta = res.get('meta_data', {})
        traj_id = meta.get('id', '')

        if format_type == 'score_gt':
            gt_score = parse_score(res.get('ground_truth_score'))
        else:
            gt_score = meta.get('progress_score')

        if gt_score is not None:
            pred_score = parse_score(res.get(pred_field, ''))
            trajectories[traj_id].append({
                'gt_score': gt_score,
                'pred_score': pred_score
            })

    # Calculate VOC for each trajectory
    voc_values = []
    for traj_id, samples in trajectories.items():
        valid_samples = [s for s in samples if s['pred_score'] is not None]

        if len(valid_samples) <= 1:
            continue

        gt_scores = [s['gt_score'] for s in valid_samples]
        pred_scores = [s['pred_score'] for s in valid_samples]

        if len(set(pred_scores)) == 1 or len(set(gt_scores)) == 1:
            continue

        correlation, _ = spearmanr(gt_scores, pred_scores)
        if not np.isnan(correlation):
            voc_values.append(correlation)

    return voc_values


def calculate_errors(results, format_type):
    """Calculate normalized errors for score and ref. Also returns N/A stats."""
    new_score_errors = []
    new_ref_errors = []
    total_score_samples = 0
    na_score_samples = 0
    total_ref_samples = 0
    na_ref_samples = 0

    for r in results:
        meta = r.get('meta_data', {})
        num_demos = len(meta.get('text_demo', []))

        # Get ground truth
        if format_type == 'score_gt':
            gt_score = parse_score(r.get('ground_truth_score'))
            gt_ref = None
        else:
            gt_score = meta.get('progress_score')
            gt_ref = meta.get('closest_idx')

        # Get predictions based on format
        if format_type == 'ref_score':
            pred_score = parse_score(r.get('score'))
            pred_ref = parse_ref(r.get('ref'))
        elif format_type == 'score_only':
            pred_score = parse_score(r.get('predicted_score'))
            pred_ref = None
        else:  # score_gt or score_meta
            pred_score = parse_score(r.get('score'))
            pred_ref = None

        # Count Score N/A rate
        if gt_score is not None:
            total_score_samples += 1
            if pred_score is None:
                na_score_samples += 1

        # Count Ref N/A rate (only for ref_score format)
        if format_type == 'ref_score' and gt_ref is not None:
            total_ref_samples += 1
            if pred_ref is None:
                na_ref_samples += 1

        # Calculate score error
        if gt_score is not None and pred_score is not None:
            new_err = calculate_new_score_error(gt_score, pred_score)
            new_score_errors.append(new_err)

        # Calculate ref error (only for ref_score format)
        if format_type == 'ref_score' and gt_ref is not None and pred_ref is not None:
            new_err = calculate_new_ref_error(gt_ref, pred_ref, num_demos)
            new_ref_errors.append(new_err)

    score_na_rate = na_score_samples / total_score_samples if total_score_samples > 0 else 0.0
    ref_na_rate = na_ref_samples / total_ref_samples if total_ref_samples > 0 else 0.0
    return new_score_errors, new_ref_errors, score_na_rate, ref_na_rate


def main():
    parser = argparse.ArgumentParser(
        description="Calculate VOC and Normalized Error metrics"
    )
    parser.add_argument("input", type=str, help="Path to the results JSONL file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return 1

    # Load results
    format_type, results = load_results(args.input)

    # Calculate normalized errors (includes N/A rates)
    new_score_errors, new_ref_errors, score_na_rate, ref_na_rate = calculate_errors(results, format_type)

    # Print N/A rates first
    print(f"Score N/A: {score_na_rate:.2%}")
    if format_type == 'ref_score':
        print(f"Ref N/A: {ref_na_rate:.2%}")

    # Calculate VOC
    voc_values = calculate_voc(results, format_type)
    if voc_values:
        print(f"VOC   - Mean: {np.mean(voc_values):.4f}, Std: {np.std(voc_values):.4f}")

    new_scores = [x for x in new_score_errors if x != float('inf')]
    if new_scores:
        print(f"Score - Mean: {np.mean(new_scores):.4f}, Std: {np.std(new_scores):.4f}")

    if new_ref_errors:
        print(f"Ref   - Mean: {np.mean(new_ref_errors):.4f}, Std: {np.std(new_ref_errors):.4f}")

    return 0


if __name__ == "__main__":
    exit(main())
