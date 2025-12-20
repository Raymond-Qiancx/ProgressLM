#!/usr/bin/env python3
"""
VOC (Trajectory Order Consistency) Calculator with Auto Format Detection

Automatically detects the format based on available fields:
- Format 1: "score" field (e.g., 3B RL results)
- Format 2: "predicted_score" field (e.g., 72B nothink results)

Usage:
    python calc_voc_auto.py <jsonl_file> [--output <output_file>] [--format <1|2|auto>]
    python calc_voc_auto.py results.jsonl
    python calc_voc_auto.py results.jsonl --format 1
    python calc_voc_auto.py results.jsonl --output voc_results.json
    python /projects/p32958/chengxuan/ProgressLM/eval/eval_results/voc/calc_voc_auto.py 
"""

import json
import argparse
import os
from collections import defaultdict
from scipy.stats import spearmanr
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def parse_score(score_str):
    """Parse score string to float [0, 1]"""
    if score_str is None or score_str == '' or str(score_str).lower() == 'n/a':
        return 0.0
    if isinstance(score_str, (int, float)):
        return float(score_str) if score_str <= 1.0 else score_str / 100.0
    score_str = str(score_str).strip()
    if score_str.endswith('%'):
        return float(score_str[:-1]) / 100.0
    try:
        val = float(score_str)
        return val if val <= 1.0 else val / 100.0
    except:
        return 0.0


def detect_format(filepath):
    """
    Detect the format of the JSONL file.

    Returns:
        'format1' if 'score' field exists
        'format2' if 'predicted_score' field exists
        None if neither
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if first_line.strip():
            data = json.loads(first_line)
            if 'score' in data and 'predicted_score' not in data:
                return 'format1'
            elif 'predicted_score' in data:
                return 'format2'
            elif 'score' in data:
                return 'format1'
    return None


def calculate_voc(filepath, format_type='auto'):
    """
    Calculate VOC using Spearman correlation.

    Uses scipy.stats.spearmanr directly on scores, which handles ties
    by assigning average ranks to equal values.

    Args:
        filepath: Path to the JSONL file
        format_type: 'format1', 'format2', or 'auto'

    Returns:
        Dictionary with VOC statistics
    """
    # Auto-detect format if needed
    if format_type == 'auto':
        format_type = detect_format(filepath)
        if format_type is None:
            return {'error': 'Could not detect format'}

    # Determine field name based on format
    if format_type == 'format1' or format_type == '1':
        pred_field = 'score'
        format_name = 'format1 (score field)'
    else:
        pred_field = 'predicted_score'
        format_name = 'format2 (predicted_score field)'

    # Load results
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    # Group by trajectory ID
    trajectories = defaultdict(list)
    for res in results:
        meta = res.get('meta_data', {})
        traj_id = meta.get('id', '')
        gt_score = meta.get('progress_score')

        if gt_score is not None:
            pred_str = res.get(pred_field, '')
            pred_score = parse_score(pred_str)

            trajectories[traj_id].append({
                'gt_score': gt_score,
                'pred_score': pred_score
            })

    # Calculate VOC for each trajectory
    voc_values = []
    skipped_single = 0
    skipped_constant_pred = 0
    skipped_constant_gt = 0

    for traj_id, samples in trajectories.items():
        if len(samples) <= 1:
            skipped_single += 1
            continue

        gt_scores = [s['gt_score'] for s in samples]
        pred_scores = [s['pred_score'] for s in samples]

        # Skip if all predictions are the same (no variation)
        if len(set(pred_scores)) == 1:
            skipped_constant_pred += 1
            continue

        # Skip if all GT scores are the same
        if len(set(gt_scores)) == 1:
            skipped_constant_gt += 1
            continue

        # Calculate Spearman correlation directly on scores
        # scipy handles ties by assigning average ranks
        correlation, p_value = spearmanr(gt_scores, pred_scores)
        if not np.isnan(correlation):
            voc_values.append(correlation)

    # Calculate statistics
    result = {
        'file': filepath,
        'format': format_name,
        'pred_field': pred_field,
        'total_samples': len(results),
        'total_trajectories': len(trajectories),
        'valid_trajectories': len(voc_values),
        'skipped_single_sample': skipped_single,
        'skipped_constant_pred': skipped_constant_pred,
        'skipped_constant_gt': skipped_constant_gt,
    }

    if voc_values:
        result.update({
            'voc_mean': float(np.mean(voc_values)),
            'voc_std': float(np.std(voc_values)),
            'voc_min': float(np.min(voc_values)),
            'voc_max': float(np.max(voc_values)),
            'voc_median': float(np.median(voc_values)),
        })
    else:
        result.update({
            'voc_mean': None,
            'voc_std': None,
            'voc_min': None,
            'voc_max': None,
            'voc_median': None,
        })

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Calculate VOC with auto format detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('jsonl_file', type=str, help='Path to the JSONL results file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file path (optional)')
    parser.add_argument('--format', '-f', type=str, default='auto',
                        choices=['auto', '1', '2', 'format1', 'format2'],
                        help='Format type (default: auto)')

    args = parser.parse_args()

    if not os.path.exists(args.jsonl_file):
        print(f"Error: File not found: {args.jsonl_file}")
        return 1

    # Calculate VOC
    result = calculate_voc(args.jsonl_file, args.format)

    if 'error' in result:
        print(f"Error: {result['error']}")
        return 1

    # Print results
    print("=" * 60)
    print("VOC Calculation Results")
    print("=" * 60)
    print(f"File: {result['file']}")
    print(f"Format: {result['format']}")
    print(f"Total samples: {result['total_samples']}")
    print(f"Total trajectories: {result['total_trajectories']}")
    print(f"Valid trajectories: {result['valid_trajectories']}")
    print(f"Skipped (single sample): {result['skipped_single_sample']}")
    print(f"Skipped (constant pred): {result['skipped_constant_pred']}")
    print(f"Skipped (constant GT): {result['skipped_constant_gt']}")
    print("-" * 60)

    if result['voc_mean'] is not None:
        print(f"Mean VOC:   {result['voc_mean']:.4f}")
        print(f"Std VOC:    {result['voc_std']:.4f}")
        print(f"Min VOC:    {result['voc_min']:.4f}")
        print(f"Max VOC:    {result['voc_max']:.4f}")
        print(f"Median VOC: {result['voc_median']:.4f}")
    else:
        print("VOC: N/A (no valid trajectories)")
    print("=" * 60)

    # Save to file if specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == '__main__':
    exit(main())
