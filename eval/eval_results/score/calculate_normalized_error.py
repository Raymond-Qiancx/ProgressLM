#!/usr/bin/env python3
"""
Normalized Error Calculator for Progress Estimation Results

Supports two JSONL formats:
1. Format with ref+score (e.g., rl_3b): has 'ref', 'score' fields
2. Format with score only (e.g., nothink_72b): has 'predicted_score' field

Normalized Error Formulas:
- Score Error: |gt - pred| / max(gt, 1 - gt)
- Ref Error: |gt_ref - pred_ref| / max(gt_ref - 1, num_demos - gt_ref)

Usage:
    python calculate_normalized_error.py <results.jsonl>
"""

import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def detect_format(first_record: Dict) -> str:
    """
    Detect JSONL format based on field names.

    Returns:
        'ref_score': Format with both ref and score (e.g., rl_3b)
        'score_only': Format with only score (e.g., nothink_72b)
        'score_gt': Format with score and ground_truth_score (e.g., gpt5_nothink)
    """
    if 'ref' in first_record and 'score' in first_record:
        return 'ref_score'
    elif 'predicted_score' in first_record:
        return 'score_only'
    elif 'score' in first_record and 'ground_truth_score' in first_record:
        return 'score_gt'
    elif 'score' in first_record and 'meta_data' in first_record:
        # Format with score field and progress_score in meta_data
        return 'score_meta'
    else:
        raise ValueError("Unknown JSONL format. Expected 'ref'+'score', 'predicted_score', or 'score'+'ground_truth_score' fields.")


def parse_score(score_str: Optional[str]) -> Optional[float]:
    """Parse score string to float (0-1 range)."""
    if score_str is None or score_str == 'n/a' or score_str == '':
        return None
    try:
        if isinstance(score_str, (int, float)):
            val = float(score_str)
        elif score_str.endswith('%'):
            val = float(score_str[:-1]) / 100.0
        else:
            val = float(score_str)

        # Normalize if > 1 (assume percentage)
        if val > 1.0:
            val = val / 100.0
        return val
    except (ValueError, AttributeError):
        return None


def parse_ref(ref_str: Optional[str]) -> Optional[int]:
    """Parse ref string to int."""
    if ref_str is None or ref_str == 'n/a' or ref_str == '':
        return None
    try:
        return int(ref_str)
    except (ValueError, TypeError):
        return None


def calculate_old_score_error(gt: float, pred: float) -> float:
    """Old method: |gt - pred| / gt"""
    if gt == 0:
        return 0.0 if pred == 0 else float('inf')
    return abs(gt - pred) / gt


def calculate_new_score_error(gt: float, pred: float) -> float:
    """New method (normalized): |gt - pred| / max(gt, 1 - gt)"""
    max_possible = max(gt, 1.0 - gt)
    if max_possible == 0:
        return 0.0
    return abs(gt - pred) / max_possible


def calculate_old_ref_error(gt_ref: int, pred_ref: int) -> float:
    """Old method: |gt_ref - pred_ref|"""
    return float(abs(gt_ref - pred_ref))


def calculate_new_ref_error(gt_ref: int, pred_ref: int, num_demos: int) -> float:
    """New method (normalized): |gt_ref - pred_ref| / max(gt_ref - 1, num_demos - gt_ref)"""
    max_possible = max(gt_ref - 1, num_demos - gt_ref)
    if max_possible == 0:
        return 0.0
    return abs(gt_ref - pred_ref) / max_possible


def process_results(filepath: str) -> Tuple[str, List[Dict]]:
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


def analyze_results(format_type: str, results: List[Dict]) -> Dict:
    """Analyze results and calculate errors."""

    # Storage for errors
    old_score_errors = []
    new_score_errors = []
    old_ref_errors = []
    new_ref_errors = []

    # For grouping analysis
    score_groups = defaultdict(list)
    ref_groups = defaultdict(list)

    for r in results:
        meta = r.get('meta_data', {})
        gt_score = meta.get('progress_score')
        gt_ref = meta.get('closest_idx')
        num_demos = len(meta.get('text_demo', []))

        # Parse predicted values based on format
        if format_type == 'ref_score':
            pred_score = parse_score(r.get('score'))
            pred_ref = parse_ref(r.get('ref'))
        elif format_type == 'score_only':
            pred_score = parse_score(r.get('predicted_score'))
            pred_ref = None
        elif format_type == 'score_gt':
            pred_score = parse_score(r.get('score'))
            gt_score = parse_score(r.get('ground_truth_score'))  # Override gt_score from ground_truth_score field
            pred_ref = None
        else:  # score_meta
            pred_score = parse_score(r.get('score'))
            pred_ref = None

        # Calculate score errors
        if gt_score is not None and pred_score is not None:
            old_err = calculate_old_score_error(gt_score, pred_score)
            new_err = calculate_new_score_error(gt_score, pred_score)

            old_score_errors.append(old_err)
            new_score_errors.append(new_err)

            gt_pct = int(gt_score * 100)
            score_groups[gt_pct].append({
                'old': old_err,
                'new': new_err,
                'gt': gt_score,
                'pred': pred_score
            })

        # Calculate ref errors (only for ref_score format)
        if format_type == 'ref_score' and gt_ref is not None and pred_ref is not None:
            old_err = calculate_old_ref_error(gt_ref, pred_ref)
            new_err = calculate_new_ref_error(gt_ref, pred_ref, num_demos)

            old_ref_errors.append(old_err)
            new_ref_errors.append(new_err)

            ref_groups[gt_ref].append({
                'old': old_err,
                'new': new_err,
                'gt_ref': gt_ref,
                'pred_ref': pred_ref,
                'num_demos': num_demos
            })

    return {
        'format_type': format_type,
        'total_samples': len(results),
        'old_score_errors': old_score_errors,
        'new_score_errors': new_score_errors,
        'old_ref_errors': old_ref_errors,
        'new_ref_errors': new_ref_errors,
        'score_groups': dict(score_groups),
        'ref_groups': dict(ref_groups)
    }


def print_results(analysis: Dict):
    """Print analysis results to console."""
    new_scores = [x for x in analysis['new_score_errors'] if x != float('inf')]
    new_refs = analysis['new_ref_errors']

    if new_scores:
        print(f"Score - Mean: {np.mean(new_scores):.4f}, Std: {np.std(new_scores):.4f}")

    if new_refs:
        print(f"Ref   - Mean: {np.mean(new_refs):.4f}, Std: {np.std(new_refs):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate normalized errors for progress estimation results"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the results JSONL file"
    )

    args = parser.parse_args()

    # Process and analyze
    format_type, results = process_results(args.input)
    analysis = analyze_results(format_type, results)

    # Print results
    print_results(analysis)


if __name__ == "__main__":
    main()
