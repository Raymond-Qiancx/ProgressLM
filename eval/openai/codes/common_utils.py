#!/usr/bin/env python3
"""
Common Utilities for OpenAI Progress Evaluation

Provides shared functions for:
- Response parsing (XML tag extraction)
- Evaluation metrics calculation
- Result formatting
"""

import re
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr


def parse_response(response: str) -> Dict[str, Any]:
    """
    Parse the model's response to extract XML tags.

    Expected format:
    <ref_think>reasoning...</ref_think>
    <ref>2</ref>  (expects 1-based integer index or "n/a")
    <score_think>reasoning...</score_think>
    <score>33%</score>  (supports "33%", "0.33", or "n/a")

    Args:
        response: Model output string

    Returns:
        Dictionary with parsed fields:
        {
            'ref_think': str,
            'ref': int or "n/a",
            'score_think': str,
            'score': float or "n/a" or None,
            'parse_error': bool
        }
    """
    result = {
        'ref_think': '',
        'ref': '',
        'score_think': '',
        'score': None,
        'parse_error': False
    }

    if not response:
        result['parse_error'] = True
        return result

    try:
        # Extract ref_think
        ref_think_match = re.search(r'<ref_think>(.*?)</ref_think>', response, re.DOTALL)
        if ref_think_match:
            result['ref_think'] = ref_think_match.group(1).strip()

        # Extract ref (expects integer 1-based index or "n/a")
        ref_match = re.search(r'<ref>(.*?)</ref>', response, re.DOTALL)
        if ref_match:
            ref_str = ref_match.group(1).strip()
            # Check for "n/a" first
            if ref_str.lower() in ["n/a", "na"]:
                result['ref'] = "n/a"
            else:
                try:
                    # Extract just the number (handle "No. 2", "2", "step 2", etc.)
                    ref_num = re.search(r'\d+', ref_str)
                    if ref_num:
                        result['ref'] = int(ref_num.group())
                    else:
                        result['ref'] = ref_str  # Keep original if no number found
                except (ValueError, AttributeError):
                    result['ref'] = ref_str

        # Extract score_think
        score_think_match = re.search(r'<score_think>(.*?)</score_think>', response, re.DOTALL)
        if score_think_match:
            result['score_think'] = score_think_match.group(1).strip()

        # Extract score (supports "33%", "0.33", or "n/a")
        score_match = re.search(r'<score>(.*?)</score>', response, re.DOTALL)
        if score_match:
            score_str = score_match.group(1).strip()
            # Check for "n/a" first
            if score_str.lower() in ["n/a", "na"]:
                result['score'] = "n/a"
            else:
                try:
                    # Remove % sign if present
                    if score_str.endswith('%'):
                        score_value = float(score_str[:-1]) / 100.0
                    else:
                        score_value = float(score_str)
                        # If > 1.0, assume it's percentage without % sign
                        if score_value > 1.0:
                            score_value = score_value / 100.0

                    # Clamp to [0, 1]
                    result['score'] = max(0.0, min(1.0, score_value))
                except ValueError:
                    result['parse_error'] = True
        else:
            result['parse_error'] = True

    except Exception as e:
        result['parse_error'] = True

    return result


def parse_nothink_response(response: str) -> Dict[str, Any]:
    """
    Parse the nothink version's simple numeric response.

    Expected format: direct progress score output
    Examples: "45%", "0.45", "45", "n/a"

    Args:
        response: Model output string

    Returns:
        Dictionary with parsed fields:
        {
            'score': float or "n/a" or None,
            'parse_error': bool
        }
    """
    result = {
        'score': None,
        'parse_error': False
    }

    if not response:
        result['parse_error'] = True
        return result

    response = response.strip()

    # Check for "n/a" first
    if response.lower() in ["n/a", "na"]:
        result['score'] = "n/a"
        return result

    try:
        # Try to match percentage format (e.g., "45%", "45.5%")
        match = re.search(r'(\d+\.?\d*)\s*%', response)
        if match:
            score_value = float(match.group(1)) / 100.0
            result['score'] = max(0.0, min(1.0, score_value))
            return result

        # Try to match decimal format (e.g., "0.45")
        match = re.search(r'0?\.\d+', response)
        if match:
            score_value = float(match.group())
            result['score'] = max(0.0, min(1.0, score_value))
            return result

        # Try to match integer (e.g., "45" - assume percentage)
        match = re.search(r'\d+', response)
        if match:
            value = float(match.group())
            # If > 1.0, assume it's percentage without % sign
            if value > 1.0:
                score_value = value / 100.0
            else:
                score_value = value
            result['score'] = max(0.0, min(1.0, score_value))
            return result

        # No number found
        result['parse_error'] = True

    except Exception as e:
        result['parse_error'] = True

    return result


def calculate_evaluation_score(predicted: Optional[float], ground_truth: float) -> float:
    """
    Calculate evaluation score: |ground_truth - predicted| / ground_truth

    Uses pure relative error metric. Lower is better (0.0 = perfect prediction).

    Args:
        predicted: Predicted progress score (0-1) or None if parsing failed
        ground_truth: Ground truth progress score (0-1)

    Returns:
        Relative error (0.0 = perfect, higher = worse), or inf if invalid
    """
    if predicted is None:
        return float('inf')

    # Avoid division by zero
    if ground_truth == 0.0:
        return 0.0 if predicted == 0.0 else float('inf')

    relative_error = abs(ground_truth - predicted) / ground_truth
    return relative_error


def calculate_ref_error(predicted_ref: Optional[int], ground_truth_ref: int) -> float:
    """
    Calculate reference index error: |ground_truth_ref - predicted_ref|

    Measures absolute difference between predicted and ground truth indices.

    Args:
        predicted_ref: Predicted reference index (1-based) or None
        ground_truth_ref: Ground truth closest index (1-based)

    Returns:
        Absolute error (0.0 = perfect, higher = worse), or inf if invalid
    """
    if predicted_ref is None:
        return float('inf')

    if not isinstance(predicted_ref, int):
        return float('inf')

    absolute_error = abs(ground_truth_ref - predicted_ref)
    return float(absolute_error)


def calculate_false_positives(
    predicted_ref: Union[int, str, None],
    predicted_score: Union[float, str, None],
    gt_ref: Optional[int],
    gt_score: Optional[float]
) -> Tuple[bool, bool]:
    """
    Calculate false positive rates for ref and score.

    False positive occurs when:
    - GT is numeric but prediction is "n/a"
    - GT is "n/a" but prediction is numeric

    Args:
        predicted_ref: Predicted ref (int, "n/a", or invalid)
        predicted_score: Predicted score (float, "n/a", or invalid)
        gt_ref: Ground truth ref (int or None)
        gt_score: Ground truth score (float or None)

    Returns:
        (is_ref_false_positive, is_score_false_positive)
    """
    # Check ref false positive
    gt_ref_is_na = (gt_ref is None)
    pred_ref_is_na = (predicted_ref == "n/a" or predicted_ref == "" or not isinstance(predicted_ref, int))
    ref_fp = gt_ref_is_na != pred_ref_is_na

    # Check score false positive
    gt_score_is_na = (gt_score is None)
    pred_score_is_na = (predicted_score == "n/a" or predicted_score is None or not isinstance(predicted_score, (int, float)))
    score_fp = gt_score_is_na != pred_score_is_na

    return ref_fp, score_fp


def calculate_voc_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate VOC (trajectory order consistency) using Spearman correlation.

    Process:
    1. Group samples by trajectory ID
    2. Filter: only keep trajectories where GT is numeric
    3. For each trajectory:
       - Sort by GT progress_score to get true order
       - Sort by predicted score (n/a -> 0.0) to get predicted order
       - Calculate Spearman correlation between rankings
    4. Return mean and std of all valid VOCs

    Args:
        results: List of result dictionaries with meta_data

    Returns:
        Dictionary with VOC statistics
    """
    # Group by trajectory ID
    trajectories = defaultdict(list)
    for res in results:
        meta = res.get('meta_data', {})
        traj_id = meta.get('id', '')

        # Only include if GT has numeric values
        gt_ref = meta.get('closest_idx')
        gt_score = meta.get('progress_score')

        if gt_ref is not None and gt_score is not None:
            pred_score = res.get('score')
            # Convert n/a to 0.0 for ranking
            if pred_score == "n/a" or pred_score is None:
                pred_score_numeric = 0.0
            elif isinstance(pred_score, str):
                # Handle percentage strings
                try:
                    if pred_score.endswith('%'):
                        pred_score_numeric = float(pred_score[:-1]) / 100.0
                    else:
                        pred_score_numeric = float(pred_score)
                except ValueError:
                    pred_score_numeric = 0.0
            else:
                pred_score_numeric = float(pred_score) if isinstance(pred_score, (int, float)) else 0.0

            trajectories[traj_id].append({
                'gt_score': gt_score,
                'pred_score': pred_score_numeric,
                'result': res
            })

    # Calculate VOC for each trajectory
    voc_values = []
    for traj_id, samples in trajectories.items():
        if len(samples) <= 1:
            continue

        # Sort by GT score to get true ranking
        samples_sorted_by_gt = sorted(samples, key=lambda x: x['gt_score'])
        true_order = list(range(len(samples_sorted_by_gt)))

        # Sort by predicted score to get predicted ranking
        samples_sorted_by_pred = sorted(samples, key=lambda x: x['pred_score'])

        # Map each sample to its predicted rank
        pred_rank_map = {id(s['result']): rank for rank, s in enumerate(samples_sorted_by_pred)}
        pred_order = [pred_rank_map[id(s['result'])] for s in samples_sorted_by_gt]

        # Calculate Spearman correlation
        if len(set(true_order)) > 1 and len(set(pred_order)) > 1:
            correlation, _ = spearmanr(true_order, pred_order)
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


def format_score_string(score: Union[float, str, None]) -> Optional[str]:
    """
    Format a score value to percentage string.

    Args:
        score: Score value (float 0-1), "n/a", or None

    Returns:
        Formatted string like "33%" or "n/a" or None
    """
    if score == "n/a":
        return "n/a"
    elif isinstance(score, (int, float)):
        return f"{int(score * 100)}%"
    else:
        return None


def format_ref_string(ref: Union[int, str, None]) -> Optional[str]:
    """
    Format a reference index to string.

    Args:
        ref: Reference index (int), "n/a", or invalid

    Returns:
        Formatted string
    """
    if ref == "n/a":
        return "n/a"
    elif isinstance(ref, int):
        return str(ref)
    else:
        return None


def get_sample_unique_id(sample: Dict[str, Any]) -> str:
    """
    Generate a unique ID for a sample based on id and progress_score.

    Args:
        sample: Sample dictionary

    Returns:
        Unique ID string: "id_progress_score"
    """
    sample_id = sample.get('id', 'unknown')
    progress_score = sample.get('progress_score', 'unknown')

    # Handle None progress_score
    if progress_score is None:
        progress_score = 'na'
    elif isinstance(progress_score, float):
        progress_score = f"{int(progress_score * 100)}%"

    return f"{sample_id}_{progress_score}"
