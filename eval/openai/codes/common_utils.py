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
    Calculate evaluation score: |ground_truth - predicted| / max(ground_truth, 1 - ground_truth)

    Uses normalized error metric. Lower is better (0.0 = perfect prediction).
    This normalization treats small and large GT values fairly.

    Args:
        predicted: Predicted progress score (0-1) or None if parsing failed
        ground_truth: Ground truth progress score (0-1)

    Returns:
        Normalized error (0.0 = perfect, 1.0 = max possible error), or inf if invalid
    """
    if predicted is None:
        return float('inf')

    # Calculate max possible error for normalization
    max_possible = max(ground_truth, 1.0 - ground_truth)

    # Avoid division by zero (only happens when gt = 0.5 exactly, but max_possible would be 0.5)
    if max_possible == 0.0:
        return 0.0 if predicted == ground_truth else float('inf')

    normalized_error = abs(ground_truth - predicted) / max_possible
    return normalized_error


def calculate_ref_error(predicted_ref: Optional[int], ground_truth_ref: int, num_demos: int = None) -> float:
    """
    Calculate normalized reference index error: |gt_ref - pred_ref| / max(gt_ref - 1, num_demos - gt_ref)

    Uses normalized error metric for fair comparison across different trajectory lengths.

    Args:
        predicted_ref: Predicted reference index (1-based) or None
        ground_truth_ref: Ground truth closest index (1-based)
        num_demos: Total number of demos in the trajectory (required for normalization)

    Returns:
        Normalized error (0.0 = perfect, 1.0 = max possible error), or inf if invalid
    """
    if predicted_ref is None:
        return float('inf')

    if not isinstance(predicted_ref, int):
        return float('inf')

    # If num_demos not provided, fall back to absolute error
    if num_demos is None:
        return float(abs(ground_truth_ref - predicted_ref))

    # Calculate max possible error for normalization
    max_possible = max(ground_truth_ref - 1, num_demos - ground_truth_ref)

    if max_possible == 0:
        return 0.0

    normalized_error = abs(ground_truth_ref - predicted_ref) / max_possible
    return normalized_error


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

    Uses sample-level N/A filtering: N/A predictions are excluded from
    calculation rather than being converted to 0.0.

    Process:
    1. Group samples by complete trajectory ID
    2. Filter out N/A predictions at sample level
    3. For each trajectory with >= 2 valid samples:
       - Calculate Spearman correlation between GT and predicted scores
       - Skip if all predictions or GT scores are constant
    4. Return mean, std and detailed statistics

    Args:
        results: List of result dictionaries with meta_data containing id, progress_score

    Returns:
        Dictionary with VOC statistics and sample coverage info
    """
    from collections import defaultdict

    # Group by trajectory ID and collect samples
    trajectories = defaultdict(list)
    total_samples = 0
    total_na_samples = 0

    for res in results:
        meta = res.get('meta_data', {})
        traj_id = meta.get('id', '')
        gt_score = meta.get('progress_score')

        if gt_score is not None:
            total_samples += 1
            pred_score_raw = res.get('score')

            # Parse score, N/A returns None (not 0.0)
            pred_score = None
            if pred_score_raw is not None and pred_score_raw != "n/a":
                if isinstance(pred_score_raw, (int, float)):
                    pred_score = float(pred_score_raw)
                elif isinstance(pred_score_raw, str):
                    # Parse percentage format "45%"
                    score_str = pred_score_raw.strip()
                    if score_str.endswith('%'):
                        try:
                            pred_score = float(score_str[:-1]) / 100.0
                        except:
                            pass
                    else:
                        try:
                            val = float(score_str)
                            pred_score = val if val <= 1.0 else val / 100.0
                        except:
                            pass

            if pred_score is None:
                total_na_samples += 1

            trajectories[traj_id].append({
                'gt_score': gt_score,
                'pred_score': pred_score  # Can be None
            })

    # Calculate VOC for each trajectory (with sample-level filtering)
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

        # Skip if all predictions are the same (no variation)
        if len(set(pred_scores)) == 1:
            skipped_constant_pred += 1
            continue

        # Skip if all GT scores are the same
        if len(set(gt_scores)) == 1:
            skipped_constant_gt += 1
            continue

        correlation, _ = spearmanr(gt_scores, pred_scores)
        if not np.isnan(correlation):
            voc_values.append(correlation)

    # Calculate statistics
    valid_samples_count = total_samples - total_na_samples
    sample_coverage = valid_samples_count / total_samples if total_samples > 0 else 0.0

    if len(voc_values) > 0:
        return {
            'voc_mean': float(np.mean(voc_values)),
            'voc_std': float(np.std(voc_values)),
            'voc_count': len(voc_values),
            'voc_values': voc_values,
            'total_samples': total_samples,
            'valid_samples': valid_samples_count,
            'na_samples': total_na_samples,
            'sample_coverage': sample_coverage,
            'total_trajectories': len(trajectories),
            'skipped_all_na': skipped_all_na,
            'skipped_single': skipped_single,
            'skipped_constant_pred': skipped_constant_pred,
            'skipped_constant_gt': skipped_constant_gt
        }
    else:
        return {
            'voc_mean': None,
            'voc_std': None,
            'voc_count': 0,
            'voc_values': [],
            'total_samples': total_samples,
            'valid_samples': valid_samples_count,
            'na_samples': total_na_samples,
            'sample_coverage': sample_coverage,
            'total_trajectories': len(trajectories),
            'skipped_all_na': skipped_all_na,
            'skipped_single': skipped_single,
            'skipped_constant_pred': skipped_constant_pred,
            'skipped_constant_gt': skipped_constant_gt
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
