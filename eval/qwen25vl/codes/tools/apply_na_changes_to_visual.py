#!/usr/bin/env python3
"""
Script to apply n/a support changes from run_text_demo.py to run_visual_demo.py.
This applies the same modifications that were made to support "n/a" values in text_demo to visual_demo.
"""

import re

def apply_changes(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Fix calculate_evaluation_score function signature and logic
    content = re.sub(
        r'def calculate_evaluation_score\(predicted: Optional\[float\], ground_truth: float\) -> float:',
        r'def calculate_evaluation_score(predicted: Union[float, str, None], ground_truth: Union[float, str]) -> float:',
        content
    )

    # Replace calculate_evaluation_score function body
    old_calc_eval = r'''def calculate_evaluation_score\(predicted: Union\[float, str, None\], ground_truth: Union\[float, str\]\) -> float:
    """
    Calculate evaluation score: \|ground_truth - predicted\| / ground_truth

    Uses pure relative error metric\. Lower is better \(0\.0 = perfect prediction\)\.
    This is more suitable for progress estimation as it considers the magnitude
    of the true value\.

    Args:
        predicted: Predicted progress score \(0-1\) or None if parsing failed
        ground_truth: Ground truth progress score \(0-1\)

    Returns:
        Relative error \(0\.0 = perfect, higher = worse\), or inf if predicted is None or ground_truth is 0
    """
    if predicted is None:
        return float\('inf'\)

    # Avoid division by zero
    if ground_truth == 0\.0:
        # If ground_truth is 0, only perfect prediction gets 0\.0
        return 0\.0 if predicted == 0\.0 else float\('inf'\)

    relative_error = abs\(ground_truth - predicted\) / ground_truth
    return relative_error'''

    new_calc_eval = '''def calculate_evaluation_score(predicted: Union[float, str, None], ground_truth: Union[float, str]) -> float:
    """
    Calculate evaluation score: |ground_truth - predicted| / ground_truth

    Uses pure relative error metric. Lower is better (0.0 = perfect prediction).
    For "n/a" ground truth, checks if predicted is also "n/a" (Pass/Fail).

    Args:
        predicted: Predicted progress score (0-1), "n/a", or None if parsing failed
        ground_truth: Ground truth progress score (0-1) or "n/a"

    Returns:
        Relative error (0.0 = perfect, higher = worse), or inf if prediction failed
    """
    # Handle "n/a" ground truth: only "n/a" prediction is correct
    if isinstance(ground_truth, str) and ground_truth.lower() == "n/a":
        if isinstance(predicted, str) and predicted.lower() == "n/a":
            return 0.0  # Perfect match: both "n/a"
        else:
            return float('inf')  # Wrong: should predict "n/a" but didn't

    # Handle "n/a" prediction with numeric ground truth
    if isinstance(predicted, str) and predicted.lower() == "n/a":
        return float('inf')  # Wrong: predicted "n/a" but ground truth is numeric

    if predicted is None:
        return float('inf')

    # Avoid division by zero
    if ground_truth == 0.0:
        # If ground_truth is 0, only perfect prediction gets 0.0
        return 0.0 if predicted == 0.0 else float('inf')

    relative_error = abs(ground_truth - predicted) / ground_truth
    return relative_error'''

    content = re.sub(old_calc_eval, new_calc_eval, content, flags=re.DOTALL)

    # 2. Fix calculate_ref_error function
    content = re.sub(
        r'def calculate_ref_error\(predicted_ref: Optional\[int\], ground_truth_ref: int\) -> float:',
        r'def calculate_ref_error(predicted_ref: Union[int, str, None], ground_truth_ref: Union[int, str]) -> float:',
        content
    )

    old_calc_ref = r'''def calculate_ref_error\(predicted_ref: Union\[int, str, None\], ground_truth_ref: Union\[int, str\]\) -> float:
    """
    Calculate reference index error: \|ground_truth_ref - predicted_ref\|

    Measures absolute difference between predicted and ground truth image indices\.
    Lower is better \(0\.0 = perfect match\)\.

    Args:
        predicted_ref: Predicted reference image index \(1-based\) or None if parsing failed
        ground_truth_ref: Ground truth closest image index \(1-based\)

    Returns:
        Absolute error \(0\.0 = perfect, higher = worse\), or inf if predicted_ref is None or not an integer
    """
    if predicted_ref is None:
        return float\('inf'\)

    # Ensure predicted_ref is an integer
    if not isinstance\(predicted_ref, int\):
        return float\('inf'\)

    absolute_error = abs\(ground_truth_ref - predicted_ref\)
    return float\(absolute_error\)'''

    new_calc_ref = '''def calculate_ref_error(predicted_ref: Union[int, str, None], ground_truth_ref: Union[int, str]) -> float:
    """
    Calculate reference index error: |ground_truth_ref - predicted_ref|

    Measures absolute difference between predicted and ground truth image indices.
    For "n/a" ground truth, checks if predicted is also "n/a" (Pass/Fail).

    Args:
        predicted_ref: Predicted reference image index (1-based), "n/a", or None if parsing failed
        ground_truth_ref: Ground truth closest image index (1-based) or "n/a"

    Returns:
        Absolute error (0.0 = perfect, higher = worse), or inf if prediction failed
    """
    # Handle "n/a" ground truth: only "n/a" prediction is correct
    if isinstance(ground_truth_ref, str) and ground_truth_ref.lower() == "n/a":
        if isinstance(predicted_ref, str) and predicted_ref.lower() == "n/a":
            return 0.0  # Perfect match: both "n/a"
        else:
            return float('inf')  # Wrong: should predict "n/a" but didn't

    # Handle "n/a" prediction with numeric ground truth
    if isinstance(predicted_ref, str) and predicted_ref.lower() == "n/a":
        return float('inf')  # Wrong: predicted "n/a" but ground truth is numeric

    if predicted_ref is None:
        return float('inf')

    # Ensure predicted_ref is an integer
    if not isinstance(predicted_ref, int):
        return float('inf')

    absolute_error = abs(ground_truth_ref - predicted_ref)
    return float(absolute_error)'''

    content = re.sub(old_calc_ref, new_calc_ref, content, flags=re.DOTALL)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Applied changes to {file_path}")

if __name__ == "__main__":
    apply_changes("run_visual_demo.py")
