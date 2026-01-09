#!/usr/bin/env python3
"""
Script to apply n/a support patches to all run_*.py scripts.
This script applies the same modifications that were made to run_text_demo_nothink.py
to all other run scripts.
"""

import re
import sys

def patch_parse_response_ref(content):
    """Patch parse response function to support n/a for ref field"""
    # Pattern for ref extraction
    old_pattern = r"(# Extract ref \(now expects integer 1-based (?:index|image number)\)\s+ref_match = re\.search\(r'<ref>\(\.\*\?\)</ref>', response, re\.DOTALL\)\s+if ref_match:\s+ref_str = ref_match\.group\(1\)\.strip\(\)\s+try:\s+# Extract just the number[^\n]*\s+ref_num = re\.search\(r'\\d\+', ref_str\)\s+if ref_num:\s+result\['ref'\] = int\(ref_num\.group\(\)\)\s+else:\s+result\['ref'\] = ref_str[^\n]*\s+except \(ValueError, AttributeError\):\s+result\['ref'\] = ref_str)"

    new_pattern = r"""# Extract ref (now expects integer 1-based \1 or "n/a")
        ref_match = re.search(r'<ref>(.*?)</ref>', response, re.DOTALL)
        if ref_match:
            ref_str = ref_match.group(1).strip()
            # Check for "n/a" first
            if ref_str.lower() in ["n/a", "na"]:
                result['ref'] = "n/a"
            else:
                try:
                    # Extract just the number \2
                    ref_num = re.search(r'\d+', ref_str)
                    if ref_num:
                        result['ref'] = int(ref_num.group())
                    else:
                        result['ref'] = ref_str  # Keep original if no number found
                except (ValueError, AttributeError):
                    result['ref'] = ref_str"""

    # Simpler approach: find and replace specific patterns
    if '# Extract ref (now expects integer 1-based' in content and 'if ref_str.lower() in ["n/a"' not in content:
        # Find the ref extraction block
        ref_pattern = re.compile(
            r'(        # Extract ref \(now expects integer 1-based .*?\))\n'
            r'(        ref_match = re\.search.*?\n)'
            r'(        if ref_match:)\n'
            r'(            ref_str = ref_match\.group\(1\)\.strip\(\))\n'
            r'(            try:)\n'
            r'(                # Extract just the number.*?\n)'
            r'(                ref_num = re\.search.*?\n)'
            r'(                if ref_num:)\n'
            r'(                    result\[.ref.\] = int\(ref_num\.group\(\)\))\n'
            r'(                else:)\n'
            r'(                    result\[.ref.\] = ref_str  # Keep original.*?\n)'
            r'(            except \(ValueError, AttributeError\):)\n'
            r'(                result\[.ref.\] = ref_str)',
            re.MULTILINE
        )

        replacement = r'''\1 or "n/a")
\2\3
\4
            # Check for "n/a" first
            if ref_str.lower() in ["n/a", "na"]:
                result['ref'] = "n/a"
            else:
\5
\6\7\8
\9
\10\11
\12
\13'''

        content = ref_pattern.sub(replacement, content)

    return content

def patch_parse_response_score(content):
    """Patch parse response function to support n/a for score field"""
    if '# Extract score (supports' in content and 'if score_str.lower() in ["n/a"' not in content:
        score_pattern = re.compile(
            r'(        # Extract score \(supports .*?\))\n'
            r'(        score_match = re\.search.*?\n)'
            r'(        if score_match:)\n'
            r'(            score_str = score_match\.group\(1\)\.strip\(\))\n'
            r'(            try:)\n',
            re.MULTILINE
        )

        replacement = r'''\1 or "n/a")
\2\3
\4
            # Check for "n/a" first
            if score_str.lower() in ["n/a", "na"]:
                result['score'] = "n/a"
            else:
\5
'''
        content = score_pattern.sub(replacement, content)

        # Also need to add else block before except
        content = content.replace(
            '                # Clamp to [0, 1]\n'
            '                result[\'score\'] = max(0.0, min(1.0, score_value))\n'
            '            except ValueError:',
            '                    # Clamp to [0, 1]\n'
            '                    result[\'score\'] = max(0.0, min(1.0, score_value))\n'
            '            except ValueError:'
        )

    return content

def patch_imports(content):
    """Add scipy and numpy imports if missing"""
    if 'from scipy.stats import spearmanr' not in content:
        # Find the import section
        import_pattern = re.compile(
            r'(from typing import List, Dict, Any, Optional)\n',
            re.MULTILINE
        )
        replacement = r'\1, Tuple\n'
        content = import_pattern.sub(replacement, content)

        # Add scipy imports after multiprocessing
        mp_pattern = re.compile(
            r'(from multiprocessing import Manager, Process, Queue)\n',
            re.MULTILINE
        )
        replacement = r'\1\nfrom scipy.stats import spearmanr\nimport numpy as np\n'
        content = mp_pattern.sub(replacement, content)

    return content

def add_helper_functions(content):
    """Add false positive and VOC calculation functions"""
    # Check if functions already exist
    if 'def calculate_false_positives' in content:
        return content

    # Find the location after calculate_ref_error
    insert_pattern = re.compile(
        r'(def calculate_ref_error.*?return float\(absolute_error\)\n)\n\n',
        re.DOTALL | re.MULTILINE
    )

    functions_to_add = '''

def calculate_false_positives(predicted_ref, predicted_score, gt_ref, gt_score) -> Tuple[bool, bool]:
    """
    Calculate false positive rates for ref and score.

    False positive occurs when:
    - GT is numeric but prediction is "n/a"
    - GT is "n/a" but prediction is numeric

    Correct cases:
    - Both GT and prediction are "n/a"
    - Both GT and prediction are numeric (use error calculation instead)

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

    # False positive: mismatch between GT and prediction
    ref_fp = gt_ref_is_na != pred_ref_is_na

    # Check score false positive
    gt_score_is_na = (gt_score is None)
    pred_score_is_na = (predicted_score == "n/a" or predicted_score is None or not isinstance(predicted_score, (int, float)))

    # False positive: mismatch between GT and prediction
    score_fp = gt_score_is_na != pred_score_is_na

    return ref_fp, score_fp


def calculate_voc_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate VOC (trajectory order consistency) using Spearman correlation.

    Process:
    1. Group samples by complete trajectory ID
    2. Filter: only keep trajectories where GT closest_idx and progress_score are both numeric
    3. For each trajectory:
       - Sort by GT progress_score to get true order
       - Sort by predicted score (n/a → 0.0) to get predicted order
       - Calculate Spearman correlation between rankings
       - Single-sample trajectories → VOC = None
    4. Return mean and std of all valid VOCs

    Args:
        results: List of result dictionaries with meta_data containing id, closest_idx, progress_score

    Returns:
        Dictionary with VOC statistics:
        {
            'voc_mean': float or None,
            'voc_std': float or None,
            'voc_count': int,  # number of trajectories with VOC
            'voc_values': List[float]  # individual VOC values
        }
    """
    from collections import defaultdict

    # Group by trajectory ID
    trajectories = defaultdict(list)
    for res in results:
        meta = res.get('meta_data', {})
        traj_id = meta.get('id', '')

        # Only include if GT has numeric values
        gt_ref = meta.get('closest_idx')
        gt_score = meta.get('progress_score')

        if gt_ref is not None and gt_score is not None:
            # GT is numeric
            pred_score = res.get('score')
            # Convert n/a to 0.0 for ranking
            if pred_score == "n/a" or pred_score is None:
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
            # Cannot calculate correlation for single sample
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


'''

    content = insert_pattern.sub(r'\1' + functions_to_add, content)
    return content

def main(filename):
    """Apply all patches to a file"""
    print(f"Processing {filename}...")

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Apply patches
    content = patch_imports(content)
    content = patch_parse_response_ref(content)
    content = patch_parse_response_score(content)
    content = add_helper_functions(content)

    # More complex patches would go here (worker_process, tqdm, summary)
    # For now, we'll focus on the parse functions and helper functions

    if content != original_content:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Updated {filename}")
    else:
        print(f"  - No changes needed for {filename}")

if __name__ == '__main__':
    files = [
        'run_visual_demo_nothink.py',
        'run_text_demo.py',
        'run_visual_demo.py',
        'run_text_demo_single.py',
        'run_visual_demo_single.py',
        'run_text_demo_72B_nothink.py',
        'run_visual_demo_72B_nothink.py',
    ]

    for f in files:
        try:
            main(f)
        except Exception as e:
            print(f"  ✗ Error processing {f}: {e}")
