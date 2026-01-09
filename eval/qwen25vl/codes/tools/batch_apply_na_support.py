#!/usr/bin/env python3
"""
Batch apply n/a support modifications to all remaining run_*.py files.
This script copies the modifications from run_text_demo_nothink.py and run_visual_demo_nothink.py
to the remaining files.
"""

import re
import os

# Helper functions code to insert
HELPER_FUNCTIONS = '''

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

def process_file(filepath):
    """Process a single file to add n/a support"""
    print(f"\nProcessing {os.path.basename(filepath)}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    modified = False

    # 1. Update imports
    if 'from scipy.stats import spearmanr' not in content:
        content = content.replace(
            'from typing import List, Dict, Any, Optional',
            'from typing import List, Dict, Any, Optional, Tuple'
        )
        content = content.replace(
            'from multiprocessing import Manager, Process, Queue',
            'from multiprocessing import Manager, Process, Queue\nfrom scipy.stats import spearmanr\nimport numpy as np'
        )
        print("  ✓ Added imports")
        modified = True

    # 2. Update ref parsing in parse response function
    ref_pattern = re.compile(
        r'(\s+# Extract ref \(now expects integer 1-based (?:index|image number)\))\n'
        r'(\s+ref_match = re\.search.*?\n)'
        r'(\s+if ref_match:)\n'
        r'(\s+ref_str = ref_match\.group\(1\)\.strip\(\))\n'
        r'(\s+try:)\n',
        re.MULTILINE
    )

    if ref_pattern.search(content) and 'if ref_str.lower() in ["n/a"' not in content:
        content = ref_pattern.sub(
            r'\1 or "n/a")\n'
            r'\2'
            r'\3\n'
            r'\4\n'
            r'            # Check for "n/a" first\n'
            r'            if ref_str.lower() in ["n/a", "na"]:\n'
            r'                result[\'ref\'] = "n/a"\n'
            r'            else:\n'
            r'\5\n',
            content
        )

        # Indent the try block content
        content = re.sub(
            r'(\s+else:\n\s+try:\n)(                # Extract just the number)',
            r'\1    \2',
            content
        )
        content = re.sub(
            r'(if ref_str\.lower.*?\n.*?else:\n.*?try:\n)(                ref_num = re\.search)',
            r'\1    \2',
            content
        )
        content = re.sub(
            r'(if ref_str\.lower.*?\n.*?else:\n.*?try:\n.*?\n)(                if ref_num:)',
            r'\1    \2',
            content
        )
        content = re.sub(
            r'(if ref_str\.lower.*?\n.*?else:\n.*?try:\n.*?\n.*?\n)(                    result\[.ref.\] = int)',
            r'\1        \2',
            content
        )
        content = re.sub(
            r'(if ref_str\.lower.*?\n.*?else:\n.*?try:\n.*?\n.*?\n.*?\n)(                else:)',
            r'\1    \2',
            content
        )
        content = re.sub(
            r'(if ref_str\.lower.*?\n.*?else:\n.*?try:\n.*?\n.*?\n.*?\n.*?\n)(                    result\[.ref.\] = ref_str)',
            r'\1        \2',
            content
        )
        content = re.sub(
            r'(if ref_str\.lower.*?\n.*?else:\n.*?try:\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)(            except)',
            r'\1    \2',
            content
        )
        content = re.sub(
            r'(if ref_str\.lower.*?\n.*?else:\n.*?try:\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)(                result\[.ref.\] = ref_str)',
            r'\1        \2',
            content
        )

        print("  ✓ Updated ref parsing")
        modified = True

    # 3. Update score parsing
    score_pattern = re.compile(
        r'(\s+# Extract score \(supports (?:both )?"(?:\d+%|8%)" (?:and|or) "0\.\d+"\))\n'
        r'(\s+score_match = re\.search.*?\n)'
        r'(\s+if score_match:)\n'
        r'(\s+score_str = score_match\.group\(1\)\.strip\(\))\n'
        r'(\s+try:)\n',
        re.MULTILINE
    )

    if score_pattern.search(content) and 'if score_str.lower() in ["n/a"' not in content:
        content = score_pattern.sub(
            r'\1, or "n/a")\n'
            r'\2'
            r'\3\n'
            r'\4\n'
            r'            # Check for "n/a" first\n'
            r'            if score_str.lower() in ["n/a", "na"]:\n'
            r'                result[\'score\'] = "n/a"\n'
            r'            else:\n'
            r'\5\n',
            content
        )

        # Indent the try block
        content = re.sub(
            r'(if score_str\.lower.*?\n.*?else:\n.*?try:\n)(                # Remove % sign)',
            r'\1    \2',
            content
        )
        content = re.sub(
            r'(if score_str\.lower.*?\n.*?else:\n.*?try:\n.*?\n)(                if score_str\.endswith)',
            r'\1    \2',
            content
        )
        content = re.sub(
            r'(if score_str\.lower.*?\n.*?else:\n.*?try:\n.*?\n.*?\n)(                    score_value = float)',
            r'\1        \2',
            content
        )
        content = re.sub(
            r'(if score_str\.lower.*?\n.*?else:\n.*?try:\n.*?\n.*?\n.*?\n)(                else:)',
            r'\1    \2',
            content
        )
        content = re.sub(
            r'(if score_str\.lower.*?\n.*?else:\n.*?try:\n.*?\n.*?\n.*?\n.*?\n)(                    score_value = float\(score_str\))',
            r'\1        \2',
            content
        )
        content = re.sub(
            r'(if score_str\.lower.*?\n.*?else:\n.*?try:\n.*?\n.*?\n.*?\n.*?\n.*?\n)(                    # If > 1\.0)',
            r'\1        \2',
            content
        )
        content = re.sub(
            r'(if score_str\.lower.*?\n.*?else:\n.*?try:\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)(                    if score_value > 1\.0:)',
            r'\1        \2',
            content
        )
        content = re.sub(
            r'(if score_str\.lower.*?\n.*?else:\n.*?try:\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)(                        score_value = score_value / 100\.0)',
            r'\1            \2',
            content
        )
        content = re.sub(
            r'(if score_str\.lower.*?\n.*?else:\n.*?try:\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)(                # Clamp to \[0, 1\])',
            r'\1        \2',
            content
        )
        content = re.sub(
            r'(if score_str\.lower.*?\n.*?else:\n.*?try:\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)(                result\[.score.\] = max)',
            r'\1        \2',
            content
        )
        content = re.sub(
            r'(if score_str\.lower.*?\n.*?else:\n.*?try:\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)(            except ValueError:)',
            r'\1    \2',
            content
        )

        print("  ✓ Updated score parsing")
        modified = True

    # 4. Add helper functions if not present
    if 'def calculate_false_positives' not in content:
        # Find location after calculate_ref_error
        insert_pattern = re.compile(
            r'(def calculate_ref_error.*?return float\(absolute_error\))\n\n\ndef worker_process',
            re.DOTALL | re.MULTILINE
        )

        if insert_pattern.search(content):
            content = insert_pattern.sub(r'\1' + HELPER_FUNCTIONS + '\n\ndef worker_process', content)
            print("  ✓ Added helper functions")
            modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Successfully updated {os.path.basename(filepath)}")
    else:
        print(f"  - No changes needed (already updated)")

def main():
    """Main function"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # List of files to process
    files = [
        'run_text_demo.py',
        'run_visual_demo.py',
        'run_text_demo_single.py',
        'run_visual_demo_single.py',
        'run_text_demo_72B_nothink.py',
        'run_visual_demo_72B_nothink.py',
    ]

    for filename in files:
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            try:
                process_file(filepath)
            except Exception as e:
                print(f"  ✗ Error: {e}")
        else:
            print(f"  ✗ File not found: {filename}")

    print("\n" + "="*70)
    print("Batch processing complete!")
    print("="*70)
    print("\nNote: This script only updates the parsing functions and adds helper")
    print("functions. You still need to manually update:")
    print("  - worker_process result building logic")
    print("  - tqdm monitoring loop")
    print("  - Summary statistics calculation")
    print("\nRefer to run_text_demo_nothink.py for the complete pattern.")

if __name__ == '__main__':
    main()
