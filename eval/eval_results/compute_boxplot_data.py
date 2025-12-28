#!/usr/bin/env python3
"""
Compute Boxplot Data for VOC and Score Error (nothink models only)

This script parses all_results_paths.txt, extracts nothink model paths,
computes VOC and Score Error for each model, and outputs JSON data
suitable for boxplot visualization.

Usage:
    python compute_boxplot_data.py

Output:
    boxplot_data_nothink.json
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from scipy.stats import spearmanr
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Constants
PATHS_FILE = "/projects/p32958/chengxuan/ProgressLM/eval/eval_results/map/all_results_paths.txt"
OUTPUT_FILE = "/projects/p32958/chengxuan/ProgressLM/eval/eval_results/boxplot_data_nothink.json"

# Tasks to exclude (negative samples have null progress_score)
EXCLUDED_TASKS = ['edit_nega', 'text_nega']

# Task name normalization mapping
TASK_NORMALIZATION = {
    'nega_text': 'text_nega',
    'normal_text': 'text_normal',
    'multi_view': 'visual_multi_view',
    'normal_view': 'visual_single_view',
    'visual_edit_nega': 'edit_nega',
    'visual_multi': 'visual_multi_view',
    'visual_normal': 'visual_single_view',
}


def parse_score(score_str):
    """Parse score string to float [0, 1]. Returns None for N/A or invalid."""
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


def detect_format(first_record):
    """Detect JSONL format based on field names."""
    if 'predicted_score' in first_record:
        return 'predicted_score'
    elif 'score' in first_record:
        return 'score'
    return None


def calculate_normalized_error(gt, pred):
    """Calculate normalized score error: |gt - pred| / max(gt, 1 - gt)"""
    max_possible = max(gt, 1.0 - gt)
    if max_possible == 0:
        return 0.0
    return abs(gt - pred) / max_possible


def compute_boxplot_stats(values):
    """
    Compute 5-number summary for boxplot visualization.

    Returns:
        dict with min, q1, median, q3, max, mean, n
    """
    if not values:
        return None

    arr = np.array(values)
    return {
        'min': float(np.min(arr)),
        'q1': float(np.percentile(arr, 25)),
        'median': float(np.median(arr)),
        'q3': float(np.percentile(arr, 75)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'n': len(values)
    }


def compute_voc_and_errors(filepath):
    """
    Compute VOC values and score errors for a single JSONL file.

    Returns:
        dict with 'voc_values' (list) and 'score_errors' (list)
    """
    if not os.path.exists(filepath):
        return {'voc_values': [], 'score_errors': [], 'error': f'File not found: {filepath}'}

    # Load results
    results = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    except Exception as e:
        return {'voc_values': [], 'score_errors': [], 'error': str(e)}

    if not results:
        return {'voc_values': [], 'score_errors': [], 'error': 'Empty file'}

    # Detect format
    pred_field = detect_format(results[0])
    if pred_field is None:
        return {'voc_values': [], 'score_errors': [], 'error': 'Unknown format'}

    # Group by trajectory ID
    trajectories = defaultdict(list)
    score_errors = []

    for res in results:
        meta = res.get('meta_data', {})
        traj_id = meta.get('id', '')
        gt_score = meta.get('progress_score')

        # Skip if no ground truth
        if gt_score is None:
            continue

        pred_str = res.get(pred_field, '')
        pred_score = parse_score(pred_str)

        # Calculate score error if prediction is valid
        if pred_score is not None:
            error = calculate_normalized_error(gt_score, pred_score)
            score_errors.append(error)

        # Store for VOC calculation
        trajectories[traj_id].append({
            'gt_score': gt_score,
            'pred_score': pred_score
        })

    # Calculate VOC for each trajectory
    voc_values = []
    for traj_id, samples in trajectories.items():
        # Filter out samples with N/A predictions
        valid_samples = [s for s in samples if s['pred_score'] is not None]

        if len(valid_samples) <= 1:
            continue

        gt_scores = [s['gt_score'] for s in valid_samples]
        pred_scores = [s['pred_score'] for s in valid_samples]

        # Skip if no variation
        if len(set(pred_scores)) == 1 or len(set(gt_scores)) == 1:
            continue

        # Calculate Spearman correlation
        correlation, _ = spearmanr(gt_scores, pred_scores)
        if not np.isnan(correlation):
            voc_values.append(float(correlation))

    # Compute 5-number summary statistics
    return {
        'voc': compute_boxplot_stats(voc_values),
        'score_error': compute_boxplot_stats(score_errors)
    }


def parse_paths_file(filepath):
    """
    Parse all_results_paths.txt and extract nothink model paths,
    plus ProgressLM sft/rl paths (merged into new_pro_bench).

    Returns:
        list of dicts with 'model_family', 'task', 'size', 'path'
    """
    entries = []
    current_model_family = None
    current_task = None

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Parse model family header
        if line.startswith('# Model Family:'):
            current_model_family = line.split(':')[1].strip()

        # Parse task header
        elif line.startswith('# Task:'):
            # Extract primary task name (before any parentheses)
            task_match = re.match(r'# Task:\s*(\S+)', line)
            if task_match:
                current_task = task_match.group(1)
                # Normalize task name
                current_task = TASK_NORMALIZATION.get(current_task, current_task)

        # Parse type and size for nothink
        elif line.startswith('# type: nothink'):
            # Extract size
            size_match = re.search(r'size:\s*(\S+)', line)
            size = size_match.group(1) if size_match else 'unknown'

            # Next line should be the path
            i += 1
            if i < len(lines):
                path_line = lines[i].strip()
                if path_line and not path_line.startswith('#'):
                    entries.append({
                        'model_family': current_model_family,
                        'task': current_task,
                        'size': size,
                        'path': path_line
                    })

        # Parse ProgressLM sft/rl entries (merge into new_pro_bench)
        elif current_model_family == 'ProgressLM' and line.startswith('# type:'):
            type_match = re.search(r'# type:\s*(\w+)', line)
            size_match = re.search(r'size:\s*(\S+)', line)
            if type_match and size_match:
                model_type = type_match.group(1)  # sft or rl
                size = size_match.group(1)  # 3B
                # Create combined size name like "3B-SFT" or "3B-RL"
                combined_size = f"{size}-{model_type.upper()}"

                # Next line should be the path
                i += 1
                if i < len(lines):
                    path_line = lines[i].strip()
                    if path_line and not path_line.startswith('#'):
                        entries.append({
                            'model_family': 'new_pro_bench',  # Merge into new_pro_bench
                            'task': current_task,
                            'size': combined_size,
                            'path': path_line
                        })

        i += 1

    return entries


def main():
    print("=" * 60)
    print("Computing Boxplot Data for nothink Models")
    print("=" * 60)
    print(f"Paths file: {PATHS_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print()

    # Parse paths file
    print("Parsing paths file...")
    entries = parse_paths_file(PATHS_FILE)
    print(f"Found {len(entries)} nothink entries")

    # Filter out excluded tasks
    entries = [e for e in entries if e['task'] not in EXCLUDED_TASKS]
    print(f"After excluding {EXCLUDED_TASKS}: {len(entries)} entries")
    print()

    # Organize results by model family and size
    results = {
        'generated_at': datetime.now().isoformat(),
        'excluded_tasks': EXCLUDED_TASKS,
        'models': defaultdict(lambda: defaultdict(dict))
    }

    # Process each entry
    for entry in entries:
        model_family = entry['model_family']
        task = entry['task']
        size = entry['size']
        path = entry['path']

        print(f"Processing: {model_family} / {size} / {task}")

        data = compute_voc_and_errors(path)

        if 'error' in data:
            print(f"  WARNING: {data['error']}")
        else:
            voc_n = data['voc']['n'] if data['voc'] else 0
            err_n = data['score_error']['n'] if data['score_error'] else 0
            print(f"  VOC samples: {voc_n}, Score samples: {err_n}")

        results['models'][model_family][size][task] = data

    # Convert defaultdict to regular dict for JSON serialization
    results['models'] = {
        mf: {sz: dict(tasks) for sz, tasks in sizes.items()}
        for mf, sizes in results['models'].items()
    }

    # Save to JSON
    print()
    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done!")

    # Print summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for model_family, sizes in results['models'].items():
        print(f"\n{model_family}:")
        for size, tasks in sizes.items():
            print(f"  {size}:")
            for task, data in tasks.items():
                voc = data.get('voc')
                err = data.get('score_error')
                if voc and voc.get('n', 0) > 0:
                    err_mean = err['mean'] if err else 0
                    err_n = err['n'] if err else 0
                    print(f"    {task}: VOC={voc['mean']:.3f} (n={voc['n']}), Error={err_mean:.3f} (n={err_n})")
                else:
                    print(f"    {task}: No valid data")


if __name__ == '__main__':
    main()
