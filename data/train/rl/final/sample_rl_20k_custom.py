#!/usr/bin/env python3
"""
Custom RL data sampling script.
- 2 visual positive datasets: sample 5k each
- 2 text datasets: take all
- 2 visual negative datasets: duplicate (2x)
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Set random seed for reproducibility
random.seed(42)

# Base directory
BASE_DIR = Path("/gpfs/projects/p32958/chengxuan/ProgressLM/data/train/rl/final/raw")
OUTPUT_DIR = BASE_DIR.parent
OUTPUT_FILE = OUTPUT_DIR / "sampled_rl_20k_custom.jsonl"
REPORT_FILE = OUTPUT_DIR / "sampling_report_20k_custom.txt"

# Dataset configuration: (filename, original_count, target_count, strategy)
# strategy: "sample" = stratified sample, "all" = take all, "double" = duplicate
FILES_CONFIG = [
    ("visual_16k_normal_view_rl.jsonl", 16068, 5000, "sample"),      # visual positive 1
    ("visual_franka_cross_camera_ref_rl.jsonl", 7887, 5000, "sample"), # visual positive 2
    ("text_positive_6500_rl.jsonl", 6301, 6301, "all"),               # text positive
    ("text_nega_rl.jsonl", 2129, 2129, "all"),                        # text negative
    ("edited_visual_transfer_raw_rl.jsonl", 544, 1088, "double"),     # visual negative 1
    ("edited_visual_nega_raw_2.jsonl", 570, 1140, "double"),          # visual negative 2
]


def parse_id(record_id: str) -> Tuple[str, str, str]:
    """Parse id into (source, action_type, trajectory_id)."""
    parts = record_id.split('/')
    if len(parts) >= 3:
        return parts[0], parts[1], '/'.join(parts)
    return "unknown", "unknown", record_id


def load_and_group_by_trajectory(file_path: Path) -> Dict[str, List[dict]]:
    """Load jsonl file and group records by trajectory_id."""
    trajectories = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                _, _, traj_id = parse_id(record['id'])
                trajectories[traj_id].append(record)

    return dict(trajectories)


def group_trajectories_by_action_type(trajectories: Dict[str, List[dict]]) -> Dict[str, List[str]]:
    """Group trajectory IDs by action_type."""
    action_type_trajs = defaultdict(list)

    for traj_id, records in trajectories.items():
        _, action_type, _ = parse_id(records[0]['id'])
        action_type_trajs[action_type].append(traj_id)

    return dict(action_type_trajs)


def stratified_sample_trajectories(
    trajectories: Dict[str, List[dict]],
    target_count: int
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Sample trajectories using stratified sampling by action_type.
    Maintains trajectory integrity while targeting specific data count.
    """
    action_type_trajs = group_trajectories_by_action_type(trajectories)

    # Calculate current distribution
    action_type_counts = {}
    for action_type, traj_ids in action_type_trajs.items():
        total = sum(len(trajectories[tid]) for tid in traj_ids)
        action_type_counts[action_type] = total

    total_records = sum(action_type_counts.values())

    # Calculate target for each action_type (proportional)
    action_type_targets = {}
    for action_type, count in action_type_counts.items():
        proportion = count / total_records
        action_type_targets[action_type] = int(target_count * proportion)

    # Sample trajectories for each action_type
    sampled_records = []
    action_type_actual = {}

    for action_type in sorted(action_type_trajs.keys()):
        traj_ids = action_type_trajs[action_type]
        target = action_type_targets[action_type]

        random.shuffle(traj_ids)

        selected_count = 0
        for traj_id in traj_ids:
            traj_records = trajectories[traj_id]
            if selected_count + len(traj_records) <= target * 1.2:
                sampled_records.extend(traj_records)
                selected_count += len(traj_records)

            if selected_count >= target:
                break

        action_type_actual[action_type] = selected_count

    return sampled_records, action_type_actual


def load_all_records(file_path: Path) -> List[dict]:
    """Load all records from jsonl file."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def duplicate_records(records: List[dict]) -> List[dict]:
    """Duplicate records (2x)."""
    return records + records


def main():
    print("=" * 80)
    print("Custom RL Data Sampling - 20K")
    print("=" * 80)
    print()

    all_sampled_records = []
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("Custom RL Training Data Sampling Report - 20K")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Sampling Strategy:")
    report_lines.append("  - Visual positive (2 files): sample 5k each")
    report_lines.append("  - Text datasets (2 files): take all")
    report_lines.append("  - Visual negative (2 files): duplicate 2x")
    report_lines.append("")

    total_original = 0
    total_sampled = 0

    for filename, original_count, target_count, strategy in FILES_CONFIG:
        file_path = BASE_DIR / filename
        print(f"\nProcessing: {filename}")
        print(f"  Original: {original_count} records")
        print(f"  Target: {target_count} records")
        print(f"  Strategy: {strategy}")

        if strategy == "sample":
            trajectories = load_and_group_by_trajectory(file_path)
            sampled_records, action_type_actual = stratified_sample_trajectories(
                trajectories, target_count
            )
            actual_count = len(sampled_records)

        elif strategy == "all":
            sampled_records = load_all_records(file_path)
            actual_count = len(sampled_records)
            action_type_actual = {}

        elif strategy == "double":
            original_records = load_all_records(file_path)
            sampled_records = duplicate_records(original_records)
            actual_count = len(sampled_records)
            action_type_actual = {}

        print(f"  Sampled: {actual_count} records")

        all_sampled_records.extend(sampled_records)
        total_original += original_count
        total_sampled += actual_count

        report_lines.append(f"File: {filename}")
        report_lines.append(f"  Original: {original_count} records")
        report_lines.append(f"  Target: {target_count} records")
        report_lines.append(f"  Strategy: {strategy}")
        report_lines.append(f"  Sampled: {actual_count} records")

        if action_type_actual:
            report_lines.append(f"  Action types: {len(action_type_actual)}")
            top_action_types = sorted(action_type_actual.items(), key=lambda x: x[1], reverse=True)[:5]
            for action_type, count in top_action_types:
                report_lines.append(f"    - {action_type}: {count}")
        report_lines.append("")

    # Shuffle all records
    random.shuffle(all_sampled_records)

    # Write output file
    print(f"\n\nWriting output to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for record in all_sampled_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Total sampled: {total_sampled} records")

    # Add summary to report
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Total Original: {total_original} records")
    report_lines.append(f"Total Sampled: {total_sampled} records")
    report_lines.append("")

    # Analyze overall distribution
    overall_sources = defaultdict(int)
    for record in all_sampled_records:
        source, _, _ = parse_id(record['id'])
        overall_sources[source] += 1

    report_lines.append("=" * 80)
    report_lines.append("DATA SOURCE DISTRIBUTION")
    report_lines.append("=" * 80)
    for source, count in sorted(overall_sources.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_sampled * 100
        report_lines.append(f"  {source:30s} {count:5d} ({percentage:5.2f}%)")

    # Write report
    print(f"\nWriting report to: {REPORT_FILE}")
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print("\n" + "=" * 80)
    print("Sampling completed!")
    print("=" * 80)
    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"Report file: {REPORT_FILE}")


if __name__ == "__main__":
    main()
