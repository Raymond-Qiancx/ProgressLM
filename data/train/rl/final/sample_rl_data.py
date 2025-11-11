#!/usr/bin/env python3
"""
Stratified trajectory sampling for RL training data.
Samples ~5k records from 35k while maintaining:
- Trajectory integrity (same trajectory_id kept together)
- File proportions
- Action type balance
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Set random seed for reproducibility
random.seed(42)

# File paths and target counts
BASE_DIR = Path("/gpfs/projects/p32958/chengxuan/ProgressLM/data/train/rl/final/raw")
OUTPUT_FILE = BASE_DIR.parent / "sampled_rl_data_5k.jsonl"
REPORT_FILE = BASE_DIR.parent / "sampling_report_5k.txt"

FILES = [
    ("visual_16k_normal_view_rl.jsonl", 16068, 2270),  # (filename, original_count, target_count)
    ("visual_franka_cross_camera_ref_rl.jsonl", 7887, 1115),
    ("text_positive_6500_rl.jsonl", 6301, 890),
    ("visual_negative_trans_img_rl.jsonl", 2460, 345),
    ("text_nega_rl.jsonl", 2129, 300),
    ("edited_visual_transfer_raw_rl.jsonl", 544, 75),
]

TOTAL_TARGET = 5000


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
        # Get action_type from first record
        _, action_type, _ = parse_id(records[0]['id'])
        action_type_trajs[action_type].append(traj_id)

    return dict(action_type_trajs)


def stratified_sample_trajectories(
    trajectories: Dict[str, List[dict]],
    target_count: int
) -> List[dict]:
    """
    Sample trajectories using stratified sampling by action_type.
    Maintains trajectory integrity while targeting specific data count.
    """
    # Group trajectories by action_type
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

        # Shuffle trajectory order
        random.shuffle(traj_ids)

        # Greedily select trajectories until we reach target
        selected_count = 0
        for traj_id in traj_ids:
            traj_records = trajectories[traj_id]
            if selected_count + len(traj_records) <= target * 1.2:  # Allow 20% overflow
                sampled_records.extend(traj_records)
                selected_count += len(traj_records)

            if selected_count >= target:
                break

        action_type_actual[action_type] = selected_count

    return sampled_records, action_type_actual


def main():
    print("=" * 80)
    print("RL Data Stratified Trajectory Sampling")
    print("=" * 80)
    print()

    all_sampled_records = []
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("RL Training Data Sampling Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Target Total: {TOTAL_TARGET} records")
    report_lines.append("")

    total_original = 0
    total_sampled = 0

    # Process each file
    for filename, original_count, target_count in FILES:
        file_path = BASE_DIR / filename
        print(f"\nProcessing: {filename}")
        print(f"  Original: {original_count} records")
        print(f"  Target: {target_count} records")

        # Load and group by trajectory
        trajectories = load_and_group_by_trajectory(file_path)
        print(f"  Trajectories: {len(trajectories)}")

        # Perform stratified sampling
        sampled_records, action_type_actual = stratified_sample_trajectories(
            trajectories, target_count
        )

        actual_count = len(sampled_records)
        print(f"  Sampled: {actual_count} records ({actual_count/original_count*100:.1f}%)")

        # Add to combined output
        all_sampled_records.extend(sampled_records)

        # Update totals
        total_original += original_count
        total_sampled += actual_count

        # Add to report
        report_lines.append(f"File: {filename}")
        report_lines.append(f"  Original: {original_count} records")
        report_lines.append(f"  Target: {target_count} records")
        report_lines.append(f"  Sampled: {actual_count} records ({actual_count/original_count*100:.1f}%)")
        report_lines.append(f"  Trajectories: {len(trajectories)} -> sampled from subset")

        # Action type distribution
        report_lines.append(f"  Action types in sample: {len(action_type_actual)}")
        top_action_types = sorted(action_type_actual.items(), key=lambda x: x[1], reverse=True)[:10]
        report_lines.append(f"  Top 10 action types:")
        for action_type, count in top_action_types:
            report_lines.append(f"    - {action_type}: {count} records")
        report_lines.append("")

    # Write output file
    print(f"\n\nWriting output to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for record in all_sampled_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Total sampled: {total_sampled} records from {total_original}")
    print(f"Sampling rate: {total_sampled/total_original*100:.2f}%")

    # Add summary to report
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Total Original: {total_original} records")
    report_lines.append(f"Total Sampled: {total_sampled} records")
    report_lines.append(f"Sampling Rate: {total_sampled/total_original*100:.2f}%")
    report_lines.append(f"Target: {TOTAL_TARGET} records")
    report_lines.append(f"Difference: {total_sampled - TOTAL_TARGET} records ({(total_sampled - TOTAL_TARGET)/TOTAL_TARGET*100:.1f}%)")
    report_lines.append("")

    # Analyze overall action type distribution
    print("\n\nAnalyzing overall action type distribution...")
    overall_action_types = defaultdict(int)
    overall_sources = defaultdict(int)

    for record in all_sampled_records:
        source, action_type, _ = parse_id(record['id'])
        overall_action_types[action_type] += 1
        overall_sources[source] += 1

    report_lines.append("=" * 80)
    report_lines.append("OVERALL ACTION TYPE DISTRIBUTION")
    report_lines.append("=" * 80)
    report_lines.append(f"Total unique action types: {len(overall_action_types)}")
    report_lines.append("")

    sorted_action_types = sorted(overall_action_types.items(), key=lambda x: x[1], reverse=True)
    report_lines.append("Top 20 action types:")
    for i, (action_type, count) in enumerate(sorted_action_types[:20], 1):
        percentage = count / total_sampled * 100
        report_lines.append(f"  {i:2d}. {action_type:50s} {count:5d} ({percentage:5.2f}%)")

    report_lines.append("")
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
    print("Sampling completed successfully!")
    print("=" * 80)
    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"Report file: {REPORT_FILE}")
    print()


if __name__ == "__main__":
    main()
