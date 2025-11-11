#!/usr/bin/env python3
"""
Fix stage_to_estimate field format inconsistency in jsonl files.
Convert string values to single-element lists for schema consistency.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict

def fix_stage_to_estimate(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert stage_to_estimate from string to list if needed.

    Args:
        record: JSON record dictionary

    Returns:
        Fixed record with stage_to_estimate as list
    """
    if 'stage_to_estimate' in record:
        stage = record['stage_to_estimate']

        # Convert string to single-element list
        if isinstance(stage, str):
            record['stage_to_estimate'] = [stage]
        # If already a list, keep as is
        elif isinstance(stage, list):
            pass
        else:
            print(f"Warning: unexpected type for stage_to_estimate: {type(stage)}")

    return record


def fix_jsonl_file(file_path: str, backup: bool = True):
    """
    Fix stage_to_estimate format in a jsonl file.

    Args:
        file_path: Path to the jsonl file
        backup: Whether to create a backup before modifying
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    print(f"\nProcessing: {file_path}")

    # Create backup if requested
    if backup:
        backup_path = file_path.with_suffix('.jsonl.backup')
        shutil.copy2(file_path, backup_path)
        print(f"  Backup created: {backup_path}")

    # Read and fix records
    fixed_records = []
    string_count = 0
    list_count = 0
    total_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                total_count += 1

                # Track original type
                if 'stage_to_estimate' in record:
                    if isinstance(record['stage_to_estimate'], str):
                        string_count += 1
                    elif isinstance(record['stage_to_estimate'], list):
                        list_count += 1

                # Fix the record
                fixed_record = fix_stage_to_estimate(record)
                fixed_records.append(fixed_record)

    # Write fixed records back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in fixed_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"  Total records: {total_count}")
    print(f"  String → List conversions: {string_count}")
    print(f"  Already list: {list_count}")
    print(f"  ✓ Fixed successfully!")


def main():
    """Main function to fix all target files."""

    files_to_fix = [
        # Original data files
        "/gpfs/projects/p32958/chengxuan/ProgressLM/data/train/rl/final/raw/text_nega_rl.jsonl",
        "/gpfs/projects/p32958/chengxuan/ProgressLM/data/train/rl/final/raw/text_positive_6500_rl.jsonl",

        # Sampled data files
        "/gpfs/projects/p32958/chengxuan/ProgressLM/data/train/rl/final/sampled_rl_data_10k.jsonl",
        "/gpfs/projects/p32958/chengxuan/ProgressLM/data/train/rl/final/sampled_rl_data_5k.jsonl",
    ]

    print("=" * 80)
    print("Fixing stage_to_estimate Field Format")
    print("=" * 80)
    print("\nConverting all string values to list format for schema consistency")
    print("Example: \"camera_top_0454.jpg\" → [\"camera_top_0454.jpg\"]")

    for file_path in files_to_fix:
        fix_jsonl_file(file_path, backup=True)

    print("\n" + "=" * 80)
    print("All files fixed successfully!")
    print("=" * 80)
    print("\nBackup files created with .jsonl.backup extension")
    print("You can delete them after verifying the fix works correctly.")


if __name__ == "__main__":
    main()
