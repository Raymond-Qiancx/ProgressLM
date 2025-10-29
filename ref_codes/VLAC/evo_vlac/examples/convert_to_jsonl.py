#!/usr/bin/env python3
"""
Convert VLAC example images to Visual Demo JSONL format
将VLAC示例图像转换为Visual Demo JSONL格式
"""

import json
import os
from pathlib import Path

def generate_visual_demo_jsonl():
    """
    Generate JSONL dataset from VLAC example images

    Structure:
    - visual_demo: ref images (6 timesteps)
    - stage_to_estimate: test images (6 timesteps)
    - 3 camera views (0, 1, 2)
    - Total: 18 records (6 timesteps × 3 views)
    """

    # Base directory
    base_dir = Path(__file__).parent
    images_dir = base_dir / "images"

    # Task description
    task_goal = "Scoop the rice into the rice cooker."

    # Ref image sequence (visual_demo) - 6 timesteps
    ref_timesteps = [0, 100, 200, 300, 400, 457]
    ref_sequence_id = "599-521"

    # Test image sequence (stage_to_estimate) - 6 timesteps
    test_timesteps = [6, 44, 134, 139, 292, 354]
    test_sequence_id = "595-565"

    # Camera views
    camera_views = [0, 1, 2]

    # Total steps (from 0% to 100%, so total_steps = len(visual_demo) - 1)
    total_steps = len(ref_timesteps) - 1  # 5 steps

    # Generate dataset
    dataset = []

    for camera_idx in camera_views:
        # Build visual_demo paths for this camera view
        visual_demo = []
        for ts in ref_timesteps:
            filename = f"599-{ts}-521-{camera_idx}.jpg"
            # Use relative path from the examples directory
            rel_path = f"images/ref/{filename}"
            visual_demo.append(rel_path)

        # For each test timestep, create a record
        for test_idx, test_ts in enumerate(test_timesteps):
            # Build stage_to_estimate path
            stage_filename = f"595-{test_ts}-565-{camera_idx}.jpg"
            stage_path = f"images/test/{stage_filename}"

            # Calculate progress based on test sequence position
            # test_timesteps are: [6, 44, 134, 139, 292, 354]
            # Map them to progress: roughly 0%, 17%, 33%, 50%, 67%, 83%
            # Since we have 6 test images and 7 reference images (0-100%),
            # we estimate progress based on position in test sequence
            progress_ratio = test_idx / len(test_timesteps)  # 0.0, 0.167, 0.333, 0.5, 0.667, 0.833
            progress_score = f"{int(progress_ratio * 100)}%"

            # Find closest_idx in visual_demo (1-based indexing)
            # We map test position to closest reference position
            # test_idx=0 -> closest_idx=1 (start)
            # test_idx=5 -> closest_idx=6 (near end, but not final)
            closest_idx = min(test_idx + 1, len(visual_demo) - 1)  # 1-based, max is len(visual_demo)

            # Calculate delta (difference from closest stage)
            closest_progress = (closest_idx - 1) / total_steps  # Convert 1-based idx to 0-based, then to ratio
            delta_value = progress_ratio - closest_progress
            delta_sign = "+" if delta_value >= 0 else ""
            delta = f"{delta_sign}{int(delta_value * 100)}%"

            # Create record
            record = {
                "id": f"vlac_example/{test_sequence_id}/camera_{camera_idx}_test_{test_idx:02d}",
                "task_goal": task_goal,
                "visual_demo": visual_demo,
                "total_steps": str(total_steps),
                "stage_to_estimate": [stage_path],
                "closest_idx": str(closest_idx),
                "delta": delta,
                "progress_score": progress_score,
                "data_source": "vlac_example_scoop_rice"
            }

            dataset.append(record)

    # Save to JSONL
    output_file = base_dir / "vlac_example_visual_demo.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"✓ Generated {len(dataset)} records")
    print(f"✓ Saved to: {output_file}")
    print(f"\nDataset summary:")
    print(f"  - Task: {task_goal}")
    print(f"  - Camera views: {len(camera_views)}")
    print(f"  - Test timesteps per view: {len(test_timesteps)}")
    print(f"  - Total records: {len(dataset)}")
    print(f"  - Visual demo images per record: {len(visual_demo)}")

    # Print first 2 examples
    print(f"\nFirst 2 examples:")
    for i, record in enumerate(dataset[:2]):
        print(f"\n--- Record {i+1} ---")
        print(json.dumps(record, indent=2, ensure_ascii=False))

    return output_file, dataset


if __name__ == "__main__":
    generate_visual_demo_jsonl()
