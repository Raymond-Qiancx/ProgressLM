"""
Adversarial Image Editing Dataset Loader

This module loads and processes adversarial image editing datasets for Qwen2-VL evaluation.
"""

import os
import json
from typing import List, Dict, Any, Tuple


def load_adversarial_editing_dataset(
    dataset_path: str,
    num_inferences: int = 1,
    image_root: str = None
) -> List[Dict[str, Any]]:
    """
    Load adversarial editing dataset from JSONL file.

    Expected JSONL format:
    {
        "id": "h5_tienkung_xsens_1rgb/tool_liftn_box_place/2024-09-28-10-54-08",
        "task_goal": "placing a tool into a box",
        "text_demo": ["reach for the tool", "grasp the tool", ...],
        "total_steps": 6,
        "stage_to_estimate": "camera_top_0556.jpg",
        "closest_idx": 6,
        "progress_score": "100%",
        "data_source": "h5_tienkung_xsens_1rgb"
    }

    Image Path Construction:
        Final path = image_root/id/stage_to_estimate
        Example: /path/to/images/h5_tienkung_xsens_1rgb/tool_liftn_box_place/2024-09-28-10-54-08/camera_top_0556.jpg

    Args:
        dataset_path: Path to JSONL file
        num_inferences: Number of times to replicate each sample (for multiple sampling)
        image_root: Root directory to prepend to image paths (e.g., "/path/to/images/")
                   Images will be loaded from: image_root/id/stage_to_estimate

    Returns:
        List of dataset items (expanded if num_inferences > 1)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)

                # Validate required fields
                required_fields = ['id', 'task_goal', 'text_demo', 'stage_to_estimate', 'closest_idx']
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    print(f"Warning: Line {line_num} missing fields {missing_fields}, skipping")
                    continue

                # Prepend image_root/id to stage_to_estimate if provided
                # Path format: image_root/id/stage_to_estimate
                if image_root:
                    # Store original path without prefix for metadata
                    item['stage_to_estimate_original'] = item['stage_to_estimate']
                    # Add prefix for actual loading: image_root/id/image_file
                    item_id = item['id']
                    item['stage_to_estimate'] = os.path.join(image_root, item_id, item['stage_to_estimate'])
                else:
                    item['stage_to_estimate_original'] = item['stage_to_estimate']

                # Ensure text_demo is a list
                if isinstance(item['text_demo'], str):
                    item['text_demo'] = json.loads(item['text_demo'])

                # Convert closest_idx to int if string
                if isinstance(item['closest_idx'], str):
                    item['closest_idx'] = int(item['closest_idx'])

                # Convert total_steps to int if string
                if 'total_steps' in item and isinstance(item['total_steps'], str):
                    item['total_steps'] = int(item['total_steps'])

                data.append(item)

            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} invalid JSON: {e}, skipping")
                continue
            except Exception as e:
                print(f"Warning: Line {line_num} error: {e}, skipping")
                continue

    print(f"Loaded {len(data)} samples from {dataset_path}")

    # Expand dataset if num_inferences > 1
    if num_inferences > 1:
        original_count = len(data)
        data = data * num_inferences
        print(f"Expanded dataset {num_inferences}x: {original_count} -> {len(data)} samples")

    return data


def validate_image_path(item: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that the image path exists.

    Args:
        item: Dataset item

    Returns:
        (is_valid, error_message)
    """
    stage_to_estimate = item.get('stage_to_estimate', '')

    if not stage_to_estimate:
        return False, "stage_to_estimate is empty"

    if not os.path.exists(stage_to_estimate):
        return False, f"Image not found: {stage_to_estimate}"

    return True, ""
