"""
Text Object Replacement Dataset Loader

This module loads and processes text object replacement datasets for Qwen2-VL evaluation.
"""

import os
import json
from typing import List, Dict, Any, Tuple


def load_text_object_replacement_dataset(
    dataset_path: str,
    num_inferences: int = 1,
    image_root: str = None
) -> List[Dict[str, Any]]:
    """
    Load text object replacement dataset from JSONL file.

    Expected JSONL format:
    {
        "id": "h5_agilex_3rgb/10_packplate_2/2024_09_28-17_07_01-172863393748093664.00",
        "task_goal": "with both arms placing two plates into a rack",
        "text_demo": ["[left] move towards the green plate and [right] grab the green plate", ...],
        "total_steps": "10",
        "stage_to_estimate": "camera_front_0062.jpg",
        "closest_idx": 1,
        "progress_score": "10%",
        "data_source": "h5_agilex_3rgb"
    }

    Image Path Construction (if image_root provided):
        Final path = image_root/id/stage_to_estimate
        Example: /path/to/images/h5_agilex_3rgb/10_packplate_2/.../camera_front_0062.jpg

    Args:
        dataset_path: Path to JSONL file
        num_inferences: Number of times to replicate each sample (for multiple sampling)
        image_root: Optional root directory to prepend to image paths
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
                required_fields = ['id', 'task_goal', 'text_demo']
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    print(f"Warning: Line {line_num} missing fields {missing_fields}, skipping")
                    continue

                # If image_root is provided and stage_to_estimate exists, construct full path
                if image_root and 'stage_to_estimate' in item and item['stage_to_estimate']:
                    # Store original path without prefix for metadata
                    item['stage_to_estimate_original'] = item['stage_to_estimate']
                    # Add prefix for actual loading: image_root/id/image_file
                    item_id = item['id']
                    item['stage_to_estimate'] = os.path.join(image_root, item_id, item['stage_to_estimate'])
                elif 'stage_to_estimate' in item:
                    item['stage_to_estimate_original'] = item.get('stage_to_estimate', '')

                # Ensure text_demo is a list
                if isinstance(item['text_demo'], str):
                    item['text_demo'] = json.loads(item['text_demo'])

                # Convert closest_idx to int if string
                if 'closest_idx' in item and isinstance(item['closest_idx'], str):
                    item['closest_idx'] = int(item['closest_idx'])

                # Convert total_steps to int if string
                if 'total_steps' in item and isinstance(item['total_steps'], str):
                    try:
                        item['total_steps'] = int(item['total_steps'])
                    except ValueError:
                        # Keep as string if conversion fails
                        pass

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
    Validate that the image path exists (if provided).

    Args:
        item: Dataset item

    Returns:
        (is_valid, error_message)
    """
    stage_to_estimate = item.get('stage_to_estimate', '')

    # If no image path provided, it's still valid (text-only mode)
    if not stage_to_estimate:
        return True, ""

    if not os.path.exists(stage_to_estimate):
        return False, f"Image not found: {stage_to_estimate}"

    return True, ""
