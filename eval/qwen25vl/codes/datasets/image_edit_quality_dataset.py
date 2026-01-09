"""
Image Edit Quality Evaluation Dataset Loader

This module loads and processes edited image datasets for quality evaluation.
"""

import os
import json
from typing import List, Dict, Any, Tuple


def load_image_edit_quality_dataset(
    dataset_path: str,
    image_root: str = None
) -> List[Dict[str, Any]]:
    """
    Load image edit quality evaluation dataset from JSONL file.

    Expected JSONL format:
    {
        "strategy": "Color Change",
        "prompt": "Change the green plate to red, ensuring the rest...",
        "raw_demo": "[left] grab the plate while [right] lift the plate",
        "response": "...",
        "meta_data": {
            "task_goal": "Place the two plates into the dish rack with both arms",
            "image": "camera_front_0227_edited.jpg",
            "text_demo": ["step1", "step2", ...],
            "id": "h5_agilex_3rgb/10_packplate_2/2024_09_28-17_42_01-172863177768757312.00",
            "data_source": "h5_agilex_3rgb",
            "status": "success"
        }
    }

    Image Path Construction:
        Final path = image_root/id/image
        Example: /path/to/edited_images/h5_agilex_3rgb/10_packplate_2/.../camera_front_0227_edited.jpg

    Args:
        dataset_path: Path to JSONL file
        image_root: Root directory to prepend to image paths (e.g., "/path/to/edited_images/")
                   Images will be loaded from: image_root/id/image

    Returns:
        List of dataset items with added 'edited_image_path' field
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
                required_fields = ['prompt', 'raw_demo', 'meta_data']
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    print(f"Warning: Line {line_num} missing fields {missing_fields}, skipping")
                    continue

                meta_data = item.get('meta_data', {})

                # Validate meta_data fields
                required_meta_fields = ['task_goal', 'image', 'text_demo', 'id', 'data_source']
                missing_meta_fields = [field for field in required_meta_fields if field not in meta_data]
                if missing_meta_fields:
                    print(f"Warning: Line {line_num} missing meta_data fields {missing_meta_fields}, skipping")
                    continue

                # Extract fields from meta_data to top level for easier access
                item['task_goal'] = meta_data.get('task_goal', '')
                item['text_demo'] = meta_data.get('text_demo', [])
                item['edited_image'] = meta_data.get('image', '')
                item['id'] = meta_data.get('id', '')
                item['data_source'] = meta_data.get('data_source', '')

                # Get editing strategy (optional)
                item['editing_strategy'] = item.get('strategy', 'Unknown')

                # Construct full image path: image_root/id/image
                if image_root:
                    item_id = item['id']
                    edited_image = item['edited_image']
                    item['edited_image_path'] = os.path.join(image_root, item_id, edited_image)
                else:
                    # If no image_root provided, assume 'image' is already full path
                    item['edited_image_path'] = item['edited_image']

                # Ensure text_demo is a list
                if isinstance(item['text_demo'], str):
                    try:
                        item['text_demo'] = json.loads(item['text_demo'])
                    except:
                        item['text_demo'] = [item['text_demo']]

                data.append(item)

            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} invalid JSON: {e}, skipping")
                continue
            except Exception as e:
                print(f"Warning: Line {line_num} error: {e}, skipping")
                continue

    print(f"Loaded {len(data)} samples from {dataset_path}")

    return data


def validate_edited_image_path(item: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that the edited image path exists.

    Args:
        item: Dataset item with 'edited_image_path' field

    Returns:
        (is_valid, error_message)
    """
    edited_image_path = item.get('edited_image_path', '')

    if not edited_image_path:
        return False, "edited_image_path is empty"

    if not os.path.exists(edited_image_path):
        return False, f"Edited image not found: {edited_image_path}"

    return True, ""
