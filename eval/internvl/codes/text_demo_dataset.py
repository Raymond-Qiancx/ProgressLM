"""
Text Demo dataset loading utilities for InternVL.
(Model-agnostic, adapted from Qwen2.5VL implementation)
"""

import os
import json
from typing import Dict, Any, List, Tuple, Union


def parse_percentage(score_str: Union[str, int, float]) -> Union[float, None]:
    """
    Convert percentage string or numeric value to 0.0-1.0 range.

    Examples:
        "33%" -> 0.33
        "100%" -> 1.0
        0.33 -> 0.33
        33 -> 0.33 (assumed to be percentage)
        "n/a" -> None

    Args:
        score_str: Progress score as string (with/without %), int, or float, or "n/a"

    Returns:
        Float value in [0.0, 1.0] range, or None if "n/a"
    """
    if isinstance(score_str, str):
        score_str_stripped = score_str.strip().lower()
        if score_str_stripped in ["n/a", "na"]:
            return None

    if isinstance(score_str, (int, float)):
        if score_str > 1.0:
            return score_str / 100.0
        return float(score_str)

    score_str = str(score_str).strip()
    if score_str.endswith('%'):
        return float(score_str[:-1]) / 100.0
    else:
        val = float(score_str)
        if val > 1.0:
            return val / 100.0
        return val


def load_text_demo_dataset(
    dataset_path: str,
    num_inferences: int = 4,
    image_root: str = None
) -> List[Dict[str, Any]]:
    """
    Load Text Demo dataset from JSONL file and expand each sample N times.

    Expected format for each line:
    {
        "id": "h5_tienkung_xsens_1rgb/battery_insertion_with_pullout/2024-09-19-10-35-18",
        "task_goal": "inserting a battery into a power bank and then removing it",
        "text_demo": ["reach for the power bank", "insert the battery", "remove the battery"],
        "total_steps": 3,
        "stage_to_estimate": "camera_top_0474.jpg",
        "closest_idx": 1,
        "progress_score": "33%",
        "data_source": "h5_tienkung_xsens_1rgb"
    }

    Args:
        dataset_path: Path to the JSONL dataset file
        num_inferences: Number of times to replicate each sample (default: 4)
        image_root: Root directory for image path construction

    Returns:
        List of expanded dataset items
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    raw_data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)

                # Validate required fields
                required_fields = ['id', 'text_demo', 'stage_to_estimate', 'progress_score',
                                   'task_goal', 'closest_idx', 'total_steps']
                missing_fields = [f for f in required_fields if f not in item]
                if missing_fields:
                    print(f"Warning: Line {line_num} missing fields {missing_fields}, skipping")
                    continue

                # Validate text_demo
                if not isinstance(item['text_demo'], list) or len(item['text_demo']) == 0:
                    print(f"Warning: Line {line_num} has invalid text_demo, skipping")
                    continue

                # Validate task_goal
                if not isinstance(item['task_goal'], str) or len(item['task_goal'].strip()) == 0:
                    print(f"Warning: Line {line_num} has invalid task_goal, skipping")
                    continue

                # Validate total_steps
                try:
                    total_steps = int(item['total_steps'])
                    if total_steps != len(item['text_demo']):
                        print(f"Warning: Line {line_num} total_steps ({total_steps}) doesn't match text_demo length ({len(item['text_demo'])}), skipping")
                        continue
                    item['total_steps'] = total_steps
                except (ValueError, TypeError):
                    print(f"Warning: Line {line_num} has invalid total_steps, skipping")
                    continue

                # Validate closest_idx
                closest_idx_raw = item['closest_idx']
                if isinstance(closest_idx_raw, str) and closest_idx_raw.strip().lower() in ["n/a", "na"]:
                    item['closest_idx'] = None
                else:
                    try:
                        closest_idx = int(closest_idx_raw)
                        if not (1 <= closest_idx <= len(item['text_demo'])):
                            print(f"Warning: Line {line_num} has invalid closest_idx ({closest_idx}), skipping")
                            continue
                        item['closest_idx'] = closest_idx
                    except (ValueError, TypeError):
                        print(f"Warning: Line {line_num} has invalid closest_idx, skipping")
                        continue

                # Normalize stage_to_estimate
                if isinstance(item['stage_to_estimate'], str):
                    stage_img = item['stage_to_estimate']
                elif isinstance(item['stage_to_estimate'], list) and len(item['stage_to_estimate']) == 1:
                    stage_img = item['stage_to_estimate'][0]
                else:
                    print(f"Warning: Line {line_num} has invalid stage_to_estimate, skipping")
                    continue
                item['stage_to_estimate'] = stage_img

                # Parse progress_score
                try:
                    parsed_score = parse_percentage(item['progress_score'])
                    item['progress_score'] = parsed_score
                except (ValueError, TypeError) as e:
                    print(f"Warning: Line {line_num} has invalid progress_score: {e}, skipping")
                    continue

                # Construct image path
                if image_root and not os.path.isabs(item['stage_to_estimate']):
                    item['stage_to_estimate'] = os.path.join(image_root, item['id'], item['stage_to_estimate'])

                raw_data.append(item)

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON at line {line_num}: {e}")
                continue

    print(f"Loaded {len(raw_data)} raw samples from {dataset_path}")
    if image_root:
        print(f"Image root directory: {image_root}")

    # Expand data
    expanded_data = []
    for item in raw_data:
        for inference_idx in range(num_inferences):
            expanded_item = item.copy()
            expanded_item['_inference_idx'] = inference_idx
            expanded_data.append(expanded_item)

    print(f"Expanded to {len(expanded_data)} samples (×{num_inferences})")

    return expanded_data


def validate_image_path(item: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that the image path in stage_to_estimate exists.

    Args:
        item: Dataset item

    Returns:
        (is_valid, error_message)
    """
    stage_img = item['stage_to_estimate']
    if not os.path.exists(stage_img):
        return False, f"stage_to_estimate image not found: {stage_img}"

    return True, ""


def get_text_and_image(item: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract text_demo and stage_to_estimate from Text Demo dataset item.

    Args:
        item: Dataset item

    Returns:
        (text_demo, stage_to_estimate_path)
    """
    text_demo = item['text_demo']
    stage_to_estimate_path = item['stage_to_estimate']

    return text_demo, stage_to_estimate_path
