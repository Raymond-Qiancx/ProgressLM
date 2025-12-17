"""
Visual Demo dataset loading utilities for InternVL.
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
    # Handle "n/a" case
    if isinstance(score_str, str):
        score_str_stripped = score_str.strip().lower()
        if score_str_stripped in ["n/a", "na"]:
            return None

    if isinstance(score_str, (int, float)):
        if score_str > 1.0:
            return score_str / 100.0
        return float(score_str)

    # Handle string format
    score_str = str(score_str).strip()
    if score_str.endswith('%'):
        return float(score_str[:-1]) / 100.0
    else:
        val = float(score_str)
        if val > 1.0:
            return val / 100.0
        return val


def load_visual_demo_dataset(
    dataset_path: str,
    num_inferences: int = 4,
    image_root: str = None
) -> List[Dict[str, Any]]:
    """
    Load Visual Demo dataset from JSONL file and expand each sample N times.

    Expected format for each line:
    {
        "id": "h5_tienkung_xsens_1rgb/brick_piled_then_press_thrice/2024-10-17-10-53-16",
        "task_goal": "Put the blue block next to the purple block in front.",
        "visual_demo": ["camera_top_0000.jpg", ...],
        "total_steps": "4",
        "stage_to_estimate": ["camera_top_0013.jpg"],
        "closest_idx": "1",
        "delta": "+7%",
        "progress_score": "8%",
        "data_source": "robomind_h5_tienkung_xsens_1rgb"
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
                required_fields = ['id', 'task_goal', 'visual_demo', 'stage_to_estimate',
                                   'progress_score', 'closest_idx', 'total_steps']
                missing_fields = [f for f in required_fields if f not in item]
                if missing_fields:
                    print(f"Warning: Line {line_num} missing fields {missing_fields}, skipping")
                    continue

                # Validate task_goal
                if not isinstance(item['task_goal'], str) or len(item['task_goal'].strip()) == 0:
                    print(f"Warning: Line {line_num} has invalid task_goal, skipping")
                    continue

                # Validate visual_demo
                if not isinstance(item['visual_demo'], list) or len(item['visual_demo']) == 0:
                    print(f"Warning: Line {line_num} has invalid visual_demo, skipping")
                    continue

                # Validate total_steps
                try:
                    total_steps = int(item['total_steps'])
                    if total_steps + 1 != len(item['visual_demo']):
                        print(f"Warning: Line {line_num} total_steps+1 ({total_steps+1}) doesn't match visual_demo length ({len(item['visual_demo'])}), skipping")
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
                        if not (1 <= closest_idx <= total_steps + 1):
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

                # Construct image paths
                if image_root:
                    item['visual_demo'] = [
                        os.path.join(image_root, item['id'], img) if not os.path.isabs(img) else img
                        for img in item['visual_demo']
                    ]
                    if not os.path.isabs(item['stage_to_estimate']):
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


def validate_image_paths(item: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that all image paths in the item exist.

    Args:
        item: Dataset item

    Returns:
        (is_valid, error_message)
    """
    for img_path in item['visual_demo']:
        if not os.path.exists(img_path):
            return False, f"visual_demo image not found: {img_path}"

    stage_img = item['stage_to_estimate']
    if not os.path.exists(stage_img):
        return False, f"stage_to_estimate image not found: {stage_img}"

    return True, ""


def get_image_paths(item: Dict[str, Any]) -> Tuple[List[str], str]:
    """
    Extract image paths from Visual Demo dataset item.

    Args:
        item: Dataset item

    Returns:
        (visual_demo_paths, stage_to_estimate_path)
    """
    visual_demo_paths = item['visual_demo']
    stage_to_estimate_path = item['stage_to_estimate']

    return visual_demo_paths, stage_to_estimate_path
