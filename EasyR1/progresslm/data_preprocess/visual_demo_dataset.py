import os
import json
from typing import Dict, Any, List, Tuple, Union


def parse_percentage(score_str: Union[str, int, float]) -> float:
    """
    Convert percentage string or numeric value to 0.0-1.0 range.

    Examples:
        "33%" -> 0.33
        "100%" -> 1.0
        0.33 -> 0.33
        33 -> 0.33 (assumed to be percentage)

    Args:
        score_str: Progress score as string (with/without %), int, or float

    Returns:
        Float value in [0.0, 1.0] range
    """
    if isinstance(score_str, (int, float)):
        # If numeric value > 1.0, assume it's percentage form (e.g., 33 means 33%)
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

    Expected format for each line (NEW VERSION):
    {
        "id": "h5_tienkung_xsens_1rgb/brick_piled_then_press_thrice/2024-10-17-10-53-16",
        "task_goal": "Put the blue block next to the purple block in front.",
        "visual_demo": ["camera_top_0000.jpg", "camera_top_0041.jpg", "camera_top_0068.jpg", "camera_top_0191.jpg", "camera_top_0394.jpg"],
        "total_steps": "4",
        "stage_to_estimate": ["camera_top_0013.jpg"],
        "closest_idx": "1",
        "delta": "+7%",
        "progress_score": "8%",
        "data_source": "robomind_h5_tienkung_xsens_1rgb"
    }

    Image Path Construction:
        - If image_root is provided: IMAGE_ROOT/{id}/{filename}
        - Example: /data/CoMM/comm/h5_tienkung_xsens_1rgb/brick_piled_then_press_thrice/2024-10-17-10-53-16/camera_top_0013.jpg
        - If absolute path in data: use as-is (no modification)

    Args:
        dataset_path: Path to the JSONL dataset file
        num_inferences: Number of times to replicate each sample (default: 4)
        image_root: Root directory for image path construction (IMAGE_ROOT/{id}/{filename})

    Returns:
        List of expanded dataset items (length = original_length * num_inferences)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Load raw data
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

                # Validate task_goal is a non-empty string
                if not isinstance(item['task_goal'], str) or len(item['task_goal'].strip()) == 0:
                    print(f"Warning: Line {line_num} has invalid task_goal (must be non-empty string), skipping")
                    continue

                # Validate visual_demo is a list
                if not isinstance(item['visual_demo'], list) or len(item['visual_demo']) == 0:
                    print(f"Warning: Line {line_num} has invalid visual_demo (must be non-empty list), skipping")
                    continue

                # Validate total_steps
                try:
                    total_steps = int(item['total_steps'])
                    # total_steps + 1 should equal visual_demo length (0% to 100%)
                    if total_steps + 1 != len(item['visual_demo']):
                        print(f"Warning: Line {line_num} total_steps+1 ({total_steps+1}) doesn't match visual_demo length ({len(item['visual_demo'])}), skipping")
                        continue
                    item['total_steps'] = total_steps
                except (ValueError, TypeError):
                    print(f"Warning: Line {line_num} has invalid total_steps (must be integer), skipping")
                    continue

                # Validate closest_idx (1-based, must be in range [1, total_steps+1])
                try:
                    closest_idx = int(item['closest_idx'])
                    if not (1 <= closest_idx <= total_steps + 1):
                        print(f"Warning: Line {line_num} has invalid closest_idx ({closest_idx}), must be 1-{total_steps+1}, skipping")
                        continue
                    item['closest_idx'] = closest_idx
                except (ValueError, TypeError):
                    print(f"Warning: Line {line_num} has invalid closest_idx (must be integer), skipping")
                    continue

                # Normalize stage_to_estimate to string format
                if isinstance(item['stage_to_estimate'], str):
                    stage_img = item['stage_to_estimate']
                elif isinstance(item['stage_to_estimate'], list) and len(item['stage_to_estimate']) == 1:
                    stage_img = item['stage_to_estimate'][0]
                else:
                    print(f"Warning: Line {line_num} has invalid stage_to_estimate (must be string or list with 1 element), skipping")
                    continue

                item['stage_to_estimate'] = stage_img

                # Validate and convert progress_score (supports "33%" or 0.33)
                try:
                    item['progress_score'] = parse_percentage(item['progress_score'])
                except (ValueError, TypeError) as e:
                    print(f"Warning: Line {line_num} has invalid progress_score ({item.get('progress_score')}): {e}, skipping")
                    continue

                # Construct image paths: IMAGE_ROOT / id / filename
                # If image_root is provided and path is not absolute, build path as: image_root/id/filename
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

    # Expand data: replicate each sample num_inferences times
    expanded_data = []
    for item in raw_data:
        for inference_idx in range(num_inferences):
            expanded_item = item.copy()
            expanded_item['_inference_idx'] = inference_idx  # Internal marker
            expanded_data.append(expanded_item)

    print(f"Expanded to {len(expanded_data)} samples (×{num_inferences})")

    return expanded_data


def validate_image_paths(item: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that all image paths in the item exist.

    Args:
        item: Dataset item with 'visual_demo' and 'stage_to_estimate' fields

    Returns:
        (is_valid, error_message)
    """
    # Check visual_demo images
    for img_path in item['visual_demo']:
        if not os.path.exists(img_path):
            return False, f"visual_demo image not found: {img_path}"

    # Check stage_to_estimate image (now a string, not a list)
    stage_img = item['stage_to_estimate']
    if not os.path.exists(stage_img):
        return False, f"stage_to_estimate image not found: {stage_img}"

    return True, ""


def get_image_paths(item: Dict[str, Any]) -> Tuple[List[str], str]:
    """
    Extract image paths from Visual Demo dataset item.

    Args:
        item: Dataset item with 'visual_demo' and 'stage_to_estimate' fields

    Returns:
        (visual_demo_paths, stage_to_estimate_path)
    """
    visual_demo_paths = item['visual_demo']
    # stage_to_estimate is now a string, not a list
    stage_to_estimate_path = item['stage_to_estimate']

    return visual_demo_paths, stage_to_estimate_path
