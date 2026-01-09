import os
import json
from typing import Dict, Any, List, Tuple, Union


def parse_percentage(score_str: Union[str, int, float]) -> Union[float, str]:
    """
    Convert percentage string or numeric value to 0.0-1.0 range, or return "n/a".

    Examples:
        "33%" -> 0.33
        "100%" -> 1.0
        0.33 -> 0.33
        33 -> 0.33 (assumed to be percentage)
        "n/a" -> "n/a"
        "N/A" -> "n/a"

    Args:
        score_str: Progress score as string (with/without %), int, float, or "n/a"

    Returns:
        Float value in [0.0, 1.0] range, or "n/a" string
    """
    # Handle "n/a" string (case-insensitive)
    if isinstance(score_str, str):
        normalized = score_str.strip().lower()
        if normalized in ['n/a', 'na']:
            return "n/a"

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


def load_text_demo_dataset(
    dataset_path: str,
    num_inferences: int = 4,
    image_root: str = None
) -> List[Dict[str, Any]]:
    """
    Load Text Nega (Negative Sample) dataset from JSONL file and expand each sample N times.

    Expected format for each line (NEGATIVE SAMPLE VERSION):
    {
        "id": "h5_agilex_3rgb/10_packplate_2/2024_09_28-17_07_01-172863393748093664.00",
        "task_goal": "with both arms placing two cups into a rack",
        "text_demo": ["[left] move towards the green cup...", ...],
        "raw_task_goal": "with both arms placing two plates into a rack",
        "raw_text_demo": ["[left] move towards the green plate...", ...],
        "total_steps": "10",
        "stage_to_estimate": "camera_front_0062.jpg",
        "closest_idx": "n/a",
        "progress_score": "n/a",
        "rank": 0,
        "data_source": "h5_agilex_3rgb"
    }

    Image Path Construction:
        - If image_root is provided: IMAGE_ROOT/{id}/{stage_to_estimate}
        - Example: /data/CoMM/comm/h5_tienkung_xsens_1rgb/battery_insertion/camera_top_0474.jpg
        - If absolute path in data: use as-is (no modification)

    Args:
        dataset_path: Path to the JSONL dataset file
        num_inferences: Number of times to replicate each sample (default: 4)
        image_root: Root directory for image path construction (IMAGE_ROOT/{id}/{stage_to_estimate})

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

                # Validate required fields (for negative samples)
                required_fields = ['id', 'text_demo', 'stage_to_estimate', 'progress_score',
                                   'task_goal', 'closest_idx', 'total_steps', 'raw_text_demo', 'rank']
                missing_fields = [f for f in required_fields if f not in item]
                if missing_fields:
                    print(f"Warning: Line {line_num} missing fields {missing_fields}, skipping")
                    continue

                # Validate text_demo is a non-empty list
                if not isinstance(item['text_demo'], list) or len(item['text_demo']) == 0:
                    print(f"Warning: Line {line_num} has invalid text_demo (must be non-empty list), skipping")
                    continue

                # Validate task_goal is a non-empty string
                if not isinstance(item['task_goal'], str) or len(item['task_goal'].strip()) == 0:
                    print(f"Warning: Line {line_num} has invalid task_goal (must be non-empty string), skipping")
                    continue

                # Validate total_steps
                try:
                    total_steps = int(item['total_steps'])
                    if total_steps != len(item['text_demo']):
                        print(f"Warning: Line {line_num} total_steps ({total_steps}) doesn't match text_demo length ({len(item['text_demo'])}), skipping")
                        continue
                    item['total_steps'] = total_steps
                except (ValueError, TypeError):
                    print(f"Warning: Line {line_num} has invalid total_steps (must be integer), skipping")
                    continue

                # Validate closest_idx (1-based, must be in range [1, len(text_demo)], or "n/a")
                try:
                    if isinstance(item['closest_idx'], str) and item['closest_idx'].strip().lower() in ['n/a', 'na']:
                        item['closest_idx'] = "n/a"
                    else:
                        closest_idx = int(item['closest_idx'])
                        if not (1 <= closest_idx <= len(item['text_demo'])):
                            print(f"Warning: Line {line_num} has invalid closest_idx ({closest_idx}), must be 1-{len(item['text_demo'])} or 'n/a', skipping")
                            continue
                        item['closest_idx'] = closest_idx
                except (ValueError, TypeError):
                    print(f"Warning: Line {line_num} has invalid closest_idx (must be integer or 'n/a'), skipping")
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

                # Validate and convert progress_score (supports "33%", 0.33, or "n/a")
                try:
                    item['progress_score'] = parse_percentage(item['progress_score'])
                except (ValueError, TypeError) as e:
                    print(f"Warning: Line {line_num} has invalid progress_score ({item.get('progress_score')}): {e}, skipping")
                    continue

                # Validate raw_text_demo is a non-empty list (for negative samples)
                if not isinstance(item.get('raw_text_demo'), list) or len(item['raw_text_demo']) == 0:
                    print(f"Warning: Line {line_num} has invalid raw_text_demo (must be non-empty list), skipping")
                    continue

                # Validate rank (must be valid integer, 0-based index)
                try:
                    rank = int(item['rank'])
                    if rank < 0:
                        print(f"Warning: Line {line_num} has invalid rank ({rank}), must be >= 0, skipping")
                        continue
                    item['rank'] = rank
                except (ValueError, TypeError):
                    print(f"Warning: Line {line_num} has invalid rank (must be integer), skipping")
                    continue

                # Extract original_step from raw_text_demo using rank
                if rank >= len(item['raw_text_demo']):
                    print(f"Warning: Line {line_num} rank ({rank}) out of range for raw_text_demo (length {len(item['raw_text_demo'])}), skipping")
                    continue

                item['original_step'] = item['raw_text_demo'][rank]

                # Construct image path: IMAGE_ROOT / id / stage_to_estimate
                # If image_root is provided and path is not absolute, build path as: image_root/id/stage_to_estimate
                if image_root and not os.path.isabs(item['stage_to_estimate']):
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


def validate_image_path(item: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that the image path in stage_to_estimate exists.

    Args:
        item: Dataset item with 'stage_to_estimate' field

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
        item: Dataset item with 'text_demo' and 'stage_to_estimate' fields

    Returns:
        (text_demo, stage_to_estimate_path)
    """
    text_demo = item['text_demo']
    stage_to_estimate_path = item['stage_to_estimate']

    return text_demo, stage_to_estimate_path
