"""
Shared utility functions for SFT data conversion.
"""
import json
import re
from typing import List, Dict, Any, Union


def format_text_demo_with_progress(text_demo: List[str], total_steps: int) -> str:
    """
    Format text_demo list into a structured string with step numbers and progress percentages.

    Args:
        text_demo: List of text demo steps
        total_steps: Total number of steps

    Returns:
        Formatted string like:
            Step 1. reach for the power bank
            The Progress for now is 33%.

            Step 2. insert the battery into the power bank
            The Progress for now is 66%.

    Example:
        >>> format_text_demo_with_progress(["reach", "insert", "remove"], 3)
        'Step 1. reach\\nThe Progress for now is 33%.\\n\\nStep 2. insert\\nThe Progress for now is 66%.\\n\\nStep 3. remove\\nThe Progress for now is 100%.'
    """
    formatted_parts = []

    for idx, step_text in enumerate(text_demo, start=1):
        # Calculate progress percentage for this step (1-based)
        progress_percentage = round((idx / total_steps) * 100)

        # Format: "Step X. <text>\nThe Progress for now is Y%."
        step_block = f"Step {idx}. {step_text}\nThe Progress for now is {progress_percentage}%."
        formatted_parts.append(step_block)

    # Join with double newline for separation
    return "\n\n".join(formatted_parts)


def format_visual_demo_progress_shifts(total_steps: int) -> str:
    """
    Format progress shifts for visual demo images based on total_steps.

    The progress shifts between images: 0% -> 25% -> 50% -> 75% -> 100% (for total_steps=4)

    Args:
        total_steps: Total number of steps (not including the initial 0% state)

    Returns:
        Formatted string with <image> tags and progress scores

    Example:
        >>> format_visual_demo_progress_shifts(4)
        '<image> 0% <image> 25% <image> 50% <image> 75% <image> 100%'
    """
    # Number of images is total_steps + 1 (0% to 100%)
    num_images = total_steps + 1
    parts = []

    for i in range(num_images):
        # Calculate progress percentage for this image
        progress_percentage = round((i / total_steps) * 100)
        parts.append(f"<image> {progress_percentage}%")

    return " ".join(parts)


def build_image_path(id: str, filename: str) -> str:
    """
    Build image path in format: {id}/{filename}

    Args:
        id: Sample ID (e.g., "h5_agilex_3rgb/10_packplate_2/2024_09_28-16_39_32-172863588507336032.00")
        filename: Image filename (e.g., "camera_front_0061.jpg")

    Returns:
        Image path string

    Example:
        >>> build_image_path("h5_agilex_3rgb/task/id", "img.jpg")
        'h5_agilex_3rgb/task/id/img.jpg'
    """
    return f"{id}/{filename}"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL file and return list of dictionaries.

    Args:
        path: Path to JSONL file

    Returns:
        List of parsed JSON objects

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON parsing fails
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON at line {line_num}: {e}")
                continue
    return data


def save_json(data: List[Dict[str, Any]], path: str, indent: int = 2) -> None:
    """
    Save data as JSON format (for LLaMA-Factory).

    Args:
        data: List of dictionaries to save
        path: Output file path
        indent: JSON indentation (default: 2)
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def build_match_key(id: str, stage_to_estimate: Union[str, List[str]]) -> str:
    """
    Build matching key for joining original data with CoT responses.

    Args:
        id: Sample ID
        stage_to_estimate: Stage image filename (can be str or list with 1 element)

    Returns:
        Matching key in format: "{id}|{stage_filename}"

    Example:
        >>> build_match_key("task/id", "img.jpg")
        'task/id|img.jpg'
        >>> build_match_key("task/id", ["img.jpg"])
        'task/id|img.jpg'
    """
    # Handle stage_to_estimate being a list or string
    if isinstance(stage_to_estimate, list):
        if len(stage_to_estimate) == 0:
            raise ValueError("stage_to_estimate list is empty")
        stage = stage_to_estimate[0]
    else:
        stage = stage_to_estimate

    return f"{id}|{stage}"


def normalize_stage_to_estimate(stage: Union[str, List[str]]) -> str:
    """
    Normalize stage_to_estimate to string format.

    Args:
        stage: Stage image filename (can be str or list with 1 element)

    Returns:
        Stage filename as string

    Example:
        >>> normalize_stage_to_estimate("img.jpg")
        'img.jpg'
        >>> normalize_stage_to_estimate(["img.jpg"])
        'img.jpg'
    """
    if isinstance(stage, list):
        if len(stage) == 0:
            raise ValueError("stage_to_estimate list is empty")
        return stage[0]
    return stage


def validate_image_tag_count(content: str, images: List[str]) -> bool:
    """
    Validate that the number of <image> tags in content matches the length of images array.

    Args:
        content: User message content string
        images: List of image paths

    Returns:
        True if counts match, False otherwise

    Example:
        >>> validate_image_tag_count("Text <image> more text", ["img1.jpg"])
        True
        >>> validate_image_tag_count("Text <image> <image>", ["img1.jpg"])
        False
    """
    image_tag_count = content.count('<image>')
    return image_tag_count == len(images)


def count_image_tags(content: str) -> int:
    """
    Count the number of <image> tags in content.

    Args:
        content: Content string

    Returns:
        Number of <image> tags found
    """
    return content.count('<image>')


def validate_xml_tags(content: str) -> bool:
    """
    Validate that assistant response contains all required XML tags.

    Args:
        content: Assistant response content

    Returns:
        True if all required tags are present, False otherwise
    """
    required_tags = ['<ref_think>', '</ref_think>', '<ref>', '</ref>',
                     '<score_think>', '</score_think>', '<score>', '</score>']
    return all(tag in content for tag in required_tags)


def extract_score_from_response(response: str) -> str:
    """
    Extract score value from assistant response.

    Args:
        response: Assistant response with XML tags

    Returns:
        Score string (e.g., "33%") or empty string if not found
    """
    match = re.search(r'<score>(.*?)</score>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def normalize_total_steps(total_steps: Union[str, int]) -> int:
    """
    Normalize total_steps to integer (can be string or int in data).

    Args:
        total_steps: Total steps value

    Returns:
        Integer value

    Example:
        >>> normalize_total_steps("8")
        8
        >>> normalize_total_steps(8)
        8
    """
    if isinstance(total_steps, str):
        return int(total_steps)
    return total_steps


def print_conversion_stats(stats: Dict[str, Any]) -> None:
    """
    Print conversion statistics in a formatted way.

    Args:
        stats: Dictionary containing conversion statistics
    """
    print("\n" + "="*50)
    print("Conversion Summary")
    print("="*50)
    print(f"Dataset: {stats.get('dataset_name', 'N/A')}")
    print(f"Total original samples: {stats.get('total_original', 0)}")
    print(f"Total CoT responses loaded: {stats.get('total_cot', 0)}")
    print(f"Matched samples: {stats.get('matched', 0)}")
    print(f"  - Success status: {stats.get('success', 0)}")
    print(f"  - Failed/Other status: {stats.get('failed_status', 0)}")
    print(f"Unmatched samples: {stats.get('unmatched', 0)}")
    print(f"Final output samples: {stats.get('output_samples', 0)}")
    print(f"Image tag validation: {stats.get('validation_passed', 0)}/{stats.get('output_samples', 0)} passed")
    print(f"Output file: {stats.get('output_file', 'N/A')}")
    print("="*50 + "\n")
