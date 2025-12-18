"""
Visual Demo prompt templates for InternVL progress estimation (think version).
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


# System prompt for inference mode
VISUAL_DEMO_SYSTEM_PROMPT = """You are a progress estimator that evaluates the progress of the current state during an ongoing task based on a visual demonstration. The demonstration consists of a sequence of vision-based states and their corresponding progress value (ranging from 0% to 100%), showing how the task evolves from start to completion."""


VISUAL_DEMO_INSTRUCTION_PART1 = """Here is the demonstration:"""


VISUAL_DEMO_INSTRUCTION_PART2 = """Here is the current state that you need to estimate:"""


VISUAL_DEMO_INSTRUCTION_PART3 = """Your task:
1. Check the current state image carefully.
2. Analyze the overall task goal and visual demonstration to understand how the task progresses from start to completion.
3. Identify the reference states from the visual demonstration that are most related to the current state image.
4. Compare the current state image with the chosen reference state, determining whether the image is behind or after the reference state.
5. Estimate the progress numerically as a floating-point value between 0% and 100%.
6. If you really cannot match the current state image to any of the states from demonstration, you need to explain the reason within `<ref_think></ref_think>` and output "n/a" within `<ref></ref>`, `<score_think></score_think>`, and `<score></score>`.

Your response **must** strictly follow this format:
<ref_think>Reason for choosing the most related state from the demonstration as the reference or explanation of why the current state image does not match the task goal or any steps from demonstration</ref_think>
<ref>which state from the visual demonstration is most related to the current state (output only the number of the state) or "n/a"</ref>
<score_think>Reason for comparing the current state image with the reference state or "n/a"</score_think>
<score>Your final estimated progress score or "n/a"</score>"""


def build_image_prefix(num_images: int) -> str:
    """
    Build InternVL image prefix in required format: Image-N: <image>

    Args:
        num_images: Total number of images

    Returns:
        Formatted string with Image-N: <image> for each image

    Example:
        >>> build_image_prefix(3)
        'Image-1: <image>\nImage-2: <image>\nImage-3: <image>'
    """
    parts = [f"Image-{i + 1}: <image>" for i in range(num_images)]
    return "\n".join(parts)


def format_progress_description(total_steps: int) -> str:
    """
    Format progress description for demonstration images.

    Args:
        total_steps: Total number of steps (not including the initial 0% state)

    Returns:
        Formatted string describing progress of each image

    Example:
        >>> format_progress_description(4)
        'Image-1 (0%) -> Image-2 (25%) -> Image-3 (50%) -> Image-4 (75%) -> Image-5 (100%)'
    """
    num_images = total_steps + 1
    parts = []

    for i in range(num_images):
        progress_percentage = round((i / total_steps) * 100)
        parts.append(f"Image-{i + 1} ({progress_percentage}%)")

    return " -> ".join(parts)


def build_visual_demo_prompt(
    task_goal: str,
    visual_demo_paths: List[str],
    total_steps: int,
    stage_to_estimate_path: str,
) -> Tuple[List[str], str]:
    """
    Build a prompt for Visual Demo progress estimation task (InternVL format).

    Uses InternVL's required format:
    - All Image-N: <image> prefixes at the start
    - Then task description and instructions

    System prompt is NOT included here - it should be added by model.py.

    Args:
        task_goal: Task goal description
        visual_demo_paths: List of paths to demonstration images
        total_steps: Total number of steps (not including the initial 0% state)
        stage_to_estimate_path: Path to the current state image

    Returns:
        Tuple of (all_image_paths, prompt_text)
    """
    # Collect all image paths: demo images + current state image
    all_image_paths = visual_demo_paths + [stage_to_estimate_path]
    num_total_images = len(all_image_paths)
    current_image_idx = num_total_images  # Last image is current state

    # Build text prompt
    prompt_parts = []

    # 1. Image prefix at the start (InternVL required format)
    prompt_parts.append(build_image_prefix(num_total_images))
    prompt_parts.append("")

    # 2. Task goal
    prompt_parts.append(f"The overall task goal is {task_goal}")
    prompt_parts.append("")

    # 3. Demonstration with progress description
    prompt_parts.append(VISUAL_DEMO_INSTRUCTION_PART1)
    progress_desc = format_progress_description(total_steps)
    prompt_parts.append(f"The progress shifts across the demonstration images: {progress_desc}")
    prompt_parts.append("")

    # 4. Current state reference (use Image-N notation)
    prompt_parts.append(f"{VISUAL_DEMO_INSTRUCTION_PART2} Image-{current_image_idx}")
    prompt_parts.append("")

    # 5. Task instructions
    prompt_parts.append(VISUAL_DEMO_INSTRUCTION_PART3)

    prompt_text = "\n".join(prompt_parts)

    return all_image_paths, prompt_text


def build_visual_demo_prompt_from_item(
    item: Dict[str, Any],
) -> Tuple[List[str], str]:
    """
    Build Visual Demo prompt from a dataset item (InternVL format).

    Args:
        item: Dataset item with required fields:
            - task_goal: str
            - visual_demo: List[str]
            - total_steps: int
            - stage_to_estimate: str

    Returns:
        Tuple of (all_image_paths, prompt_text)
    """
    return build_visual_demo_prompt(
        task_goal=item['task_goal'],
        visual_demo_paths=item['visual_demo'],
        total_steps=item['total_steps'],
        stage_to_estimate_path=item['stage_to_estimate'],
    )
