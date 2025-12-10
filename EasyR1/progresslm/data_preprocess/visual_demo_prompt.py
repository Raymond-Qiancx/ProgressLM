from __future__ import annotations
from typing import Dict, Any, List


# System prompt for inference mode
VISUAL_DEMO_SYSTEM_PROMPT = """You are a progress estimator specializing in evaluating the progress of an ongoing task based on visual evidence. The demonstration consists of a sequence of video frames (images) showing how the task evolves from 0% (start) to 100% (completion). Your goal is to produce a human-like reasoning chain that logically supports the given progress score."""


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


def build_visual_demo_prompt(
    task_goal: str,
    visual_demo_paths: List[str],
    total_steps: int,
    stage_to_estimate_path: str,
    min_pixels: int | None = None,
    max_pixels: int | None = None
) -> List[Dict[str, Any]]:
    """
    Build a multi-part prompt for Visual Demo progress estimation task (inference mode).

    Prompt structure:
    1. Text: Task goal
    2. Text: "Here is the demonstration:"
    3. Images: visual_demo (N images, variable length)
    4. Text: Progress shift information (e.g., "<image> 0% <image> 25% <image> 50% <image> 75% <image> 100%")
    5. Text: "Here is the current state that you need to estimate:"
    6. Image: stage_to_estimate (1 image)
    7. Text: Task instructions

    Args:
        task_goal: Task goal description
        visual_demo_paths: List of paths to demonstration images (variable length)
        total_steps: Total number of steps (not including the initial 0% state)
        stage_to_estimate_path: Path to the current state image
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing

    Returns:
        List of message dicts for the model
    """
    msgs = []

    # Part 0: System prompt (included in user prompt)
    msgs.append({"type": "text", "value": VISUAL_DEMO_SYSTEM_PROMPT})

    # Part 1: Task goal
    msgs.append({"type": "text", "value": f"The overall task goal is {task_goal}"})

    # Part 2: Demonstration introduction
    msgs.append({"type": "text", "value": VISUAL_DEMO_INSTRUCTION_PART1})

    # Part 2: Visual demo images (variable length)
    for demo_img_path in visual_demo_paths:
        img_msg = {"type": "image", "value": demo_img_path}
        if min_pixels is not None:
            img_msg["min_pixels"] = min_pixels
        if max_pixels is not None:
            img_msg["max_pixels"] = max_pixels
        msgs.append(img_msg)

    # Part 3: Progress shift information
    progress_shifts = format_visual_demo_progress_shifts(total_steps)
    msgs.append({"type": "text", "value": f"The progress shifts across all given visual demos is: {progress_shifts}"})

    # Part 4: Current state introduction
    msgs.append({"type": "text", "value": VISUAL_DEMO_INSTRUCTION_PART2})

    # Part 5: Current state image (single image)
    stage_img_msg = {"type": "image", "value": stage_to_estimate_path}
    if min_pixels is not None:
        stage_img_msg["min_pixels"] = min_pixels
    if max_pixels is not None:
        stage_img_msg["max_pixels"] = max_pixels
    msgs.append(stage_img_msg)

    # Part 6: Task instructions
    msgs.append({"type": "text", "value": VISUAL_DEMO_INSTRUCTION_PART3})

    return msgs


def build_visual_demo_prompt_from_item(
    item: Dict[str, Any],
    min_pixels: int | None = None,
    max_pixels: int | None = None
) -> List[Dict[str, Any]]:
    """
    Standalone function to build Visual Demo prompt from a dataset item (inference mode).

    Args:
        item: Dataset item with required fields:
            - task_goal: str
            - visual_demo: List[str]
            - total_steps: int
            - stage_to_estimate: str
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing

    Returns:
        List of message dicts for the model
    """
    return build_visual_demo_prompt(
        task_goal=item['task_goal'],
        visual_demo_paths=item['visual_demo'],
        total_steps=item['total_steps'],
        stage_to_estimate_path=item['stage_to_estimate'],
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
