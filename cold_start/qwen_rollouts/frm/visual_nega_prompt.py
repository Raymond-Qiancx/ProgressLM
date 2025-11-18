from __future__ import annotations
from typing import Dict, Any, List, Union


# System prompt for training mode (CoT generation with ground-truth)
VISUAL_DEMO_SYSTEM_PROMPT_TRAIN = """Given the below Task and the Ground-truth Partial Response, you only need to fill the content within <ref_think></ref_think> and <score_think></score_think> to complete the response and should not change the Ground-truth Partial Response's content within <ref></ref> and <score></score>. Note that you need to pretend you do not know the ground-truth answer and provide a coherent reasoning chain based on what is given.\n\nThe Task is:\n"""

# System prompt for normal inference mode
VISUAL_DEMO_SYSTEM_PROMPT_INFERENCE = """You are a progress estimator that evaluates the progress of the current state during an ongoing task based on a visual demonstration. The demonstration consists of a sequence of vision-based states and their corresponding progress value (ranging from 0% to 100%), showing how the task evolves from start to completion."""


# Default system prompt (use training mode)
VISUAL_DEMO_SYSTEM_PROMPT = VISUAL_DEMO_SYSTEM_PROMPT_TRAIN


VISUAL_DEMO_INSTRUCTION_PART1 = """Here is the demonstration:"""


VISUAL_DEMO_INSTRUCTION_PART2 = """Here is the current state that you need to estimate:"""


VISUAL_DEMO_INSTRUCTION_PART3 = """Your task:
1. Check the current state image carefully.
2. Analyze the visual demonstration to understand how the task progresses from start to completion.
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


def build_ground_truth_section(closest_idx: Union[int, str], progress_score: Union[str, float]) -> str:
    """
    Build the ground-truth section for training mode (CoT generation).

    Args:
        closest_idx: 1-based index of the closest visual_demo image, or "n/a"
        progress_score: Progress score (can be "33%", 0.33, or "n/a")

    Returns:
        Formatted ground-truth section string

    Example:
        >>> build_ground_truth_section(1, "8%")
        '**Critical Rule** The correct final progress score will be provided to you...'
    """
    # Handle "n/a" for closest_idx
    if isinstance(closest_idx, str) and closest_idx.lower() == "n/a":
        closest_idx_str = "n/a (no valid reference found)"
    else:
        closest_idx_str = f"The No. {closest_idx} demo image is the most relevant frame"

    # Handle "n/a" for progress_score
    if isinstance(progress_score, str) and progress_score.lower() == "n/a":
        progress_score_str = "n/a (no valid progress estimation)"
    else:
        # Normalize progress_score to percentage string format
        if isinstance(progress_score, str):
            # Already string, keep as is if it has %, otherwise add it
            if not progress_score.endswith('%'):
                try:
                    val = float(progress_score)
                    if val <= 1.0:
                        progress_score_str = f"{int(val * 100)}%"
                    else:
                        progress_score_str = f"{int(val)}%"
                except ValueError:
                    progress_score_str = progress_score  # Keep original
            else:
                progress_score_str = progress_score
        elif isinstance(progress_score, (int, float)):
            # Convert numeric to percentage
            if progress_score <= 1.0:
                progress_score_str = f"{int(progress_score * 100)}%"
            else:
                progress_score_str = f"{int(progress_score)}%"
        else:
            progress_score_str = str(progress_score)

    ground_truth_text = f"""\n\nGround-truth Partial Response:\n
    <ref_think></ref_think>
    <ref>{closest_idx_str}</ref>
    <score_think>n/a</score_think>
    <score>{progress_score_str}</score>\n\n
    You **must** only add content within <ref_think></ref_think> and must not change what we already provided in the Ground-truth Partial Response.
    When filling in <ref_think>, base your reasoning on what you can independently observe and infer from the state, rather than referencing this description as given information. Your reasoning should appear as your own discovery, not as something taken from an external hint.
"""

    return ground_truth_text


def build_visual_demo_prompt(
    task_goal: str,
    visual_demo_paths: List[str],
    total_steps: int,
    stage_to_estimate_path: str,
    closest_idx: Union[int, str] = None,
    progress_score: Union[str, float] = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    use_ground_truth: bool = True
) -> List[Dict[str, Any]]:
    """
    Build a multi-part prompt for Visual Demo progress estimation task.

    Prompt structure:
    1. Text: Task goal
    2. Text: "Here is the demonstration:"
    3. Images: visual_demo (N images, variable length)
    4. Text: Progress shift information (e.g., "<image> 0% <image> 25% <image> 50% <image> 75% <image> 100%")
    5. Text: "Here is the current state that you need to estimate:"
    6. Image: stage_to_estimate (1 image)
    7. Text: Ground-truth section (if use_ground_truth=True)
    8. Text: Task instructions

    Args:
        task_goal: Task goal description
        visual_demo_paths: List of paths to demonstration images (variable length)
        total_steps: Total number of steps (not including the initial 0% state)
        stage_to_estimate_path: Path to the current state image
        closest_idx: 1-based index of closest visual_demo (required if use_ground_truth=True)
        progress_score: Ground truth progress score (required if use_ground_truth=True)
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing
        use_ground_truth: Whether to include ground-truth section (default: True)

    Returns:
        List of message dicts for the model
    """
    msgs = []

    # Part 1: System prompts (Train + Inference) + Task goal
    combined_prompt = (
        VISUAL_DEMO_SYSTEM_PROMPT_TRAIN +
        VISUAL_DEMO_SYSTEM_PROMPT_INFERENCE +
        f"\n\nOur goal is {task_goal}."
    )
    msgs.append({"type": "text", "value": combined_prompt})

    # Part 2: Demonstration introduction
    msgs.append({"type": "text", "value": VISUAL_DEMO_INSTRUCTION_PART1})

    # Part 3: Visual demo images (variable length)
    for demo_img_path in visual_demo_paths:
        img_msg = {"type": "image", "value": demo_img_path}
        if min_pixels is not None:
            img_msg["min_pixels"] = min_pixels
        if max_pixels is not None:
            img_msg["max_pixels"] = max_pixels
        msgs.append(img_msg)

    # Part 4: Progress shift information
    progress_shifts = format_visual_demo_progress_shifts(total_steps)
    msgs.append({"type": "text", "value": f"The progress shifts across all given visual demos is: {progress_shifts}"})

    # Part 5: Current state introduction
    msgs.append({"type": "text", "value": VISUAL_DEMO_INSTRUCTION_PART2})

    # Part 6: Current state image (single image)
    stage_img_msg = {"type": "image", "value": stage_to_estimate_path}
    if min_pixels is not None:
        stage_img_msg["min_pixels"] = min_pixels
    if max_pixels is not None:
        stage_img_msg["max_pixels"] = max_pixels
    msgs.append(stage_img_msg)

    # Part 7: Task instructions
    msgs.append({"type": "text", "value": VISUAL_DEMO_INSTRUCTION_PART3})

    # Part 8: Ground-truth section (optional, for training mode) - placed at the end
    if use_ground_truth:
        if closest_idx is None or progress_score is None:
            raise ValueError("closest_idx and progress_score are required when use_ground_truth=True")
        ground_truth_section = build_ground_truth_section(closest_idx, progress_score)
        msgs.append({"type": "text", "value": ground_truth_section})

    return msgs


def build_visual_demo_prompt_from_item(
    item: Dict[str, Any],
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    use_ground_truth: bool = True
) -> List[Dict[str, Any]]:
    """
    Standalone function to build Visual Demo prompt from a dataset item.

    Args:
        item: Dataset item with required fields:
            - task_goal: str
            - visual_demo: List[str]
            - total_steps: int
            - stage_to_estimate: str
            - closest_idx: int (1-based, required if use_ground_truth=True)
            - progress_score: str or float (required if use_ground_truth=True)
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing
        use_ground_truth: Whether to include ground-truth section (default: True)

    Returns:
        List of message dicts for the model
    """
    return build_visual_demo_prompt(
        task_goal=item['task_goal'],
        visual_demo_paths=item['visual_demo'],
        total_steps=item['total_steps'],
        stage_to_estimate_path=item['stage_to_estimate'],
        closest_idx=item.get('closest_idx'),
        progress_score=item.get('progress_score'),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        use_ground_truth=use_ground_truth
    )
