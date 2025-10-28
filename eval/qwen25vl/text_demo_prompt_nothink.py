from __future__ import annotations
from typing import Dict, Any, List


# System prompt for inference mode
TEXT_DEMO_SYSTEM_PROMPT = """You are a progress estimator that evaluates the progress of an ongoing task based on a textual demonstration of its step-by-step progression.

The demonstration consists of a sequence of text instructions (text_demo), each describing one step of the process.
Each step explicitly states the corresponding progress value (ranging from 0% to 100%), showing how the task evolves from start to completion."""


TEXT_DEMO_INSTRUCTION_PART1 = """Here is the demonstration:"""


TEXT_DEMO_INSTRUCTION_PART2 = """Here is the current state that you need to estimate:"""


TEXT_DEMO_INSTRUCTION_PART3 = """Your task:
1. Analyze the text_demo to understand how the task visually and conceptually progresses from start to completion.
2. Identify the step from the text_demo that are most visually and semantically similar to the current state image.
3. Compare the current state image with the chosen reference step to determine whether it represents an earlier or later stage.
4. Estimate the progress numerically as a floating-point value between 0% and 100%.

Your answer only needs to output the final progress score you estimated."""


def format_text_demo_with_progress(text_demo_list: List[str], total_steps: int) -> str:
    """
    Format text_demo list into a structured string with step numbers and progress percentages.

    Args:
        text_demo_list: List of text demo steps (e.g., ["step1", "step2", "step3"])
        total_steps: Total number of steps

    Returns:
        Formatted string like:
            Step 1. reach for the power bank
            The Progress for now is 33%.

            Step 2. insert the battery into the power bank
            The Progress for now is 66%.

            Step 3. remove the battery from the power bank
            The Progress for now is 100%.

    Example:
        >>> format_text_demo_with_progress(["reach", "insert", "remove"], 3)
        'Step 1. reach\nThe Progress for now is 33%.\n\nStep 2. insert\nThe Progress for now is 66%.\n\nStep 3. remove\nThe Progress for now is 100%.'
    """
    formatted_parts = []

    for idx, step_text in enumerate(text_demo_list, start=1):
        # Calculate progress percentage for this step (1-based)
        progress_percentage = round((idx / total_steps) * 100)

        # Format: "Step X. <text>\nThe Progress for now is Y%."
        step_block = f"Step {idx}. {step_text}\nThe Progress for now is {progress_percentage}%."
        formatted_parts.append(step_block)

    # Join with double newline for separation
    return "\n\n".join(formatted_parts)


def build_text_demo_prompt(
    task_goal: str,
    text_demo_list: List[str],
    total_steps: int,
    stage_to_estimate_path: str,
    min_pixels: int | None = None,
    max_pixels: int | None = None
) -> List[Dict[str, Any]]:
    """
    Build a multi-part prompt for Text Demo progress estimation task (inference mode).

    Prompt structure:
    1. Text: "Our goal is {task_goal}"
    2. Text: "Here is the demonstration:"
    3. Text: Formatted text_demo with step numbers and progress values
    4. Text: "Here is the current state that you need to estimate:"
    5. Image: stage_to_estimate
    6. Text: Task instructions

    Args:
        task_goal: Task goal description
        text_demo_list: List of text demo steps
        total_steps: Total number of steps
        stage_to_estimate_path: Path to the current state image
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing

    Returns:
        List of message dicts for the model
    """
    msgs = []

    # Part 1: Task goal
    msgs.append({"type": "text", "value": f"Our goal is {task_goal}."})

    # Part 2: Demonstration introduction
    msgs.append({"type": "text", "value": TEXT_DEMO_INSTRUCTION_PART1})

    # Part 3: Formatted text_demo content with progress values
    formatted_demo = format_text_demo_with_progress(text_demo_list, total_steps)
    msgs.append({"type": "text", "value": formatted_demo})

    # Part 4: Current state introduction
    msgs.append({"type": "text", "value": TEXT_DEMO_INSTRUCTION_PART2})

    # Part 5: Current state image (single image)
    stage_img_msg = {"type": "image", "value": stage_to_estimate_path}
    if min_pixels is not None:
        stage_img_msg["min_pixels"] = min_pixels
    if max_pixels is not None:
        stage_img_msg["max_pixels"] = max_pixels
    msgs.append(stage_img_msg)

    # Part 6: Task instructions
    msgs.append({"type": "text", "value": TEXT_DEMO_INSTRUCTION_PART3})

    return msgs


def build_text_demo_prompt_from_item(
    item: Dict[str, Any],
    min_pixels: int | None = None,
    max_pixels: int | None = None
) -> List[Dict[str, Any]]:
    """
    Standalone function to build Text Demo prompt from a dataset item (inference mode).

    Args:
        item: Dataset item with required fields:
            - task_goal: str
            - text_demo: List[str]
            - total_steps: int
            - stage_to_estimate: str
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing

    Returns:
        List of message dicts for the model
    """
    return build_text_demo_prompt(
        task_goal=item['task_goal'],
        text_demo_list=item['text_demo'],
        total_steps=item['total_steps'],
        stage_to_estimate_path=item['stage_to_estimate'],
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
