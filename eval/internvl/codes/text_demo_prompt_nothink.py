"""
Text Demo prompt templates for InternVL progress estimation (nothink version).
Simplified version that only outputs the progress score.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


# System prompt for inference mode
TEXT_DEMO_SYSTEM_PROMPT = """You are a progress estimator that evaluates the progress of the current state during an ongoing task based on a textual demonstration. The demonstration consists of a sequence of text-based steps and their corresponding progress value (ranging from 0% to 100%), showing how the task evolves from start to completion."""


TEXT_DEMO_INSTRUCTION_PART1 = """Here is the demonstration (If you see [left] [right], it indicates that this is a dual-arm robot, with the left and right arms working in coordination):"""


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
        text_demo_list: List of text demo steps
        total_steps: Total number of steps

    Returns:
        Formatted string with step numbers and progress values
    """
    formatted_parts = []

    for idx, step_text in enumerate(text_demo_list, start=1):
        progress_percentage = round((idx / total_steps) * 100)
        step_block = f"Step {idx}. {step_text}\nThe Progress for now is {progress_percentage}%."
        formatted_parts.append(step_block)

    return "\n\n".join(formatted_parts)


def build_text_demo_prompt(
    task_goal: str,
    text_demo_list: List[str],
    total_steps: int,
    stage_to_estimate_path: str,
) -> Tuple[List[str], str]:
    """
    Build a prompt for Text Demo progress estimation task (nothink version).

    Args:
        task_goal: Task goal description
        text_demo_list: List of text demo steps
        total_steps: Total number of steps
        stage_to_estimate_path: Path to the current state image

    Returns:
        Tuple of (image_paths, prompt_text)
    """
    # Only one image for text demo
    image_paths = [stage_to_estimate_path]

    # Build text prompt
    prompt_parts = []

    # Demonstration introduction
    prompt_parts.append(TEXT_DEMO_INSTRUCTION_PART1)
    prompt_parts.append("")

    # Formatted text_demo content with progress values
    formatted_demo = format_text_demo_with_progress(text_demo_list, total_steps)
    prompt_parts.append(formatted_demo)
    prompt_parts.append("")

    # Current state introduction
    prompt_parts.append(TEXT_DEMO_INSTRUCTION_PART2)
    prompt_parts.append("")

    # Task instructions
    prompt_parts.append(TEXT_DEMO_INSTRUCTION_PART3)

    prompt_text = "\n".join(prompt_parts)

    return image_paths, prompt_text


def build_text_demo_prompt_from_item(
    item: Dict[str, Any],
) -> Tuple[List[str], str]:
    """
    Build Text Demo prompt from a dataset item (nothink version).

    Args:
        item: Dataset item

    Returns:
        Tuple of (image_paths, prompt_text)
    """
    return build_text_demo_prompt(
        task_goal=item['task_goal'],
        text_demo_list=item['text_demo'],
        total_steps=item['total_steps'],
        stage_to_estimate_path=item['stage_to_estimate'],
    )
