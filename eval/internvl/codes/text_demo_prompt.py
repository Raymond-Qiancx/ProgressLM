"""
Text Demo prompt templates for InternVL progress estimation (think version).
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


# System prompt for inference mode
TEXT_DEMO_SYSTEM_PROMPT = """You are a progress estimator that evaluates the progress of the current state during an ongoing task based on a textual demonstration. The demonstration consists of a sequence of text-based steps and their corresponding progress value (ranging from 0% to 100%), showing how the task evolves from start to completion."""


TEXT_DEMO_INSTRUCTION_PART1 = """Here is the demonstration (If you see [left] [right], it indicates that this is a dual-arm robot, with the left and right arms working in coordination):"""


TEXT_DEMO_INSTRUCTION_PART2 = """Here is the current state that you need to estimate:"""


TEXT_DEMO_INSTRUCTION_PART3 = """Your task:
1. Read the task goal to understand the task objective and the entity being operated on.
2. Analyze the textual demonstration to understand how the task progresses from start to completion.
3. Examine the current state image carefully. If the target is incorrect (different from the object metioned in task goal) or you really cannot match the current image to any step in the demonstration, you must explain the reason within <ref_think></ref_think> and output "n/a" within <ref></ref>, <score_think></score_think>, and <score></score>.
4. If a match is possible, examine all steps in the textual demonstration, where each step represents an independent action. Identify the single step whose action is most closely related to the current state image. Then compare the current image with that reference step to determine whether it corresponds to an earlier or later stage, and finally estimate the overall progress as a floating-point value between 0% and 100%.

Your response **must** strictly follow this format:
<ref_think>
Explain the reason for selecting the most relevant step from the demonstration.
If the task target is incorrect, or the current state image cannot be matched to any demonstration step, explain why here.
</ref_think>

<ref>
If a valid matching step exists, output only the step number.
If the task target is incorrect or no step matches the current image, output only "n/a".
Please ensure that this is the same as the ref value you reasoned before.
</ref>

<score_think>
If a valid matching step exists, explain how you compare the current image with that step to judge progress.
If the task target is incorrect or no step matches the current image, output only "n/a".
</score_think>

<score>
If a valid matching step exists, output the estimated progress (0%–100%).
If the task target is incorrect or no step matches the current image, output only "n/a".
</score>
"""


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
    Build a prompt for Text Demo progress estimation task (InternVL format).

    Uses InternVL's required format:
    - Image-1: <image> prefix at the start
    - Then task description and instructions

    System prompt is NOT included here - it should be added by model.py.

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

    # 1. Image prefix at the start (InternVL required format)
    prompt_parts.append("Image-1: <image>")
    prompt_parts.append("")

    # 2. Task goal
    prompt_parts.append(f"The overall task goal is {task_goal}.")
    prompt_parts.append("")

    # 3. Demonstration introduction
    prompt_parts.append(TEXT_DEMO_INSTRUCTION_PART1)
    prompt_parts.append("")

    # 4. Formatted text_demo content with progress values
    formatted_demo = format_text_demo_with_progress(text_demo_list, total_steps)
    prompt_parts.append(formatted_demo)
    prompt_parts.append("")

    # 5. Current state reference (use Image-1 notation)
    prompt_parts.append(f"{TEXT_DEMO_INSTRUCTION_PART2} Image-1")
    prompt_parts.append("")

    # 6. Task instructions
    prompt_parts.append(TEXT_DEMO_INSTRUCTION_PART3)

    prompt_text = "\n".join(prompt_parts)

    return image_paths, prompt_text


def build_text_demo_prompt_from_item(
    item: Dict[str, Any],
) -> Tuple[List[str], str]:
    """
    Build Text Demo prompt from a dataset item (InternVL format).

    Args:
        item: Dataset item with required fields:
            - task_goal: str
            - text_demo: List[str]
            - total_steps: int
            - stage_to_estimate: str

    Returns:
        Tuple of (image_paths, prompt_text)
    """
    return build_text_demo_prompt(
        task_goal=item['task_goal'],
        text_demo_list=item['text_demo'],
        total_steps=item['total_steps'],
        stage_to_estimate_path=item['stage_to_estimate'],
    )
