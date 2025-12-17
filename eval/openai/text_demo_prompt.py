from __future__ import annotations
from typing import Dict, Any, List


# System prompt for inference mode
TEXT_DEMO_SYSTEM_PROMPT = """You are a progress estimator that evaluates the progress of the current state during an ongoing task based on a textual demonstration. The demonstration consists of a sequence of text-based steps and their corresponding progress value (ranging from 0% to 100%), showing how the task evolves from start to completion."""


TEXT_DEMO_INSTRUCTION_PART1 = """Here is the demonstration (If you see [left] [right], it indicates that this is a dual-arm robot, with the left and right arms working in coordination):"""


TEXT_DEMO_INSTRUCTION_PART2 = """Here is the current state that you need to estimate:"""


# TEXT_DEMO_INSTRUCTION_PART3 = """Your task:
# 1. Analyze the text_demo to understand how the task visually and conceptually progresses from start to completion.
# 2. Identify the step from the text_demo that are most visually and semantically similar to the current state image.
# 3. Compare the current state image with the chosen reference step to determine whether it represents an earlier or later stage.
# 4. Estimate the progress numerically as a floating-point value between 0% and 100%.

# Your response must strictly follow this format:
# <ref_think>Your reasoning for choosing the most similar text_demo step as the reference</ref_think>
# <ref>which text demo is most semantically similar to the current state, and output only the number of that text demo</ref>
# <score_think>Your reasoning for comparing the current state image with the reference step(s)</score_think>
# <score>Your final estimated progress score here</score>"""

# TEXT_DEMO_INSTRUCTION_PART3 = """Your task:
# 1. Check the current state image carefully.
# 2. Analyze the textual demonstration to understand how the task progresses from start to completion.
# 3. Identify the reference step from the textual demonstration that are most related to the current state image.
# 4. Compare the current state image with the chosen reference step, determining whether the image is behind or after the reference step.
# 5. Estimate the progress numerically as a floating-point value between 0% and 100%.
# 6. If you really cannot match the current state image to any of the steps from demonstration, you need to explain the reason within `<ref_think></ref_think>` and output "n/a" within `<ref></ref>`, `<score_think></score_think>`, and `<score></score>`.

# Your response **must** strictly follow this format:
# <ref_think>Reason for choosing the most related step from the demonstration as the reference or explanation of why the current state image does not match the task goal or any steps from demonstration</ref_think>
# <ref>which step from the textual demonstration is most related to the current state (output only the number of the step) or "n/a"</ref>
# <score_think>Reason for comparing the current state image with the reference step or "n/a"</score_think>
# <score>Your final estimated progress score or "n/a"</score>"""

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

# TEXT_DEMO_INSTRUCTION_PART3 = """Your task:
# 1. Read the task goal to understand the task objective and the entity being operated on.
# 2. Analyze the text_demo to understand how the task visually and conceptually progresses from start to completion.
# 3. Identify the step from the text_demo that are most visually and semantically similar to the current state image.
# 4. Compare the current state image with the chosen reference step to determine whether it represents an earlier or later stage.
# 5. Estimate the progress numerically as a floating-point value between 0% and 100%.

# Your response must strictly follow this format:
# <ref_think>Your reasoning for choosing the most similar text_demo step as the reference</ref_think>
# <ref>which text demo is most semantically similar to the current state, and output only the number of that text demo</ref>
# <score_think>Your reasoning for comparing the current state image with the reference step(s)</score_think>
# <score>Your final estimated progress score here</score>"""


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
    0. Text: TEXT_DEMO_SYSTEM_PROMPT
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

    # Part 0: System prompt
    msgs.append({"type": "text", "value": TEXT_DEMO_SYSTEM_PROMPT})

    # Part 1: Task goal
    msgs.append({"type": "text", "value": f"The overall task goal is {task_goal}."})

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
