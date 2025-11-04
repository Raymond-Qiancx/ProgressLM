from __future__ import annotations
from typing import Dict, Any, List, Union


# System prompt for training mode (CoT generation with ground-truth)
TEXT_DEMO_SYSTEM_PROMPT_TRAIN = """You are an expert AI analyst specializing in visual task-progress evaluations. Your objective is not to estimate from scratch. Instead, your task is to construct a perfect, human-like chain of thought that logically explains and justifies a known, ground-truth progress score. Your entire response must read as if you are deducing the conclusion independently from visual analysis alone."""


# System prompt for normal inference mode
TEXT_DEMO_SYSTEM_PROMPT_INFERENCE = """You are a progress estimator that evaluates the progress of an ongoing task based on a textual demonstration of its step-by-step progression.

The demonstration consists of a sequence of text instructions (text_demo), each describing one step of the process.
Each step explicitly states the corresponding progress value (ranging from 0% to 100%), showing how the task evolves from start to completion."""


# Default system prompt (use inference mode)
TEXT_DEMO_SYSTEM_PROMPT = TEXT_DEMO_SYSTEM_PROMPT_INFERENCE


TEXT_DEMO_INSTRUCTION_PART1 = """Here is the demonstration:"""


TEXT_DEMO_INSTRUCTION_PART2 = """Here is the current state that you need to estimate:"""


TEXT_DEMO_INSTRUCTION_PART3 = """**Abnormal Situation Handling:**
If you detect any of the following abnormal situations:
- The current state does not match the task goal or any demo steps
- The operation appears to have failed or resulted in an error state
- You must output "n/a" for both `<ref>` and `<score>`. In your reasoning sections, clearly explain why the situation is abnormal and why no valid progress estimation can be made.

Your response must strictly follow this format:
<ref_think>Your reasoning for choosing the most similar text_demo step as the reference, OR explanation of why the situation is abnormal and no reference can be identified</ref_think>
<ref>which text demo is most semantically similar to the current state (output only the number), OR "n/a" if abnormal situation detected</ref>
<score_think>Your reasoning for comparing the current state image with the reference step, OR explanation of why no valid progress score can be assigned</score_think>
<score>Your final estimated progress score, OR "n/a" if abnormal situation detected</score>"""


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


def build_ground_truth_section(closest_idx: Union[int, str], progress_score: Union[str, float]) -> str:
    """
    Build the ground-truth section for training mode (CoT generation).

    Args:
        closest_idx: 1-based index of the closest text_demo step, or "n/a"
        progress_score: Progress score (can be "33%", 0.33, or "n/a")

    Returns:
        Formatted ground-truth section string

    Example:
        >>> build_ground_truth_section(1, "33%")
        '**Critical Rule** The correct final progress score will be provided to you...'
    """
    # Handle "n/a" for closest_idx
    if isinstance(closest_idx, str) and closest_idx.lower() == "n/a":
        closest_idx_str = "n/a (no valid reference found)"
    else:
        closest_idx_str = f"The No. {closest_idx} text demo is the most relevant one"

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

    ground_truth_text = f"""**Critical Rule** The correct final progress score will be provided to you. However, you must **never** reveal or imply that you already know the answer. Your reasoning must appear as a fully original, independent visual analysis derived from the images.

**Ground-Truth Progress Result**
Closest Reference Frame: {closest_idx_str}
Final Progress Score to Justify: {progress_score_str}
Indicating that we are now facing abnormal situation, your `<ref>` and `<score>` should both be "n/a"."""

    return ground_truth_text


def build_text_demo_prompt(
    task_goal: str,
    text_demo_list: List[str],
    total_steps: int,
    stage_to_estimate_path: str,
    closest_idx: Union[int, str] = None,
    progress_score: Union[str, float] = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    use_ground_truth: bool = True
) -> List[Dict[str, Any]]:
    """
    Build a multi-part prompt for Text Demo progress estimation task.

    Prompt structure:
    1. Text: "Our goal is {task_goal}"
    2. Text: "Here is the demonstration:"
    3. Text: Formatted text_demo with step numbers and progress values
    4. Text: "Here is the current state that you need to estimate:"
    5. Image: stage_to_estimate
    6. Text: Ground-truth section (if use_ground_truth=True)
    7. Text: Task instructions

    Args:
        task_goal: Task goal description
        text_demo_list: List of text demo steps
        total_steps: Total number of steps
        stage_to_estimate_path: Path to the current state image
        closest_idx: 1-based index of closest text_demo (required if use_ground_truth=True)
        progress_score: Ground truth progress score (required if use_ground_truth=True)
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing
        use_ground_truth: Whether to include ground-truth section (default: True)

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

    # Part 6: Ground-truth section (optional, for training mode)
    if use_ground_truth:
        if closest_idx is None or progress_score is None:
            raise ValueError("closest_idx and progress_score are required when use_ground_truth=True")
        ground_truth_section = build_ground_truth_section(closest_idx, progress_score)
        msgs.append({"type": "text", "value": ground_truth_section})

    # Part 7: Task instructions
    msgs.append({"type": "text", "value": TEXT_DEMO_INSTRUCTION_PART3})

    return msgs


def build_text_demo_prompt_from_item(
    item: Dict[str, Any],
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    use_ground_truth: bool = True
) -> List[Dict[str, Any]]:
    """
    Standalone function to build Text Demo prompt from a dataset item.

    Args:
        item: Dataset item with required fields:
            - task_goal: str
            - text_demo: List[str]
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
    return build_text_demo_prompt(
        task_goal=item['task_goal'],
        text_demo_list=item['text_demo'],
        total_steps=item['total_steps'],
        stage_to_estimate_path=item['stage_to_estimate'],
        closest_idx=item.get('closest_idx'),
        progress_score=item.get('progress_score'),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        use_ground_truth=use_ground_truth
    )
