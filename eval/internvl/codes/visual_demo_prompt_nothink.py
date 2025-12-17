"""
Visual Demo prompt templates for InternVL progress estimation (nothink version).
Simplified version that only outputs the progress score.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


# System prompt for inference mode
VISUAL_DEMO_SYSTEM_PROMPT = """You are a progress estimator specializing in evaluating the progress of an ongoing task based on visual evidence. The demonstration consists of a sequence of video frames (images) showing how the task evolves from 0% (start) to 100% (completion). Your goal is to produce a human-like reasoning chain that logically supports the given progress score."""


VISUAL_DEMO_INSTRUCTION_PART1 = """Here is the demonstration:"""


VISUAL_DEMO_INSTRUCTION_PART2 = """Here is the current state that you need to estimate:"""


VISUAL_DEMO_INSTRUCTION_PART3 = """Your task:
1. Check the current state image carefully.
2. Analyze the overall task goal and visual demonstration to understand how the task progresses from start to completion.
3. Identify the reference states from the visual demonstration that are most related to the current state image.
4. Compare the current state image with the chosen reference state, determining whether the image is behind or after the reference state.
5. Estimate the progress numerically as a floating-point value between 0% and 100%, or directly output the "n/a" if you really cannot match the current state image to any of the states from demonstration.

Your answer only needs to output the final progress score you estimated, no other words needed."""


def format_visual_demo_progress_shifts(total_steps: int) -> str:
    """
    Format progress shifts for visual demo images based on total_steps.

    Args:
        total_steps: Total number of steps (not including the initial 0% state)

    Returns:
        Formatted string with image references and progress scores
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
    Build a prompt for Visual Demo progress estimation task (InternVL format, nothink).

    Args:
        task_goal: Task goal description
        visual_demo_paths: List of paths to demonstration images
        total_steps: Total number of steps
        stage_to_estimate_path: Path to the current state image

    Returns:
        Tuple of (all_image_paths, prompt_text)
    """
    # Collect all image paths
    all_image_paths = visual_demo_paths + [stage_to_estimate_path]

    # Build text prompt
    prompt_parts = []

    # Demonstration introduction
    prompt_parts.append(VISUAL_DEMO_INSTRUCTION_PART1)

    # Progress shift information
    progress_shifts = format_visual_demo_progress_shifts(total_steps)
    prompt_parts.append(f"The progress shifts across all given visual demos is: {progress_shifts}")
    prompt_parts.append("")

    # Current state introduction
    prompt_parts.append(VISUAL_DEMO_INSTRUCTION_PART2)
    prompt_parts.append(f"(This is Image-{len(visual_demo_paths) + 1})")
    prompt_parts.append("")

    # Task instructions
    prompt_parts.append(VISUAL_DEMO_INSTRUCTION_PART3)

    prompt_text = "\n".join(prompt_parts)

    return all_image_paths, prompt_text


def build_visual_demo_prompt_from_item(
    item: Dict[str, Any],
) -> Tuple[List[str], str]:
    """
    Build Visual Demo prompt from a dataset item (nothink version).

    Args:
        item: Dataset item

    Returns:
        Tuple of (all_image_paths, prompt_text)
    """
    return build_visual_demo_prompt(
        task_goal=item['task_goal'],
        visual_demo_paths=item['visual_demo'],
        total_steps=item['total_steps'],
        stage_to_estimate_path=item['stage_to_estimate'],
    )
