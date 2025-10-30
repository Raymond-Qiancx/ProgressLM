from __future__ import annotations
from typing import Dict, Any, List, Union


# System prompt for training mode (CoT generation with ground-truth)
VISUAL_DEMO_SYSTEM_PROMPT_TRAIN = """You are an expert AI analyst specializing in generating step-by-step reasoning for visual task-progress evaluations. Your objective is not to estimate from scratch. Instead, your task is to construct a perfect, human-like chain of thought that logically explains and justifies a known, ground-truth progress score. Your entire response must read as if you are deducing the conclusion independently from visual analysis alone."""


# System prompt for normal inference mode
VISUAL_DEMO_SYSTEM_PROMPT_INFERENCE = """You are a progress estimator specializing in evaluating the progress of an ongoing task based on visual evidence. The demonstration consists of a sequence of video frames (images) showing how the task evolves from 0% (start) to 100% (completion). Your goal is to produce a human-like reasoning chain that logically supports the given progress score."""


# Default system prompt (use training mode)
VISUAL_DEMO_SYSTEM_PROMPT = VISUAL_DEMO_SYSTEM_PROMPT_TRAIN


VISUAL_DEMO_INSTRUCTION_PART1 = """Here is the demonstration:"""


VISUAL_DEMO_INSTRUCTION_PART2 = """Here is the current state that you need to estimate:"""


VISUAL_DEMO_INSTRUCTION_PART3 = """Your task:
1. Analyze the demonstration images to understand how the task visually progresses from start to completion.
2. Identify which frame in the provided visual demos is visually most similar to the current state image.
3. Compare the current state to that reference frame and determine whether it shows more or less progress.
4. Finally, provide a numeric progress estimation between 0% and 100%.

**Output Format**
Your response must strictly follow this format:
<ref_think>Your reasoning for choosing the closest demonstration frame as the reference</ref_think>
<ref>identify which image is most visually similar to the current state, and output only the number of that image</ref>
<score_think>Your reasoning for comparing the current state image with the reference frame(s)</score_think>
<score>Your final estimated progress score here</score>"""


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


def build_ground_truth_section(closest_idx: int, progress_score: Union[str, float]) -> str:
    """
    Build the ground-truth section for training mode (CoT generation).

    Args:
        closest_idx: 1-based index of the closest visual_demo image
        progress_score: Progress score (can be "33%" or 0.33)

    Returns:
        Formatted ground-truth section string

    Example:
        >>> build_ground_truth_section(1, "8%")
        '**Critical Rule** The correct final progress score will be provided to you...'
    """
    # Normalize progress_score to percentage string format
    if isinstance(progress_score, str):
        # Already string, keep as is if it has %, otherwise add it
        if not progress_score.endswith('%'):
            try:
                val = float(progress_score)
                if val <= 1.0:
                    progress_score = f"{int(val * 100)}%"
                else:
                    progress_score = f"{int(val)}%"
            except ValueError:
                pass  # Keep original
    elif isinstance(progress_score, (int, float)):
        # Convert numeric to percentage
        if progress_score <= 1.0:
            progress_score = f"{int(progress_score * 100)}%"
        else:
            progress_score = f"{int(progress_score)}%"

    ground_truth_text = f"""**Critical Rule** The correct final progress score will be provided to you. However, you must **never** reveal or imply that you already know the answer. Your reasoning must appear as a fully original, independent visual analysis derived from the images.

**Ground-Truth Progress Result**
Closest Reference Frame: The No. {closest_idx} demo image is the most relevant frame
Final Progress Score to Justify: {progress_score}"""

    return ground_truth_text


def build_visual_demo_prompt_qwen3vl(
    task_goal: str,
    visual_demo_paths: List[str],
    total_steps: int,
    stage_to_estimate_path: str,
    closest_idx: int = None,
    progress_score: Union[str, float] = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    use_ground_truth: bool = True
) -> List[Dict[str, Any]]:
    """
    Build Qwen3VL format messages for Visual Demo progress estimation task.

    Qwen3VL uses the standard transformers chat template format:
    [
        {"role": "user", "content": [
            {"type": "text", "text": "..."},
            {"type": "image", "image": "file://path/to/image.jpg", "min_pixels": ..., "max_pixels": ...},
            ...
        ]}
    ]

    Args:
        task_goal: Task goal description
        visual_demo_paths: List of paths to demonstration images
        total_steps: Total number of steps
        stage_to_estimate_path: Path to the current state image
        closest_idx: 1-based index of closest visual_demo (required if use_ground_truth=True)
        progress_score: Ground truth progress score (required if use_ground_truth=True)
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing
        use_ground_truth: Whether to include ground-truth section (default: True)

    Returns:
        List of message content items (for user role)
    """
    content = []

    # Part 1: Task goal
    content.append({"type": "text", "text": f"Task Goal: {task_goal}\n\n"})

    # Part 2: Demonstration introduction
    content.append({"type": "text", "text": VISUAL_DEMO_INSTRUCTION_PART1})

    # Part 3: Visual demo images (variable length)
    for demo_img_path in visual_demo_paths:
        # Ensure file:// prefix for local files
        if not demo_img_path.startswith(('http://', 'https://', 'file://', 'data:image')):
            demo_img_path = 'file://' + demo_img_path

        img_item = {"type": "image", "image": demo_img_path}
        if min_pixels is not None:
            img_item["min_pixels"] = min_pixels
        if max_pixels is not None:
            img_item["max_pixels"] = max_pixels
        content.append(img_item)

    # Part 4: Progress shift information
    progress_shifts = format_visual_demo_progress_shifts(total_steps)
    content.append({"type": "text", "text": f"\n\nThe progress shifts across all given visual demos is: {progress_shifts}\n\n"})

    # Part 5: Current state introduction
    content.append({"type": "text", "text": VISUAL_DEMO_INSTRUCTION_PART2})

    # Part 6: Current state image
    stage_img_path = stage_to_estimate_path
    if not stage_img_path.startswith(('http://', 'https://', 'file://', 'data:image')):
        stage_img_path = 'file://' + stage_img_path

    stage_img_item = {"type": "image", "image": stage_img_path}
    if min_pixels is not None:
        stage_img_item["min_pixels"] = min_pixels
    if max_pixels is not None:
        stage_img_item["max_pixels"] = max_pixels
    content.append(stage_img_item)

    # Part 7: Ground-truth section (optional, for training mode)
    if use_ground_truth:
        if closest_idx is None or progress_score is None:
            raise ValueError("closest_idx and progress_score are required when use_ground_truth=True")
        ground_truth_section = build_ground_truth_section(closest_idx, progress_score)
        content.append({"type": "text", "text": f"\n\n{ground_truth_section}\n\n"})

    # Part 8: Task instructions
    content.append({"type": "text", "text": VISUAL_DEMO_INSTRUCTION_PART3})

    return content


def build_visual_demo_prompt_from_item(
    item: Dict[str, Any],
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    use_ground_truth: bool = True
) -> List[Dict[str, Any]]:
    """
    Build Qwen3VL prompt from a dataset item.

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
        List of message content items (for user role)
    """
    return build_visual_demo_prompt_qwen3vl(
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
