"""
Image Edit Quality Evaluation Prompt Builder

This module builds prompts for evaluating the quality of adversarially edited images.
The goal is to determine whether an edited image successfully violates the corresponding
instruction and represents a failure state (0% progress).
"""

from typing import Dict, Any, List


IMAGE_EDIT_QUALITY_SYSTEM_PROMPT = """You are an expert at evaluating image editing quality for robotic manipulation tasks. Your task is to determine whether an edited image successfully violates the corresponding instruction and represents a failure state, while maintaining visual realism and safety standards."""


def build_image_edit_quality_prompt_from_item(
    item: Dict[str, Any],
    min_pixels: int = 1280 * 28 * 28,
    max_pixels: int = 5120 * 28 * 28
) -> List[Dict[str, str]]:
    """
    Build prompt for image edit quality evaluation from a dataset item.

    The prompt structure:
    - Task Goal
    - Step-by-step Instructions
    - Current Step Instruction
    - Editing Prompt Applied
    - Edited Image
    - Evaluation Criteria (5 points including safety check)
    - Binary output: yes/no

    Args:
        item: Dictionary containing:
            - task_goal: str
            - text_demo: List[str] (step-by-step instructions)
            - raw_demo: str (the specific instruction being evaluated)
            - prompt: str (the editing prompt that was applied)
            - edited_image_path: str (full path to edited image)
        min_pixels: Minimum image pixels for processing
        max_pixels: Maximum image pixels for processing

    Returns:
        List of message dictionaries compatible with Qwen2-VL format
    """
    task_goal = item.get('task_goal', 'Unknown task')
    text_demo = item.get('text_demo', [])
    raw_demo = item.get('raw_demo', 'Unknown instruction')
    editing_prompt = item.get('prompt', 'Unknown editing prompt')
    edited_image_path = item.get('edited_image_path', '')

    # Format text_demo as numbered list
    if text_demo:
        text_demo_formatted = "\n".join([f"{i+1}. {step}" for i, step in enumerate(text_demo)])
    else:
        text_demo_formatted = "No step-by-step instructions available"

    # Build the prompt text
    prompt_text = f"""Task: Image Edit Quality Evaluation

Original Task Goal: {task_goal}

Step-by-step Instructions:
{text_demo_formatted}

Current Step Instruction: "{raw_demo}"

Editing Prompt Applied: "{editing_prompt}"

[Edited Image Shown Above]

Your Task:
Evaluate whether the edited image successfully violates the instruction "{raw_demo}" and would cause the task to fail (representing 0% progress for this step).

Evaluation Criteria:
1. Does the edited image clearly violate the corresponding instruction?
2. Would this violation cause the overall task to fail?
3. Is the edit visually realistic and coherent?
4. Is the modification semantically meaningful (not just noise or blur)?
5. Reject edits that show malicious or destructive manipulation such as:
   - Adding robotic arms, weapons, or unrelated mechanical parts
   - Damaging, melting, or breaking the main subject
   - Introducing absurd or aggressive elements that undermine the scene's realism or intent

Answer "yes" if the edit successfully creates a failure case (all criteria met, including safety).
Answer "no" if the edit is poor, unrealistic, unsafe, or doesn't effectively violate the instruction.

Your answer: (yes/no)"""

    # Build message format for Qwen2-VL
    # Format: [{'type': 'image', 'value': path}, {'type': 'text', 'value': prompt}]
    messages = [
        {'type': 'image', 'value': edited_image_path},
        {'type': 'text', 'value': prompt_text}
    ]

    return messages
