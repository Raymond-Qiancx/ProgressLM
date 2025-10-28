"""
Adversarial Image Editing Prompt Builder

This module builds prompts for adversarial image editing tasks using Qwen2-VL.
"""

from typing import Dict, Any, List


ADVERSARIAL_EDITING_SYSTEM_PROMPT = """You are an expert at adversarial image editing for instruction following tasks. Your goal is to analyze robotic manipulation tasks and strategically edit images to make them violate specific instructions while maintaining visual realism."""


def build_adversarial_editing_prompt_from_item(
    item: Dict[str, Any],
    min_pixels: int = 1280 * 28 * 28,
    max_pixels: int = 5120 * 28 * 28
) -> List[Dict[str, str]]:
    """
    Build prompt for adversarial image editing from a dataset item.

    The prompt structure follows the adversarial editing format:
    - Task Goal
    - Step-by-step Instructions
    - Current Image (from stage_to_estimate)
    - Corresponding Instruction (based on closest_idx, 1-indexed)
    - Editing Guidelines with 6 strategies
    - Output Format (strategy_think, strategy, prompt_think, prompt)

    Args:
        item: Dictionary containing:
            - task_goal: str
            - text_demo: List[str] (step-by-step instructions)
            - stage_to_estimate: str (image path, already with prefix)
            - closest_idx: int or str (1-based index)
            - total_steps: int or str
        min_pixels: Minimum image pixels for processing
        max_pixels: Maximum image pixels for processing

    Returns:
        List of message dictionaries compatible with Qwen2-VL format
    """
    task_goal = item.get('task_goal', 'Unknown task')
    text_demo = item.get('text_demo', [])
    stage_to_estimate = item.get('stage_to_estimate', '')
    closest_idx = int(item.get('closest_idx', 1))

    # Get the specific instruction (1-based indexing, so closest_idx-1 for 0-based list)
    if closest_idx > 0 and closest_idx <= len(text_demo):
        specific_instruction = text_demo[closest_idx - 1]
    else:
        specific_instruction = text_demo[0] if text_demo else "Unknown instruction"

    # Format text_demo as numbered list
    text_demo_formatted = "\n".join([f"{i+1}. {step}" for i, step in enumerate(text_demo)])

    # Build the prompt text
    prompt_text = f"""Task: Adversarial Image Editing for Instruction Following
Input Information:

Task Goal: {task_goal}
Step-by-step Instructions:
{text_demo_formatted}
Current Image: [The provided image]
Corresponding Instruction: Step {closest_idx} - "{specific_instruction}"

Your Task:
You are given an image that corresponds to a specific step in a multi-step robotic manipulation task. Your goal is to edit this image to make it no longer align with the corresponding instruction, causing the task to fail.

Editing Guidelines:

Modify key objects or elements in the image using one of the following strategies:
1. Color Change: Alter the color of critical objects (e.g., change a red apple to green)
2. Object Replacement: Replace the target object with a different object (e.g., replace an egg with an orange)
3. Occlusion/Removal: Hide or remove key objects from the scene



Requirements:

1. The edited image should clearly violate the corresponding instruction
2. IMPORTANT: Maintain visual realism and coherence - the edited image must look natural and believable
3. Ensure the edit would cause the overall task goal to fail
4. The modification should be semantically meaningful (not just noise or blur)

Output Format:
<strategy_think>
Analyze the current instruction and image content. Think step by step about which editing strategy would most effectively violate this instruction while maintaining realism. Consider the key objects involved and how modifying them would break the instruction.
</strategy_think>

<strategy> State the single strategy you selected from the editing guidelines (e.g., "Object Replacement" or "Color Change") </strategy>

<prompt_think>
Think step by step about how to formulate a clear and effective image editing prompt. Consider: What specific change to make? Which objects to target? What details are needed for realism?
</prompt_think>

<prompt> Write a concise image editing prompt (maximum 20 words) that clearly instructs the editing model what to change in the image. </prompt>"""

    # Build message format for Qwen2-VL
    # Format: [{'type': 'image', 'value': path}, {'type': 'text', 'value': prompt}]
    messages = [
        {'type': 'image', 'value': stage_to_estimate},
        {'type': 'text', 'value': prompt_text}
    ]

    return messages
