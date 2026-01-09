"""
Text Object Replacement Prompt Builder

This module builds prompts for text-based object replacement tasks using Qwen2-VL.
"""

from typing import Dict, Any, List


# TEXT_OBJECT_REPLACEMENT_SYSTEM_PROMPT = """You are an expert at creating negative cases for robotic manipulation tasks by strategically replacing objects in task instructions. Your goal is to analyze task goals and step-by-step instructions, identify the main object being manipulated, and replace it with a confusing or similar object to create a failure case while maintaining logical coherence."""
TEXT_OBJECT_REPLACEMENT_SYSTEM_PROMPT = """ """


def build_text_object_replacement_prompt_from_item(
    item: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Build prompt for text object replacement from a dataset item.

    The prompt structure follows the text object replacement format:
    - Task Goal
    - Step-by-step Instructions
    - Current Image (if available)
    - Object replacement strategy with priority
    - Output Format (think, edited_goal, edited_demo)

    Args:
        item: Dictionary containing:
            - task_goal: str
            - text_demo: List[str] (step-by-step instructions)
            - stage_to_estimate: str (optional, image path if available)
            - closest_idx: int or str (1-based index)
            - total_steps: int or str

    Returns:
        List of message dictionaries compatible with Qwen2-VL format
    """
    task_goal = item.get('task_goal', 'Unknown task')
    text_demo = item.get('text_demo', [])
    stage_to_estimate = item.get('stage_to_estimate', '')

    # Format text_demo as JSON array string for clarity
    text_demo_formatted = '["' + '", "'.join(text_demo) + '"]'

    # Build the prompt text
    prompt_text = f"""Task: Modify the Task Goal and Step-by-step Instructions to make the Current Image does not match the Task Goal or any Step-by-step Instructions.

    Input Information:
    - Task Goal: {task_goal}
    - Step-by-step Instructions: {text_demo_formatted}
    - Current Image: [The provided image]

    Editing Guidelines:
    1. Keep the original sentence format and structure - ONLY replace the object name
    2. For each step in Step-by-step Instructions, preserve ALL markers like [right], [left], [towards], etc. in their EXACT original positions

    Output Format:
    <edited_goal> "put your edited task goal here" </edited_goal>

    <edited_demo>
    ["your edited step 1", "your edited step 2", "your edited step 3", ..., "your edited step n"]
    </edited_demo>"""

#     prompt_text = f"""Task: Create a Negative Case by Object Replacement

# Input Information:
# - Task Goal: {task_goal}
# - Step-by-step Instructions: {text_demo_formatted}
# - Current Image: [The provided image]

# Your Task:
# 1. Analyze the main object being manipulated in the task (e.g., "corn cob", "green plate", "beige plate")
# 2. Replace this object following the priority strategy:
#    - **First Priority**: If there are other objects visible in the image, choose the most confusing/similar one as replacement (e.g., if both corn and carrot exist, replace corn with carrot; if both green plate and beige plate exist, replace green plate with beige plate)
#    - **Second Priority**: If no suitable objects exist in the image, choose a physically reasonable and similar but clearly different object (e.g., apple → orange is reasonable, but apple → car/water bottle is not reasonable)
# 3. Edit both the task goal and all step-by-step instructions to reflect this new object
# 4. Ensure the replacement creates a failure case

# Editing Guidelines:
# 1. Keep the original sentence format and structure - ONLY replace the object name
# 2. For each step in "text_demo", preserve ALL markers like [right], [left], [towards], etc. in their EXACT original positions
# 3. Only change the object name (e.g., "corn" → "carrot" or "green plate" → "beige plate")
# 4. Do not modify action verbs, spatial descriptions, or hand markers
# 5. The replacement object should be contextually confusing or physically similar to the original
# 6. If the object has color descriptors (e.g., "green plate"), maintain or adjust the descriptor appropriately for the new object

# Output Format:
# <think>
# - Original object identified: [object name]
# - Objects visible in current image: [list other objects if any]
# - Replacement strategy used: [Priority 1 or Priority 2]
# - Replacement object chosen: [new object name]
# - Reasoning: [why this creates a valid negative case]
# </think>

# <edited_goal> "put your edited task goal here" </edited_goal>

# <edited_demo>
# ["your edited step 1", "your edited step 2", "your edited step 3", ..., "your edited step n"]
# </edited_demo>"""

    # Build message format for Qwen2-VL
    # If image is available, include it; otherwise, text-only
    messages = []

    # Add image if stage_to_estimate is provided
    if stage_to_estimate:
        messages.append({'type': 'image', 'value': stage_to_estimate})

    messages.append({'type': 'text', 'value': prompt_text})

    return messages
