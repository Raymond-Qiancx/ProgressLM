"""
Prompt engineering Mixin for InternVL models.
Handles multi-image prompt building specific to InternVL.
"""

from __future__ import annotations

from typing import List


class InternVLPromptMixin:
    """
    Mixin class for InternVL to build prompts with multi-image support.

    InternVL uses a specific format for multi-image prompts:
        "Image-1: <image>
         Image-2: <image>
         ...
         [Your question/task here]"
    """

    def __init__(self, *args, use_custom_prompt: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._use_custom_prompt = use_custom_prompt

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset: str) -> bool:
        return True

    def build_internvl_multi_image_prompt(
        self,
        num_images: int,
        text: str
    ) -> str:
        """
        Build InternVL multi-image prompt format.

        Args:
            num_images: Number of images in the prompt
            text: The text content/question

        Returns:
            Formatted prompt string
        """
        # Build image prefix: "Image-1: <image>\nImage-2: <image>\n..."
        image_prefix = "".join([
            f"Image-{i + 1}: <image>\n"
            for i in range(num_images)
        ])
        return image_prefix + text

    def build_single_image_prompt(self, text: str) -> str:
        """
        Build prompt for single image input.

        Args:
            text: The text content/question

        Returns:
            Formatted prompt string
        """
        return f"<image>\n{text}"

    def format_progress_demo_prompt(
        self,
        demo_image_count: int,
        current_image_count: int,
        task_goal: str,
        progress_info: str,
        instructions: str
    ) -> str:
        """
        Format a complete progress estimation prompt.

        Args:
            demo_image_count: Number of demonstration images
            current_image_count: Number of current state images
            task_goal: Task goal description
            progress_info: Progress shift information
            instructions: Task instructions

        Returns:
            Formatted prompt string
        """
        total_images = demo_image_count + current_image_count

        prompt_parts = [
            f"The overall task goal is {task_goal}",
            "",
            "Here is the demonstration:",
        ]

        # Add image placeholders for demo images
        for i in range(demo_image_count):
            prompt_parts.append(f"Image-{i + 1}: <image>")

        # Add progress info
        prompt_parts.append("")
        prompt_parts.append(progress_info)
        prompt_parts.append("")
        prompt_parts.append("Here is the current state that you need to estimate:")

        # Add current state image placeholder(s)
        for i in range(current_image_count):
            prompt_parts.append(f"Image-{demo_image_count + i + 1}: <image>")

        prompt_parts.append("")
        prompt_parts.append(instructions)

        return "\n".join(prompt_parts)
