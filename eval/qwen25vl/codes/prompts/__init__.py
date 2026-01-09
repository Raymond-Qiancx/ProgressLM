"""Prompt builders for qwen25vl evaluation."""

from .text_demo_prompt import (
    build_text_demo_prompt,
    build_text_demo_prompt_from_item,
    TEXT_DEMO_SYSTEM_PROMPT,
    format_text_demo_with_progress,
)
from .text_demo_prompt_nothink import (
    build_text_demo_prompt as build_text_demo_nothink_prompt,
    build_text_demo_prompt_from_item as build_text_demo_nothink_prompt_from_item,
    TEXT_DEMO_SYSTEM_PROMPT as TEXT_DEMO_NOTHINK_SYSTEM_PROMPT,
)
from .visual_demo_prompt import (
    build_visual_demo_prompt,
    build_visual_demo_prompt_from_item,
    VISUAL_DEMO_SYSTEM_PROMPT,
    format_visual_demo_progress_shifts,
)
from .visual_demo_prompt_nothink import (
    build_visual_demo_prompt as build_visual_demo_nothink_prompt,
    build_visual_demo_prompt_from_item as build_visual_demo_nothink_prompt_from_item,
    VISUAL_DEMO_SYSTEM_PROMPT as VISUAL_DEMO_NOTHINK_SYSTEM_PROMPT,
)
from .adversarial_editing_prompt import (
    build_adversarial_editing_prompt_from_item,
    ADVERSARIAL_EDITING_SYSTEM_PROMPT,
)
from .image_edit_quality_prompt import (
    build_image_edit_quality_prompt_from_item,
    IMAGE_EDIT_QUALITY_SYSTEM_PROMPT,
)
from .text_object_replacement_prompt import (
    build_text_object_replacement_prompt_from_item,
    TEXT_OBJECT_REPLACEMENT_SYSTEM_PROMPT,
)

# CoT generation prompts
from .text_nega_prompt import (
    build_text_demo_prompt_from_item as build_text_nega_prompt_from_item,
    TEXT_DEMO_SYSTEM_PROMPT as TEXT_NEGA_SYSTEM_PROMPT,
)
from .visual_nega_prompt import (
    build_visual_demo_prompt_from_item as build_visual_nega_prompt_from_item,
    VISUAL_DEMO_SYSTEM_PROMPT as VISUAL_NEGA_SYSTEM_PROMPT,
)
from .text_demo_prompt_cot_gen import (
    build_text_demo_prompt_from_item as build_text_demo_prompt_cot_gen_from_item,
    TEXT_DEMO_SYSTEM_PROMPT as TEXT_DEMO_COT_GEN_SYSTEM_PROMPT,
)
from .visual_demo_prompt_cot_gen import (
    build_visual_demo_prompt_from_item as build_visual_demo_prompt_cot_gen_from_item,
    VISUAL_DEMO_SYSTEM_PROMPT as VISUAL_DEMO_COT_GEN_SYSTEM_PROMPT,
)

__all__ = [
    # Text demo
    'build_text_demo_prompt',
    'build_text_demo_prompt_from_item',
    'TEXT_DEMO_SYSTEM_PROMPT',
    'format_text_demo_with_progress',
    # Text demo nothink
    'build_text_demo_nothink_prompt',
    'build_text_demo_nothink_prompt_from_item',
    'TEXT_DEMO_NOTHINK_SYSTEM_PROMPT',
    # Visual demo
    'build_visual_demo_prompt',
    'build_visual_demo_prompt_from_item',
    'VISUAL_DEMO_SYSTEM_PROMPT',
    'format_visual_demo_progress_shifts',
    # Visual demo nothink
    'build_visual_demo_nothink_prompt',
    'build_visual_demo_nothink_prompt_from_item',
    'VISUAL_DEMO_NOTHINK_SYSTEM_PROMPT',
    # Adversarial editing
    'build_adversarial_editing_prompt_from_item',
    'ADVERSARIAL_EDITING_SYSTEM_PROMPT',
    # Image edit quality
    'build_image_edit_quality_prompt_from_item',
    'IMAGE_EDIT_QUALITY_SYSTEM_PROMPT',
    # Text object replacement
    'build_text_object_replacement_prompt_from_item',
    'TEXT_OBJECT_REPLACEMENT_SYSTEM_PROMPT',
    # CoT generation - nega
    'build_text_nega_prompt_from_item',
    'TEXT_NEGA_SYSTEM_PROMPT',
    'build_visual_nega_prompt_from_item',
    'VISUAL_NEGA_SYSTEM_PROMPT',
    # CoT generation - cot_gen
    'build_text_demo_prompt_cot_gen_from_item',
    'TEXT_DEMO_COT_GEN_SYSTEM_PROMPT',
    'build_visual_demo_prompt_cot_gen_from_item',
    'VISUAL_DEMO_COT_GEN_SYSTEM_PROMPT',
]
