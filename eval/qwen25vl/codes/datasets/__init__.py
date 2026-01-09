"""Dataset loaders for qwen25vl evaluation."""

from .text_demo_dataset import load_text_demo_dataset, validate_image_path
from .visual_demo_dataset import load_visual_demo_dataset, validate_image_paths
from .adversarial_editing_dataset import load_adversarial_editing_dataset
from .image_edit_quality_dataset import load_image_edit_quality_dataset, validate_edited_image_path
from .text_object_replacement_dataset import load_text_object_replacement_dataset

# CoT generation datasets
from .text_nega_dataset import load_text_demo_dataset as load_text_nega_dataset
from .text_demo_dataset_cot_gen import load_text_demo_dataset as load_text_demo_dataset_cot_gen
from .visual_demo_dataset_cot_gen import load_visual_demo_dataset as load_visual_demo_dataset_cot_gen

__all__ = [
    'load_text_demo_dataset',
    'load_visual_demo_dataset',
    'load_adversarial_editing_dataset',
    'load_image_edit_quality_dataset',
    'load_text_object_replacement_dataset',
    'validate_image_path',
    'validate_image_paths',
    'validate_edited_image_path',
    # CoT generation
    'load_text_nega_dataset',
    'load_text_demo_dataset_cot_gen',
    'load_visual_demo_dataset_cot_gen',
]
