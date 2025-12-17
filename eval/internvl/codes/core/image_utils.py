"""
Image preprocessing utilities for InternVL models.
Adapted from InternVL reference implementation.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    """
    Build image transformation pipeline for InternVL.

    Args:
        input_size: Target size for images (e.g., 448)

    Returns:
        Composed transformation pipeline
    """
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return transform


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: set,
    width: int,
    height: int,
    image_size: int
) -> Tuple[int, int]:
    """
    Find the closest aspect ratio from target ratios.

    Args:
        aspect_ratio: Original image aspect ratio
        target_ratios: Set of target (width, height) ratios
        width: Original image width
        height: Original image height
        image_size: Base tile size

    Returns:
        Best matching (width_tiles, height_tiles) tuple
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False
) -> List[Image.Image]:
    """
    Dynamically preprocess image into tiles based on aspect ratio.

    Args:
        image: PIL Image to process
        min_num: Minimum number of tiles
        max_num: Maximum number of tiles
        image_size: Size of each tile
        use_thumbnail: Whether to add a thumbnail of the full image

    Returns:
        List of PIL Images (tiles and optionally thumbnail)
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate all valid target ratios
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find best matching ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and split into tiles
    resized_img = image.resize((target_width, target_height))
    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    # Optionally add thumbnail
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(
    image_file: str,
    input_size: int = 448,
    max_num: int = 12
) -> torch.Tensor:
    """
    Load and preprocess a single image for InternVL.

    Args:
        image_file: Path to image file
        input_size: Size of each tile
        max_num: Maximum number of tiles

    Returns:
        Tensor of shape (num_patches, 3, input_size, input_size)
    """
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(im) for im in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_images_batch(
    image_files: List[str],
    input_size: int = 448,
    max_num: int = 12,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda"
) -> Tuple[torch.Tensor, List[int]]:
    """
    Load and preprocess multiple images for InternVL.

    Args:
        image_files: List of paths to image files
        input_size: Size of each tile
        max_num: Maximum number of tiles per image
        dtype: Tensor data type
        device: Device to move tensors to

    Returns:
        Tuple of:
            - Concatenated tensor of all patches
            - List of patch counts for each image
    """
    pixel_values_list = []
    num_patches_list = []

    for path in image_files:
        pixels = load_image(path, input_size=input_size, max_num=max_num)
        num_patches_list.append(pixels.size(0))
        pixel_values_list.append(pixels)

    # Concatenate all images
    pixel_values = torch.cat(pixel_values_list, dim=0).to(dtype).to(device)

    return pixel_values, num_patches_list
