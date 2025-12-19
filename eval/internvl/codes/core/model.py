"""
InternVL Chat model wrapper for progress estimation inference.
"""

from __future__ import annotations

import os
import math
import logging
from typing import List, Dict, Any, Optional, Union

import torch

from .base import BaseModel
from .prompt import InternVLPromptMixin
from .image_utils import load_image, load_images_batch
from .util import get_rank_and_world_size, get_gpu_memory, auto_split_flag, listinstr


def split_model(model_path: str):
    """
    Create device map for splitting large models across GPUs.

    Args:
        model_path: Path to the model

    Returns:
        Device map dict for model loading
    """
    device_map = {}

    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = total_gpus // world_size

    # Determine number of layers based on model size
    # InternVL models have varying layer counts
    if '72b' in model_path.lower():
        num_layers = 80 + 8  # 80 layers + visual embedding overhead
    elif '40b' in model_path.lower():
        num_layers = 60 + 8
    elif '26b' in model_path.lower() or '25b' in model_path.lower():
        num_layers = 48 + 8
    elif '8b' in model_path.lower():
        num_layers = 32 + 8
    else:
        num_layers = 32 + 8  # Default for smaller models

    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2
    layer_cnt = 0

    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = rank + i * world_size
            layer_cnt += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map['vision_model'] = rank
    device_map['mlp1'] = rank
    device_map['language_model.model.tok_embeddings'] = rank
    device_map['language_model.model.embed_tokens'] = rank
    device_map['language_model.output'] = last_gpu
    device_map['language_model.model.norm'] = last_gpu
    device_map['language_model.lm_head'] = last_gpu

    return device_map


class InternVLChat(InternVLPromptMixin, BaseModel):
    """
    InternVL Chat model wrapper for progress estimation.

    Supports both single-GPU and multi-GPU model parallelism.
    """

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = False

    def __init__(
        self,
        model_path: str,
        max_num_tiles: int = 12,
        input_size: int = 448,
        max_new_tokens: int = 2048,
        top_p: float = 0.001,
        top_k: int = 1,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        use_custom_prompt: bool = True,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize InternVL Chat model.

        Args:
            model_path: Path to the InternVL model
            max_num_tiles: Maximum number of tiles for image preprocessing
            input_size: Size of each image tile (default 448)
            max_new_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty
            use_custom_prompt: Whether to use custom prompt building
            system_prompt: Optional system prompt
            verbose: Whether to print verbose output
        """
        super().__init__(use_custom_prompt=use_custom_prompt)

        self.model_path = model_path
        self.max_num_tiles = max_num_tiles
        self.input_size = input_size
        self.system_prompt = system_prompt
        self.verbose = verbose

        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=True if temperature > 0 else False,
        )

        # Load model and tokenizer
        self._load_model()

        torch.cuda.empty_cache()

    def _load_model(self):
        """Load the InternVL model and tokenizer."""
        from transformers import AutoModel, AutoTokenizer
        import transformers

        # Suppress "Setting `pad_token_id` to `eos_token_id`" warning
        transformers.logging.set_verbosity_error()

        rank, world_size = get_rank_and_world_size()

        # Check if CUDA is available and initialized
        cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
        if self.verbose:
            print(f"CUDA available: {cuda_available}, device_count: {torch.cuda.device_count() if cuda_available else 0}")

        # Determine loading strategy based on model size
        is_large_model = any(x in self.model_path.lower() for x in ['72b', '40b', '26b', '25b', '38b', '14b'])

        # Try loading with flash attention first, fallback if it fails
        def load_model_with_config(use_flash_attn, device_map):
            return AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                use_flash_attn=use_flash_attn,
                trust_remote_code=True,
                device_map=device_map,
            ).eval()

        device_map = 'auto' if (is_large_model or auto_split_flag()) else 'cuda'
        if self.verbose:
            print(f"Loading model with device_map='{device_map}'")

        # Try with flash attention first if CUDA is available
        if cuda_available:
            try:
                self.model = load_model_with_config(use_flash_attn=True, device_map=device_map)
                if self.verbose:
                    print("Model loaded with flash attention")
            except Exception as e:
                if self.verbose:
                    print(f"Flash attention failed ({e}), falling back to standard attention")
                self.model = load_model_with_config(use_flash_attn=False, device_map=device_map)
        else:
            # No CUDA, load without flash attention
            if self.verbose:
                print("CUDA not available, loading without flash attention")
            self.model = load_model_with_config(use_flash_attn=False, device_map='cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False
        )

        if self.verbose:
            print(f"Model loaded: {self.model_path}")

    def _prepare_images(
        self,
        image_paths: List[str]
    ) -> tuple:
        """
        Prepare images for inference.

        Args:
            image_paths: List of image file paths

        Returns:
            Tuple of (pixel_values tensor, num_patches_list)
        """
        # Get the device of vision_model for proper placement in multi-GPU setup
        try:
            vision_device = next(self.model.vision_model.parameters()).device
        except:
            vision_device = 'cuda'

        return load_images_batch(
            image_paths,
            input_size=self.input_size,
            max_num=self.max_num_tiles,
            dtype=torch.bfloat16,
            device=vision_device
        )

    def _build_prompt_from_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> tuple:
        """
        Build prompt and collect images from message list.

        NOTE: The text parts should already contain <image> placeholders.
        This method does NOT add image prefixes.

        Args:
            messages: List of message dicts with 'type' and 'value' keys

        Returns:
            Tuple of (image_paths, prompt_text)
        """
        image_paths = []
        text_parts = []

        for msg in messages:
            if msg['type'] == 'image':
                image_paths.append(msg['value'])
            elif msg['type'] == 'text':
                text_parts.append(msg['value'])

        # Combine text parts (should already contain <image> placeholders)
        prompt = "\n\n".join(text_parts)

        return image_paths, prompt

    def generate_inner(
        self,
        message: Union[List[Dict], List[List[Dict]]],
        dataset: Optional[str] = None
    ) -> Union[str, List[str]]:
        """
        Generate response for single or batch of messages.

        Args:
            message: Single message (list of dicts) or batch (list of list of dicts)
            dataset: Dataset name (optional)

        Returns:
            Single response string or list of response strings
        """
        # Check if batch mode
        is_batch = isinstance(message[0], list) if message else False

        if is_batch:
            # Batch mode: process each message sequentially
            # InternVL's chat doesn't natively support batching
            responses = []
            for msg in message:
                response = self._generate_single(msg)
                responses.append(response)
            return responses
        else:
            # Single mode
            return self._generate_single(message)

    def _generate_single(
        self,
        message: List[Dict[str, Any]]
    ) -> str:
        """
        Generate response for a single message.

        Args:
            message: List of message dicts

        Returns:
            Generated response string
        """
        # Extract images and build prompt
        image_paths, prompt = self._build_prompt_from_messages(message)

        if self.verbose:
            print(f"\n[InternVL] Prompt: {prompt[:200]}...")
            print(f"[InternVL] Images: {image_paths}")

        if len(image_paths) > 0:
            # Load and prepare images
            pixel_values, num_patches_list = self._prepare_images(image_paths)

            # Build generation config
            generation_config = dict(**self.generate_kwargs)

            # Add system prompt if provided
            if self.system_prompt:
                full_prompt = f"{self.system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            # Call model.chat()
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                full_prompt,
                generation_config,
                num_patches_list=num_patches_list,
            )
        else:
            # Text-only inference (shouldn't happen in our use case)
            generation_config = dict(**self.generate_kwargs)

            if self.system_prompt:
                full_prompt = f"{self.system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            # For text-only, create empty pixel values
            response = self.model.chat(
                self.tokenizer,
                None,
                full_prompt,
                generation_config,
            )

        if self.verbose:
            print(f"[InternVL] Response: {response}")

        return response

    def generate_from_item(
        self,
        image_paths: List[str],
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Convenience method for direct generation from images and prompt.

        NOTE: The prompt should already contain <image> placeholders at the
        appropriate positions. This method does NOT add image prefixes.

        Args:
            image_paths: List of image file paths
            prompt: The prompt text (with <image> placeholders embedded)
            system_prompt: Optional system prompt override

        Returns:
            Generated response string
        """
        # Add system prompt (prompt already contains <image> placeholders)
        sys_prompt = system_prompt if system_prompt else self.system_prompt
        full_prompt = f"{sys_prompt}\n\n{prompt}" if sys_prompt else prompt

        if len(image_paths) > 0:
            pixel_values, num_patches_list = self._prepare_images(image_paths)

            generation_config = dict(**self.generate_kwargs)

            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                full_prompt,
                generation_config,
                num_patches_list=num_patches_list,
            )
        else:
            generation_config = dict(**self.generate_kwargs)

            response = self.model.chat(
                self.tokenizer,
                None,
                full_prompt,
                generation_config,
            )

        return response

    def batch_generate(
        self,
        batch_items: List[tuple],  # List of (image_paths, prompt)
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Batch inference for multiple samples using model.batch_chat().

        NOTE: The prompts should already contain <image> placeholders at the
        appropriate positions. This method does NOT add image prefixes.

        Args:
            batch_items: List of (image_paths, prompt) tuples
            system_prompt: Optional system prompt override

        Returns:
            List of generated response strings
        """
        if len(batch_items) == 0:
            return []

        # 1. Prepare all images and prompts
        all_pixel_values = []
        all_num_patches = []
        questions = []

        sys_prompt = system_prompt if system_prompt else self.system_prompt

        for image_paths, prompt in batch_items:
            if len(image_paths) > 0:
                pixel_values, num_patches_list = self._prepare_images(image_paths)
                all_pixel_values.append(pixel_values)
                # Total patches for this sample
                all_num_patches.append(pixel_values.size(0))
            else:
                all_num_patches.append(0)

            # Prompt already contains <image> placeholders - do NOT add image prefix
            full_prompt = prompt
            if sys_prompt:
                full_prompt = f"{sys_prompt}\n\n{full_prompt}"
            questions.append(full_prompt)

        # 2. Combine all pixel values
        if all_pixel_values:
            combined_pixels = torch.cat(all_pixel_values, dim=0)
        else:
            combined_pixels = None

        # 3. Call batch_chat
        generation_config = dict(**self.generate_kwargs)

        try:
            responses = self.model.batch_chat(
                self.tokenizer,
                combined_pixels,
                num_patches_list=all_num_patches,
                questions=questions,
                generation_config=generation_config,
            )
            if self.verbose:
                print(f"[InternVL] Batch inference completed for {len(batch_items)} samples")
        except Exception as e:
            # Fallback to sequential processing
            if self.verbose:
                print(f"[InternVL] batch_chat failed: {e}, falling back to sequential")
            responses = []
            for image_paths, prompt in batch_items:
                resp = self.generate_from_item(image_paths, prompt, system_prompt)
                responses.append(resp)

        return responses