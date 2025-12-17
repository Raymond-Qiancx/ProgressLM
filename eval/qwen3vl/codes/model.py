"""
Qwen3VL Model Wrapper for Progress Estimation Tasks

This module provides a unified wrapper for Qwen3VL models, supporting both
standard (8B) and MoE (30B-A3B) variants with batch inference capability.
"""

from __future__ import annotations

import os
import logging
import torch
from transformers import AutoProcessor


def ensure_image_url(image: str) -> str:
    """Convert image path to URL format for Qwen3VL processor."""
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


class Qwen3VLChat:
    """
    Qwen3VL model wrapper supporting both standard and MoE variants.

    Automatically detects model type based on path (moe/a3b keywords).
    Supports single and batch inference modes.
    """

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens: int = 2048,
        top_p: float = 0.001,
        top_k: int = 1,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        system_prompt: str | None = None,
        verbose: bool = False,
        use_custom_prompt: bool = False,  # For compatibility, not used
    ):
        """
        Initialize Qwen3VL model.

        Args:
            model_path: Path to Qwen3VL model weights
            min_pixels: Minimum pixels for image processing
            max_pixels: Maximum pixels for image processing
            max_new_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty
            system_prompt: System prompt for all generations
            verbose: Print debug information
            use_custom_prompt: Compatibility parameter (not used)
        """
        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose

        # Auto-detect MoE vs standard model
        model_path_lower = model_path.lower()
        if 'moe' in model_path_lower or 'a3b' in model_path_lower:
            from transformers import Qwen3VLMoeForConditionalGeneration
            MODEL_CLS = Qwen3VLMoeForConditionalGeneration
            if self.verbose:
                print(f"Detected MoE model: {model_path}")
        else:
            from transformers import Qwen3VLForConditionalGeneration
            MODEL_CLS = Qwen3VLForConditionalGeneration
            if self.verbose:
                print(f"Detected standard model: {model_path}")

        # Load model with flash attention
        self.model = MODEL_CLS.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.model.eval()

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = 'left'  # For decoder-only models

        torch.cuda.empty_cache()

        if self.verbose:
            print(f"Model loaded successfully on device: {self.model.device}")

    def _prepare_content(self, inputs: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        Convert input messages to Qwen3VL format.

        Args:
            inputs: List of dicts with 'type' and 'value' keys

        Returns:
            List of dicts in Qwen3VL message format
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                # Add pixel constraints if specified
                if 'min_pixels' in s:
                    item['min_pixels'] = s['min_pixels']
                elif self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if 'max_pixels' in s:
                    item['max_pixels'] = s['max_pixels']
                elif self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}")
            content.append(item)
        return content

    def generate(self, message, dataset=None):
        """
        Generate response for single or batch of messages.

        Args:
            message: Single message (list of dicts) or batch of messages (list of list of dicts)
            dataset: Dataset name (optional, for compatibility)

        Returns:
            Single response string or list of response strings
        """
        # Check if batch mode (list of messages) or single mode (single message)
        is_batch = isinstance(message[0], list) if message else False

        if is_batch:
            return self._generate_batch(message)
        else:
            return self._generate_single(message)

    def _generate_single(self, message: list[dict]) -> str:
        """Generate response for a single message."""
        # Build messages list
        content = self._prepare_content(message)
        messages = [{"role": "user", "content": content}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        if self.verbose:
            print(f'\033[31mMessages: {messages}\033[0m')

        # Qwen3VL style: apply_chat_template returns dict directly
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Generate
        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)

        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        response = output[0]

        if self.verbose:
            print(f'\033[32mResponse: {response}\033[0m')

        return response

    def _generate_batch(self, messages: list[list[dict]]) -> list[str]:
        """Generate responses for a batch of messages."""
        # Build all messages
        all_messages = []
        for msg in messages:
            content = self._prepare_content(msg)
            chat_msg = [{"role": "user", "content": content}]
            if self.system_prompt:
                chat_msg.insert(0, {"role": "system", "content": self.system_prompt})
            all_messages.append(chat_msg)

        if self.verbose:
            print(f'\033[31mBatch size: {len(all_messages)}\033[0m')

        # Process batch - Qwen3VL style
        # Apply chat template to each message
        texts = []
        for msg in all_messages:
            text = self.processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)

        # Process images for all messages
        try:
            from qwen_vl_utils import process_vision_info
            images, videos = process_vision_info(all_messages)
        except ImportError:
            logging.warning("qwen_vl_utils not found, trying alternative image processing")
            images = None
            videos = None

        # Tokenize with padding
        inputs = self.processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors='pt'
        )
        inputs = inputs.to(self.model.device)

        # Generate
        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)

        # Trim input tokens from each output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        responses = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        if self.verbose:
            print(f'\033[32mBatch responses: {responses}\033[0m')

        return responses
