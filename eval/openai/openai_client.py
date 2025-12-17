#!/usr/bin/env python3
"""
OpenAI Vision API Client for Progress Evaluation

Provides a unified interface for calling GPT-5 with vision capabilities.
Handles image encoding, message format conversion, and API calls.
"""

import base64
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI


class OpenAIVisionClient:
    """OpenAI Vision API client for progress evaluation."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-mini",
        max_completion_tokens: int = 3000,
        temperature: float = 1.0,
        image_detail: str = "high"
    ):
        """
        Initialize the OpenAI Vision client.

        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-5, gpt-5-mini, gpt-5-nano)
            max_completion_tokens: Maximum tokens for completion
            temperature: Sampling temperature
            image_detail: Image detail level ("low", "high", "auto")
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.image_detail = image_detail

    def encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded string of the image

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be read
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to read image {image_path}: {str(e)}")

    def get_image_media_type(self, image_path: str) -> str:
        """
        Get the media type for an image based on its extension.

        Args:
            image_path: Path to the image file

        Returns:
            Media type string (e.g., "image/jpeg", "image/png")
        """
        ext = Path(image_path).suffix.lower()
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return media_types.get(ext, 'image/jpeg')

    def convert_messages(self, msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert qwen25vl message format to OpenAI API format.

        Input format (qwen25vl):
        [
            {"type": "text", "value": "..."},
            {"type": "image", "value": "/path/to/image.jpg"}
        ]

        Output format (OpenAI):
        [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]

        Args:
            msgs: List of message dicts in qwen25vl format

        Returns:
            List of message dicts in OpenAI API format
        """
        converted = []

        for msg in msgs:
            msg_type = msg.get("type", "text")

            if msg_type == "text":
                converted.append({
                    "type": "text",
                    "text": msg.get("value", "")
                })
            elif msg_type == "image":
                image_path = msg.get("value", "")
                base64_image = self.encode_image(image_path)
                media_type = self.get_image_media_type(image_path)

                converted.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_image}",
                        "detail": self.image_detail
                    }
                })

        return converted

    def generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response from the model.

        Args:
            messages: List of message dicts in qwen25vl format
            system_prompt: Optional system prompt (if not included in messages)

        Returns:
            Dictionary containing:
            - response: The model's response text
            - tokens_used: Total tokens used
            - status: "success" or "error"
            - error: Error message if status is "error"
        """
        try:
            # Convert messages to OpenAI format
            content = self.convert_messages(messages)

            # Build the API request
            api_messages = []

            # Add system message if provided separately
            if system_prompt:
                api_messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # Add user message with all content
            api_messages.append({
                "role": "user",
                "content": content
            })

            # Call the API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens
            )

            return {
                "response": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "status": "success",
                "error": None
            }

        except Exception as e:
            return {
                "response": None,
                "tokens_used": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "status": "error",
                "error": str(e)
            }

    def generate_with_retry(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate with automatic retry on failure.

        Args:
            messages: List of message dicts in qwen25vl format
            system_prompt: Optional system prompt
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            Same as generate()
        """
        last_error = None

        for attempt in range(max_retries):
            result = self.generate(messages, system_prompt)

            if result["status"] == "success":
                return result

            last_error = result["error"]

            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff

        return {
            "response": None,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "status": "error",
            "error": f"Failed after {max_retries} retries. Last error: {last_error}"
        }
