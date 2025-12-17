"""
Base model class for InternVL.
Adapted from Qwen2VL base implementation.
"""

from __future__ import annotations

from abc import abstractmethod
from .util import parse_file


class BaseModel:
    """Base class for VLM models."""

    INTERLEAVE = False
    allowed_types = ['text', 'image', 'video']

    def __init__(self):
        self.dump_image_func = None

    def use_custom_prompt(self, dataset):
        """Whether to use custom prompt for the given dataset."""
        return False

    @abstractmethod
    def build_prompt(self, line, dataset):
        """Build custom prompts for a specific dataset."""
        raise NotImplementedError

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    @abstractmethod
    def generate_inner(self, message, dataset=None):
        """Generate response for input message(s)."""
        raise NotImplementedError

    def check_content(self, msgs):
        """Check the content type of the input."""
        if isinstance(msgs, str):
            return 'str'
        if isinstance(msgs, dict):
            return 'dict'
        if isinstance(msgs, list):
            types = [self.check_content(m) for m in msgs]
            if all(t == 'str' for t in types):
                return 'liststr'
            if all(t == 'dict' for t in types):
                return 'listdict'
        return 'unknown'

    def preproc_content(self, inputs):
        """Convert raw input messages to a list of dicts."""
        if self.check_content(inputs) == 'str':
            return [dict(type='text', value=inputs)]
        elif self.check_content(inputs) == 'dict':
            assert 'type' in inputs and 'value' in inputs
            return [inputs]
        elif self.check_content(inputs) == 'liststr':
            res = []
            for s in inputs:
                mime, pth = parse_file(s)
                if mime is None or mime == 'unknown':
                    res.append(dict(type='text', value=s))
                else:
                    res.append(dict(type=mime.split('/')[0], value=pth))
            return res
        elif self.check_content(inputs) == 'listdict':
            for item in inputs:
                assert 'type' in item and 'value' in item
                mime, s = parse_file(item['value'])
                if mime is None:
                    assert item['type'] == 'text'
                else:
                    assert mime.split('/')[0] == item['type']
                    item['value'] = s
            return inputs
        else:
            return None

    def generate(self, message, dataset=None):
        """
        Generate the output message.

        Args:
            message (list[dict] or list[list[dict]]): Input message (single or batch).
            dataset (str, optional): Name of the dataset.

        Returns:
            str or list[str]: The generated message(s).
        """
        # Check if batch mode (list of messages)
        is_batch = isinstance(message, list) and len(message) > 0 and isinstance(message[0], list)

        if is_batch:
            # Batch mode: skip preprocessing, let generate_inner handle it
            for msg in message:
                assert self.check_content(msg) == 'listdict', f'Invalid batch message type: {msg}'
                for item in msg:
                    assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'
            return self.generate_inner(message, dataset)
        else:
            # Single mode: apply preprocessing
            assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {message}'
            message = self.preproc_content(message)
            assert message is not None and self.check_content(message) == 'listdict'
            for item in message:
                assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'
            return self.generate_inner(message, dataset)

    def chat(self, messages, dataset=None):
        """Multi-turn chatting interface."""
        assert hasattr(self, 'chat_inner'), 'The API model should have the `chat_inner` method.'
        for msg in messages:
            assert isinstance(msg, dict) and 'role' in msg and 'content' in msg, msg
            assert self.check_content(msg['content']) in ['str', 'dict', 'liststr', 'listdict'], msg
            msg['content'] = self.preproc_content(msg['content'])

        while len(messages):
            try:
                return self.chat_inner(messages, dataset=dataset)
            except Exception as e:
                print(f'{type(e)}: {e}')
                messages = messages[1:]
                while len(messages) and messages[0]['role'] != 'user':
                    messages = messages[1:]
                continue
        return 'Chat Mode: Failed with all possible conversation turns.'
