"""
Utility functions for InternVL inference.
Adapted from Qwen2VL utilities.
"""

from __future__ import annotations

import os
import base64
import hashlib
import mimetypes
import os.path as osp
from io import BytesIO

from PIL import Image
import validators


def LMUDataRoot():
    """Get or create the LMUData root directory."""
    if 'LMUData' in os.environ and osp.exists(os.environ['LMUData']):
        return os.environ['LMUData']
    home = osp.expanduser('~')
    root = osp.join(home, 'LMUData')
    os.makedirs(root, exist_ok=True)
    return root


def get_rank_and_world_size():
    """Get current process rank and world size from environment."""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size


def get_gpu_memory():
    """Get free GPU memory for all available GPUs."""
    import subprocess
    try:
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values
    except Exception as e:
        print(f'{type(e)}: {str(e)}')
        return []


def auto_split_flag():
    """Check if auto split across GPUs should be enabled."""
    flag = os.environ.get('AUTO_SPLIT', '0')
    if flag == '1':
        return True
    _, world_size = get_rank_and_world_size()
    try:
        import torch
        device_count = torch.cuda.device_count()
        if device_count > world_size and device_count % world_size == 0:
            return True
        else:
            return False
    except:
        return False


def listinstr(lst, s):
    """Check if any item in list is a substring of s."""
    assert isinstance(lst, list)
    for item in lst:
        if item in s:
            return True
    return False


def md5(s):
    """Calculate MD5 hash of a string or file."""
    hash = hashlib.new('md5')
    if osp.exists(s):
        with open(s, 'rb') as f:
            for chunk in iter(lambda: f.read(2**20), b''):
                hash.update(chunk)
    else:
        hash.update(s.encode('utf-8'))
    return str(hash.hexdigest())


def decode_base64_to_image(base64_string, target_size=-1):
    """Decode base64 string to PIL Image."""
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    """Decode base64 string and save to file."""
    image = decode_base64_to_image(base64_string, target_size=target_size)
    image.save(image_path)


def download_file(url, filename=None):
    """Download a file from URL with progress bar."""
    import urllib.request
    from tqdm import tqdm

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if filename is None:
        filename = url.split('/')[-1]

    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
    except Exception as e:
        import logging
        logging.warning(f'{type(e)}: {e}')
        if 'huggingface.co' in url:
            url_new = url.replace('huggingface.co', 'hf-mirror.com')
            try:
                download_file(url_new, filename)
                return filename
            except Exception as e:
                logging.warning(f'{type(e)}: {e}')
                raise Exception(f'Failed to download {url}')
        else:
            raise Exception(f'Failed to download {url}')

    return filename


def parse_file(s):
    """Parse file path or URL and return (mime_type, path)."""
    if osp.exists(s) and s != '.':
        assert osp.isfile(s)
        suffix = osp.splitext(s)[1].lower()
        mime = mimetypes.types_map.get(suffix, 'unknown')
        return (mime, s)
    elif s.startswith('data:image/'):
        # Handle OPENAI base64 format
        content = s[11:]
        mime = content.split(';')[0]
        content = ';'.join(content.split(';')[1:])
        dname = osp.join(LMUDataRoot(), 'files')
        assert content.startswith('base64,')
        b64 = content[7:]
        os.makedirs(dname, exist_ok=True)
        tgt = osp.join(dname, md5(b64) + '.png')
        decode_base64_to_image_file(b64, tgt)
        return parse_file(tgt)
    elif validators.url(s):
        suffix = osp.splitext(s)[1].lower()
        if suffix in mimetypes.types_map:
            mime = mimetypes.types_map[suffix]
            dname = osp.join(LMUDataRoot(), 'files')
            os.makedirs(dname, exist_ok=True)
            tgt = osp.join(dname, md5(s) + suffix)
            download_file(s, tgt)
            return (mime, tgt)
        else:
            return ('url', s)
    else:
        return (None, s)
