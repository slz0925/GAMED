"""Image reading and RGB channel conversion used by GAMED datasets."""
# Adapted from BasicSR (Apache 2.0):
# https://github.com/XPixelGroup/BasicSR/blob/00bbaf87163956fd00c22db73d051b9e27bcd563/codes/data/util.py
# Modified for PIL decoding, RGB expansion and GAMED resource paths.
# See third_party/licenses/BasicSR-Apache-2.0.txt.
from resource_paths import resolve_read_path as _read_path
import numpy as np
from PIL import Image


def read_img(path):
    """Load an 8-bit PIL image as float32 HWC."""
    with Image.open(_read_path(path)) as image:
        if image.mode in ('I', 'F') or image.mode.startswith('I;16'):
            raise ValueError('Unsupported image mode ' + image.mode + ': expected 8-bit colour or greyscale')
        if image.mode not in ('RGB', 'RGBA', 'L', 'LA'):
            image = image.convert('RGB')
        img = np.array(image, dtype=np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def channel_convert(in_c, tar_type, img_list):
    """Expand luminance to RGB, ignoring the alpha channel of LA images."""
    if tar_type != 'RGB':
        raise ValueError('Only RGB channel conversion is supported')
    if in_c in (1, 2):
        return [np.repeat(img[:, :, :1], 3, axis=2) for img in img_list]
    return img_list
