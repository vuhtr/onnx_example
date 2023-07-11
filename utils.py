import numpy as np
from PIL import Image

def resize_keep_ratio(img: Image, max_size: int, mode=Image.BILINEAR):
    """
    Resize image to keep ratio.
    """
    w, h = img.size
    if w <= max_size and h <= max_size:
        return img
    if w > h:
        ratio = max_size / w
    else:
        ratio = max_size / h
    return img.resize((int(w * ratio), int(h * ratio)), mode)


def padding(img: Image, size: int, pad_val: int):
    """
    Padding image to square.
    """
    new_img = Image.new('RGB', (size, size), (pad_val, pad_val, pad_val))
    new_img.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
    return new_img, (size - img.size[0]) // 2, (size - img.size[1]) // 2
    

def preprocess(img: Image, max_size: int, pad_val: int):
    """
    Preprocess image for RTMDet-Ins
    """
    resized_img = resize_keep_ratio(img, max_size)
    padded_img, pad_left, pad_top = padding(resized_img, max_size, pad_val)
    padded_img = np.array(padded_img)

    input_tensor = padded_img.astype(np.float32)
    input_tensor = input_tensor.transpose(2, 0, 1)
    
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
    input_tensor = (input_tensor - mean[:, None, None]) / std[:, None, None]

    return input_tensor, padded_img, pad_left, pad_top
