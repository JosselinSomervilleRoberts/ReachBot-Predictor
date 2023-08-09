import numpy as np
from PIL import Image
from typing import Tuple, Union
from skimage.measure import label


def separate_mask(mask: np.ndarray):
    assert len(mask.shape) == 2, "The mask must be a 2D array"
    assert mask.dtype == bool or (mask.dtype == np.uint8 or mask.dtype == np.int64) and np.max(mask) <= 1, "The mask must be a boolean array"
    
    separated_mask = label(mask)
    blobs = []
    for i in np.unique(separated_mask):
        if i == 0:  # background
            continue
        blobs.append((separated_mask == i).astype(int))
    return blobs


def get_bounding_box(mask) -> Tuple[int, int, int, int]:
    """
    Returns the bounding box of a binary mask.
    The bounding box is defined as (x_min, y_min, x_max, y_max).
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    x_min, x_max = np.where(cols)[0][[0, -1]]
    y_min, y_max = np.where(rows)[0][[0, -1]]
    return x_min, y_min, x_max, y_max


def crop_to_bounding_box(
    image: Union[np.ndarray, Image.Image],
    bounding_box: Tuple[int, int, int, int]) -> Union[np.ndarray, Image.Image]:
    """
    Crops an image to a bounding box.
    The bounding box is defined as (x_min, y_min, x_max, y_max).
    """
    if isinstance(image, Image.Image):
        return image.crop(bounding_box)
    else:
        return image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]


def get_shared_bounding_box(mask1, mask2) -> Tuple[int, int, int, int]:
    """Returns the smallest bounding box such that boths masks are contained in it."""
    x_min1, y_min1, x_max1, y_max1 = get_bounding_box(mask1)
    x_min2, y_min2, x_max2, y_max2 = get_bounding_box(mask2)
    x_min = min(x_min1, x_min2)
    y_min = min(y_min1, y_min2)
    x_max = max(x_max1, x_max2)
    y_max = max(y_max1, y_max2)
    return x_min, y_min, x_max, y_max


def crop_as_small_as_possible(mask1, mask2):
    """Crop both masks to their shared bounding box."""
    x_min, y_min, x_max, y_max = get_shared_bounding_box(mask1, mask2)
    mask1 = crop_to_bounding_box(mask1, (x_min, y_min, x_max, y_max))
    mask2 = crop_to_bounding_box(mask2, (x_min, y_min, x_max, y_max))
    return mask1, mask2