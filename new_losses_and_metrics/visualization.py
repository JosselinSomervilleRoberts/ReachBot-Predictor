import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List


def show_mask(mask: np.ndarray, title: Optional[str] = None):
    assert len(mask.shape) == 2, "The mask must be a 2D array"
    # assert mask.dtype == bool or (mask.dtype == np.uint8 or mask.dtype == np.int64) and np.max(mask) <= 1, "The mask must be a boolean array"
    plt.imshow(mask)
    if title is not None:
        plt.title(title)
    plt.show()


def show_image(image: np.ndarray):
    assert len(image.shape) == 3, "The image must be a 3D array"
    assert image.dtype == np.uint8, "The image must be a uint8 array"
    assert image.shape[2] == 3, "The image must have 3 channels positioned in the last dimension"
    plt.imshow(image)
    plt.show()


def show_masks(masks: List[np.ndarray], colors: Optional[List[List[int]]] = None):
    # Plots a list of masks in different colors
    if colors is None:
        colors = [[255,255,255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    assert len(masks) <= len(colors), "Not enough colors to plot the masks"

    # Create a black image
    img = np.zeros((masks[0].shape[0], masks[0].shape[1], 3), np.uint8)
    for i, mask in enumerate(masks):
        img[mask == 1] = colors[i]
    plt.imshow(img)
    plt.show()