import torch
import numpy as np
from typing import Optional, Union

def show_mask(mask: Union[torch.Tensor, np.ndarray], title: Optional[str] = None):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    while len(mask.shape) > 2 and mask.shape[0] == 1:
        mask = mask[0]
    assert len(mask.shape) == 2, "The mask must be a 2D array"
    plt.imshow(mask)
    if title is not None:
        plt.title(title)
    plt.show()