import torch
import torch.nn.functional as F

# Source: https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py
def soft_dilate_large(img: torch.Tensor, size: int = 1) -> torch.Tensor:
    """Soft dilation function.

    This function is a soft approximation of the dilation operator. It is
    defined as a maxpooling between a pixel and its direct neighbors (cross pattern).
    It is a differentiable approximation of the dilation operator.

    Args:
        img: Input image.
        size: By how many pixels to dilate.
    
    Returns:
        Dilated image.
    """
    assert size >= 1, "Size should be at least 1"

    p1 = F.max_pool2d(img, (1+2*size,-1+2*size), (1,1), (size,size-1))
    p2 = F.max_pool2d(img, (-1+2*size,1+2*size), (1,1), (size-1,size))
    return torch.max(p1,p2)


def apply_smooth_gaussian_diffusion(
    mask: torch.Tensor, border_size: int = 25, factor: float = 0.9, max_iter: int = 20
) -> torch.Tensor:
    """Add a smooth border to a mask.

    The input mask should be binary.
    The output mask will be equal to 1 where the input mask is equal to 1.
    Then pixels at a distance d <= border_size from the input mask will have
    a value roughly equal to exp(-d**2 / (2*sigma**2)).
    The distance is computed with the soft_dilate function, which is
    approximate but differentiable.

    Args:
        mask: Input mask.
        border_size: Border size.
        factor: between 0 and 1, the higher it is the more the mask will be
            decreasing in terms of the distance to the input mask.

    Returns:
        Mask with smooth border.
    """
    assert 0 <= factor <= 1, "Factor should be between 0 and 1"
    assert border_size >= 0, "Border size should be positive"

    dilate_size = max(border_size // max_iter, 1)

    enlarged_mask = mask.clone()
    for d in range(min(border_size, max_iter)):
        new_enlarged_mask = soft_dilate_large(enlarged_mask, dilate_size)
        enlarged_mask = factor * enlarged_mask + (1 - factor) * new_enlarged_mask

    return enlarged_mask
