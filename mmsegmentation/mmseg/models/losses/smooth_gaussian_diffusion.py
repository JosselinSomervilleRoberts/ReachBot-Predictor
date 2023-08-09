import torch

from .smooth_skeletonization import soft_dilate


def apply_smooth_gaussian_diffusion(
    mask: torch.Tensor, border_size: int = 25, factor: float = 0.9
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

    enlarged_mask = mask.clone()
    for d in range(border_size):
        new_enlarged_mask = soft_dilate(enlarged_mask)
        enlarged_mask = factor * enlarged_mask + (1 - factor) * new_enlarged_mask

    return enlarged_mask
