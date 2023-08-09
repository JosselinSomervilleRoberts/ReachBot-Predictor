import torch

from mmseg.evaluation.metrics.smooth_skeletonization import soft_dilate


def apply_smooth_gaussian_diffusion(mask: torch.Tensor, border_size: int = 25, sigma: float = 10.0) -> torch.Tensor:
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
        sigma: Sigma of the Gaussian function.
        
    Returns:
        Mask with smooth border.
    """

    enlarged_mask = mask.clone()
    expanded_mask = mask.clone()
    for d in range(border_size):
        new_enlarged_mask = soft_dilate(enlarged_mask)
        border: torch.Tensor = (new_enlarged_mask - enlarged_mask)
        expanded_mask = torch.where(border==1, torch.exp(-border * (d+1)**2 / (2*sigma**2)), expanded_mask)
        enlarged_mask = new_enlarged_mask

    return expanded_mask