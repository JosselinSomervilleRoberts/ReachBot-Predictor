import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS

from .smooth_gaussian_diffusion import apply_smooth_gaussian_diffusion
from .smooth_skeletonization import soft_skeletonize, soft_skeletonize_thin
from .utils import get_class_weight, weighted_loss


def soft_dice(
    ground_truth: torch.Tensor, prediction: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """Soft dice.

    Args:
        ground_truth: Ground truth mask of shape (H, W) or (N, H, W).
        prediction: Predicted mask of shape (H, W) or (N, H, W).
        epsilon: Epsilon used for numerical stability.

    Returns:
        Soft dice.
    """
    numerator = 2.0 * torch.sum(ground_truth * prediction, dim=(-2, -1))
    denominator = torch.sum(ground_truth**2, dim=(-2, -1)) + torch.sum(
        prediction**2, dim=(-2, -1)
    )
    return (numerator + epsilon) / (denominator + epsilon)


def smooth_skeleton_dice_loss(
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    iterations: int = 10,
    border_size: int = 25,
    sigma: float = 10.0,
    thinner: bool = False,
) -> torch.Tensor:
    """Smooth skeleton loss.

    Uses the smooth skeletonization function to compute the skeleton of the
    ground truth and predicted masks. Then computes the DICE between the two
    smoothly enlarged skeletons (with add_smooth_border_to_mask)

    Args:
        ground_truth: Ground truth mask.
        prediction: Predicted mask.
        iterations: Number of iterations for the soft skeletonization.
        border_size: Border size.
        sigma: Sigma of the Gaussian function.

    Returns:
        Smooth skeleton loss.
    """
    if thinner:
        ground_truth_skeleton = soft_skeletonize_thin(ground_truth, iterations)
        prediction_skeleton = soft_skeletonize_thin(prediction, iterations)
    else:
        ground_truth_skeleton = soft_skeletonize(ground_truth, iterations)
        prediction_skeleton = soft_skeletonize(prediction, iterations)
    ground_truth_border = apply_smooth_gaussian_diffusion(
        ground_truth_skeleton, border_size, sigma
    )
    prediction_border = apply_smooth_gaussian_diffusion(
        prediction_skeleton, border_size, sigma
    )
    return 1 - soft_dice(ground_truth_border, prediction_border)


def smooth_skeleton_intersection_loss(
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    iterations: int = 10,
    border_size: int = 25,
    sigma: float = 10.0,
    thinner: bool = False,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Smooth skeleton loss.

    Uses the smooth skeletonization function to compute the skeleton of the
    ground truth and predicted masks. Then computes the product between the
    predicted skeleton and the ground truth skeleton enlarged and vice versa.

    Args:
        ground_truth: Ground truth mask.
        prediction: Predicted mask.
        iterations: Number of iterations for the soft skeletonization.
        border_size: Border size.
        sigma: Sigma of the Gaussian function.

    Returns:
        Smooth skeleton loss.
    """
    if thinner:
        ground_truth_skeleton = soft_skeletonize_thin(ground_truth, iterations)
        prediction_skeleton = soft_skeletonize_thin(prediction, iterations)
    else:
        ground_truth_skeleton = soft_skeletonize(ground_truth, iterations)
        prediction_skeleton = soft_skeletonize(prediction, iterations)
    ground_truth_border = apply_smooth_gaussian_diffusion(
        ground_truth_skeleton, border_size, sigma
    )
    prediction_border = apply_smooth_gaussian_diffusion(
        prediction_skeleton, border_size, sigma
    )

    p1_num = torch.sum(ground_truth_border * prediction_skeleton, dim=(-2, -1))
    p1_den = torch.sum(prediction_skeleton, dim=(-2, -1))
    p1 = (p1_num + epsilon) / (p1_den + epsilon)

    p2_num = torch.sum(prediction_border * ground_truth_skeleton, dim=(-2, -1))
    p2_den = torch.sum(ground_truth_skeleton, dim=(-2, -1))
    p2 = (p2_num + epsilon) / (p2_den + epsilon)

    return 1 - torch.sqrt(p1 * p2)


@weighted_loss
def skil_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    iterations: int = 10,
    border_size: int = 25,
    sigma: float = 10.0,
    thinner: bool = False,
):
    print(pred.shape, target.shape)
    pred = pred.float()[:, 1, :, :]
    # pred = pred.float()
    target = target.float()
    assert pred.shape[0] == target.shape[0]
    if thinner:
        ground_truth_skeleton = soft_skeletonize_thin(target, iterations)
        prediction_skeleton = soft_skeletonize_thin(pred, iterations)
    else:
        ground_truth_skeleton = soft_skeletonize(target, iterations)
        prediction_skeleton = soft_skeletonize(pred, iterations)
    ground_truth_border = apply_smooth_gaussian_diffusion(
        ground_truth_skeleton, border_size, sigma
    )
    prediction_border = apply_smooth_gaussian_diffusion(
        prediction_skeleton, border_size, sigma
    )
    return 1 - soft_dice(ground_truth_border, prediction_border)


@MODELS.register_module()
class SkilLoss(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        reduction="mean",
        class_weight=None,
        loss_weight=1.0,
        ignore_index=255,
        loss_name="loss_skil",
        **kwargs,
    ):
        super().__init__()

        # add asserts for args

        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(
        self,
        pred,
        target,
        reduction_override=None,
        **kwargs,
    ):
        reduction = reduction_override if reduction_override else self.reduction
        pred = F.softmax(pred, dim=1)

        loss = self.loss_weight * skil_loss(pred, target)

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
