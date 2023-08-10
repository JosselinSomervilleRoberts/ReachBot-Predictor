import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
import matplotlib.pyplot as plt

from .smooth_gaussian_diffusion import apply_smooth_gaussian_diffusion
from .smooth_skeletonization import soft_skeletonize, soft_skeletonize_thin
from .utils import get_class_weight, weighted_loss
from .visualization_utils import Plotter
from toolbox.printing import debug as debug_fn
from typing import Optional


def cl_dice(
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    skel_pred: torch.Tensor,
    skel_true: torch.Tensor,
    epsilon: float = 1,
) -> torch.Tensor:
    """Soft dice.

    Args:
        ground_truth: Ground truth mask of shape (H, W) or (N, H, W).
        prediction: Predicted mask of shape (H, W) or (N, H, W).
        epsilon: Epsilon used for numerical stability.

    Returns:
        Soft dice.
    """
    assert len(ground_truth.shape) == 3, "Ground truth must be of shape (N, H, W)"
    tprec = (torch.sum(skel_pred * ground_truth, dim=(-2, -1)) + epsilon) / (
        torch.sum(skel_pred, dim=(-2, -1)) + epsilon
    )
    tsens = (torch.sum(skel_true * prediction, dim=(-2, -1)) + epsilon) / (
        torch.sum(skel_true, dim=(-2, -1)) + epsilon
    )
    cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens)
    return cl_dice


@weighted_loss
def cl_dice_loss(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    iterations: int = 10,
    epsilon: float = 1,
    thinner: bool = False,
    debug: bool = False,
    debug_path: str = None,
):
    """Smooth skeleton loss.

    Uses the smooth skeletonization function to compute the skeleton of the
    ground truth and predicted masks.

    If use_dice is true, then computes the dice loss between the two enlarged
    skeletons.

    Id not, then computes the product between the predicted skeleton and the
    ground truth skeleton enlarged and vice versa.

    Args:
        ground_truth: Ground truth mask.
        prediction: Predicted mask.
        use_dice: If True, will use the dice loss, otherwise will use the
            product loss.
        smooth_threshold_factor: Factor to use for the soft threshold.
            If negative, no threshold will be applied.
            If positive, it should be greater than 1.
        iterations: Number of iterations for the soft skeletonization.
        border_size: Border size.
        sigma: Sigma of the Gaussian function.
        epsilon: Epsilon used for numerical stability.
        thinner: If True, will use the thin skeletonization.
        debug: If True, will make some plots to understand what is going on.
        debug_path: Path where to save the debug plots.

    Returns:
        Smooth skeleton loss.
    """
    if debug:
        assert debug_path is not None, "debug_path must be provided if debug is True"

    # The random resize puts the value 255 to pad, which we remove and replace
    # by zero, the value of the background.
    ground_truth = torch.where(
        ground_truth > 1, torch.zeros_like(ground_truth), ground_truth
    )
    assert torch.max(ground_truth) <= 1, "The ground truth must be binary!"
    assert torch.min(ground_truth) >= 0, "The ground truth must be binary!"

    # Format the prediciton and ground truth
    # We only keep the class 1 as we do not want to evaluate the background.
    prediction = prediction[:, 1, :, :].float()
    ground_truth = ground_truth.float()
    assert prediction.shape[0] == ground_truth.shape[0]

    # Smooth skeletonization
    if thinner:
        ground_truth_skeleton = soft_skeletonize_thin(ground_truth.clone(), iterations)
        prediction_skeleton = soft_skeletonize_thin(prediction, iterations)
    else:
        ground_truth_skeleton = soft_skeletonize(ground_truth.clone(), iterations)
        prediction_skeleton = soft_skeletonize(prediction, iterations)

    # Compute the loss
    loss: torch.Tensor = 1 - cl_dice(
        ground_truth,
        prediction,
        prediction_skeleton,
        ground_truth_skeleton,
        epsilon=epsilon,
    )

    # Debug if needed
    if debug:
        n_rows = 2
        n_cols = 2

        for batch_idx in range(prediction.shape[0]):
            Plotter.start(n_rows, n_cols, debug_path)
            Plotter.plot_mask(ground_truth[batch_idx], "Ground truth")
            Plotter.plot_mask(ground_truth_skeleton[batch_idx], "Ground truth skeleton")
            Plotter.plot_mask(
                prediction[batch_idx],
                f"Prediction - ClLoss: {loss[batch_idx].item():.4f}",
            )
            Plotter.plot_mask(prediction_skeleton[batch_idx], "Prediction skeleton")
            Plotter.finish(name="cl_dice_loss")

    return loss


@MODELS.register_module()
class ClDiceLoss(nn.Module):
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
        iterations: int = 10,
        thinner: bool = False,
        epsilon: float = 1e-6,
        debug_every: int = -1,
        debug_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        if debug_every > 0:
            assert (
                debug_path is not None
            ), "debug_path must be provided if debug_every > 0"
        self._debug_every = debug_every
        self._debug_path = debug_path
        self._debug_idx = 0

        # add asserts for args

        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

        self._iterations = iterations
        self._thinner = thinner
        self._epsilon = epsilon

    def forward(
        self,
        pred,
        target,
        reduction_override=None,
        **kwargs,
    ):
        reduction = reduction_override if reduction_override else self.reduction
        pred = F.softmax(pred, dim=1)

        loss = self.loss_weight * cl_dice_loss(
            pred,
            target,
            iterations=self._iterations,
            epsilon=self._epsilon,
            thinner=self._thinner,
            debug=self._debug_every > 0 and self._debug_idx % self._debug_every == 0,
            debug_path=self._debug_path,
        )
        self._debug_idx += 1

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
