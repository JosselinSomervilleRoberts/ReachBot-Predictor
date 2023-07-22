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


def threshold(x, val: float = 0.5, sharpness: float = 10.0):
    """Soft threshold function.

    Args:
        x: Input tensor.
        val: Threshold value.
        sharpness: Sharpness of the threshold function.

    Returns:
        Thresholded tensor.
    """
    return 1 / (1 + torch.exp(-sharpness * (x - val)))


@weighted_loss
def skil_loss(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    use_dice: bool = True,
    smooth_threshold_factor: float = 10.0,
    iterations: int = 10,
    border_size: int = 25,
    sigma: float = 10.0,
    epsilon: float = 1e-6,
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

    if smooth_threshold_factor < 0:
        threshold_fn = lambda x: x
    else:
        assert smooth_threshold_factor >= 1, "smooth_threshold_factor must be greater than 1"
        threshold_fn = lambda x: threshold(x, sharpness=smooth_threshold_factor)

    # Format the prediciton and ground truth
    # We only keep the class 1 as we do not want to evaluate the background.
    prediction = threshold_fn(prediction[:, 1, :, :].float())
    ground_truth = ground_truth.float()
    assert prediction.shape[0] == ground_truth.shape[0]

    # Smooth skeletonization
    if thinner:
        ground_truth_skeleton = soft_skeletonize_thin(ground_truth, iterations)
        prediction_skeleton = soft_skeletonize_thin(prediction, iterations)
    else:
        ground_truth_skeleton = soft_skeletonize(ground_truth, iterations)
        prediction_skeleton = soft_skeletonize(prediction, iterations)

    # Enlarge the skeleton
    ground_truth_border = apply_smooth_gaussian_diffusion(
        ground_truth_skeleton, border_size, sigma
    )
    prediction_border = apply_smooth_gaussian_diffusion(
        prediction_skeleton, border_size, sigma
    )

    # Compute the loss
    if use_dice:
        loss: torch.Tensor = 1 - soft_dice(ground_truth_border, prediction_border)
    else:
        p1_num = torch.sum(ground_truth_border * prediction_skeleton, dim=(-2, -1))
        p1_den = torch.sum(prediction_skeleton, dim=(-2, -1))
        p1 = (p1_num + epsilon) / (p1_den + epsilon)

        p2_num = torch.sum(prediction_border * ground_truth_skeleton, dim=(-2, -1))
        p2_den = torch.sum(ground_truth_skeleton, dim=(-2, -1))
        p2 = (p2_num + epsilon) / (p2_den + epsilon)

        loss: torch.Tensor = 1 - torch.sqrt(p1 * p2)

    # Debug if needed
    if debug:
        n_rows = 2
        n_cols = 4

        if use_dice:
            numerator = ground_truth_border * prediction_border
            denominator = ground_truth_border**2 + prediction_border**2
        else:
            numerator = ground_truth_border * prediction_skeleton
            denominator = ground_truth_skeleton * prediction_border

        for batch_idx in range(prediction.shape[0]):
            Plotter.start(n_rows, n_cols, debug_path)
            Plotter.plot_mask(ground_truth[batch_idx], "Ground truth")
            Plotter.plot_mask(ground_truth_skeleton[batch_idx], "Ground truth skeleton")
            Plotter.plot_mask(ground_truth_border[batch_idx], "Ground truth border")
            Plotter.plot_mask(numerator[batch_idx], "Loss first term")
            Plotter.plot_mask(prediction[batch_idx], f"Prediction - Loss: {loss[batch_idx].item():.4f}")
            Plotter.plot_mask(prediction_skeleton[batch_idx], "Prediction skeleton")
            Plotter.plot_mask(prediction_border[batch_idx], "Prediction border")
            Plotter.plot_mask(denominator[batch_idx], "Loss second term")
            Plotter.finish()

    return loss


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
        sigma: float = 10.0,
        border_size: int = 25,
        iterations: int = 10,
        smooth_threshold_factor: float = 10.0,
        thinner: bool = False,
        use_dice: bool = True,
        epsilon: float = 1e-6,
        debug: bool = False,
        debug_path: str = None,
        **kwargs,
    ):
        super().__init__()

        # add asserts for args

        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

        self._debug = debug
        self._debug_path = debug_path
        self._sigma = sigma
        self._border_size = border_size
        self._iterations = iterations
        self._smooth_threshold_factor = smooth_threshold_factor
        self._thinner = thinner
        self._use_dice = use_dice
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

        loss = self.loss_weight * skil_loss(
            pred,
            target,
            use_dice=self._use_dice,
            smooth_threshold_factor=self._smooth_threshold_factor,
            iterations=self._iterations,
            border_size=self._border_size,
            sigma=self._sigma,
            epsilon=self._epsilon,
            thinner=self._thinner,
            debug=self._debug,
            debug_path=self._debug_path
            )

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