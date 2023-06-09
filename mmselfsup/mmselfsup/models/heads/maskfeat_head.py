# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class MaskFeatPretrainHead(BaseModule):
    """Pre-training head for MaskFeat.

    It computes reconstruction loss between prediction and target in masked
    region.

    Args:
        loss (dict): Config dict for module of loss functions.
    """

    def __init__(self, loss: dict) -> None:
        super().__init__()
        self.loss = MODELS.build(loss)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Forward head.

        Args:
            latent (torch.Tensor): Predictions,
                which is of shape B x (1 + L) x C.
            target (torch.Tensor): Hog features, which is of shape B x L x C.
            mask (torch.Tensor): The mask of the hog features,
                which is of shape B x H x W.

        Returns:
            torch.Tensor: The loss tensor.
        """
        mask = mask.flatten(1).bool()
        loss = self.loss(pred[:, 1:], target, mask)

        return loss
