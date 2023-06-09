# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
from mmengine.model import BaseModule
from torch import nn

from mmselfsup.registry import MODELS


@MODELS.register_module()
class BEiTLoss(BaseModule):
    """Loss function for BEiT.

    The BEiTLoss supports 2 diffenrent logits shared 1 target, like BEiT v2.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss_cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits: Union[Tuple[torch.Tensor], torch.Tensor],
                target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function of BEiT Loss.

        Args:
            logits (torch.Tensor): The outputs from the decoder.
            target (torch.Tensor): The targets generated by dalle.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The main loss.
        """
        if isinstance(logits, torch.Tensor):
            loss = self.loss_cross_entropy(logits, target)
            return loss
        elif isinstance(logits, Tuple):
            loss_1 = self.loss_cross_entropy(logits[0], target)
            loss_2 = self.loss_cross_entropy(logits[1], target)
            return loss_1, loss_2
