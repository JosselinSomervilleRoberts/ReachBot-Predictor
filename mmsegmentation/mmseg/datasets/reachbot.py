# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio
from mmseg.registry import DATASETS

from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ReachbotDataset(BaseSegDataset):
    METAINFO = dict(
        classes=("background", "crack"), palette=[[0, 0, 0], [255, 255, 255]]
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(
            img_suffix=".png", seg_map_suffix=".png", reduce_zero_label=False, **kwargs
        )
        assert fileio.exists(
            self.data_prefix["img_path"], backend_args=self.backend_args
        )
