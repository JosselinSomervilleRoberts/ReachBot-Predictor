# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.target_generators import VQKD

vqkd_encoder = dict(
    arch='base',
    img_size=224,
    patch_size=16,
    in_channels=3,
    out_indices=-1,
    drop_rate=0.,
    drop_path_rate=0.,
    norm_cfg=dict(type='LN', eps=1e-6),
    final_norm=True,
    with_cls_token=True,
    avg_token=False,
    frozen_stages=-1,
    output_cls_token=False,
    use_abs_pos_emb=True,
    use_rel_pos_bias=False,
    use_shared_rel_pos_bias=False,
    layer_scale_init_value=0.,
    interpolate_mode='bicubic',
    patch_cfg=dict(),
    layer_cfgs=dict(),
    init_cfg=None)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_vqkd():
    model = VQKD(encoder_config=vqkd_encoder)
    fake_inputs = torch.rand((2, 3, 224, 224))
    fake_outputs = model(fake_inputs)

    assert list(fake_outputs.shape) == [2, 196]
