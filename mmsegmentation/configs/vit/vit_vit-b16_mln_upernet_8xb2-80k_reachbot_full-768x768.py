_base_ = [
    "../_base_/models/upernet_vit-b16_ln_mln.py",
    "../_base_/datasets/reachbot_full.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained="pretrain/upernet_vit-b16_mln_512x512_80k_ade20k_20210624_130547-0403cee1_fix.pth",
    decode_head=dict(
        num_classes=2,
        loss_decode=[
            dict(type="SkilLoss", loss_name="loss_skill", loss_weight=1.0),
            dict(type="DiceLoss", loss_name="loss_dice", loss_weight=1.0),
        ],
    ),
    auxiliary_head=dict(
        num_classes=2,
        loss_decode=[
            dict(type="SkilLoss", loss_name="loss_skill", loss_weight=0.4),
            dict(type="DiceLoss", loss_name="loss_dice", loss_weight=0.4),
        ],
    ),
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "pos_embed": dict(decay_mult=0.0),
            "cls_token": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=10000,
        by_epoch=False,
    ),
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=5000, val_interval=50)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=1, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=500),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook", draw=True, interval=1),
)

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="WandbVisBackend",
        init_kwargs=dict(
            project="cracks_segmentation",
            entity="single-shot-robot",
            name="full-skill",
        ),
    ),
]
visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer", alpha=0.6
)
