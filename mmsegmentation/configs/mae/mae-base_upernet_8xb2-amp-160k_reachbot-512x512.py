_base_ = [
    '../_base_/models/upernet_mae.py',
    "../_base_/datasets/reachbot.py",
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='../mmselfsup/work_dirs/selfsup/mae_vit-base-p16_8xb512-coslr-400e_reachbot/epoch_100.pth',
    backbone=dict(
        type='MAE',
        img_size=(512, 512),
        patch_size=16,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        init_values=1.0,
        drop_path_rate=0.1,
        out_indices=[3, 5, 7, 11]),
    neck=dict(embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[768, 768, 768, 768], num_classes=2, channels=768),
    auxiliary_head=dict(in_channels=768, num_classes=2),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65),
    constructor='LayerDecayOptimizerConstructor')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# mixed precision
fp16 = dict(loss_scale='dynamic')
train_cfg = dict(type="IterBasedTrainLoop", max_iters=20000, val_interval=4000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=1000),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook", draw=True, interval=5),
)


# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
