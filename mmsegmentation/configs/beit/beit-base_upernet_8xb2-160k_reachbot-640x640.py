_base_ = [
    '../_base_/models/upernet_beit.py', "../_base_/datasets/reachbot.py",
    '../_base_/default_runtime.py', "../_base_/schedules/schedule_80k.py"
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/beit_base_patch16_224_pt22k_ft22k.pth',
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(426, 426)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9),
    accumulative_counts=2)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=20000, val_interval=1000)
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
