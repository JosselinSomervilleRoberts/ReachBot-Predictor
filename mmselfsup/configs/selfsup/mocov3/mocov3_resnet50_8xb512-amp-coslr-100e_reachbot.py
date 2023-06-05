_base_ = [
    "../_base_/models/mocov3_resnet50.py",
    # '../_base_/datasets/imagenet_mocov3.py',
    "../_base_/schedules/lars_coslr-200e_in1k.py",
    "../_base_/default_runtime.py",
]

# custom dataset
dataset_type = "mmcls.CustomDataset"
data_root = "../datasets/selfsup/"

view_pipeline1 = [
    dict(type="RandomResizedCrop", size=224, scale=(0.2, 1.0), backend="pillow"),
    dict(
        type="RandomApply",
        transforms=[
            dict(
                type="ColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
            )
        ],
        prob=0.8,
    ),
    dict(
        type="RandomGrayscale",
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989),
    ),
    dict(type="RandomGaussianBlur", sigma_min=0.1, sigma_max=2.0, prob=1.0),
    dict(type="RandomSolarize", prob=0.0),
    dict(type="RandomFlip", prob=0.5),
]
view_pipeline2 = [
    dict(type="RandomResizedCrop", size=224, scale=(0.2, 1.0), backend="pillow"),
    dict(
        type="RandomApply",
        transforms=[
            dict(
                type="ColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
            )
        ],
        prob=0.8,
    ),
    dict(
        type="RandomGrayscale",
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989),
    ),
    dict(type="RandomGaussianBlur", sigma_min=0.1, sigma_max=2.0, prob=0.1),
    dict(type="RandomSolarize", prob=0.2),
    dict(type="RandomFlip", prob=0.5),
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiView", num_views=[1, 1], transforms=[view_pipeline1, view_pipeline2]
    ),
    dict(type="PackSelfSupInputs", meta_keys=["img_path"]),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(type="DefaultSampler", shuffle=True),
    collate_fn=dict(type="default_collate"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file="meta/train.txt",
        data_prefix=dict(img_path="train/"),
        pipeline=train_pipeline,
    ),
)


# optimizer
optimizer = dict(type="LARS", lr=0.15, weight_decay=1e-6, momentum=0.9)
optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            "bn": dict(decay_mult=0, lars_exclude=True),
            "bias": dict(decay_mult=0, lars_exclude=True),
            # bn layer in ResNet block downsample module
            "downsample.1": dict(decay_mult=0, lars_exclude=True),
        }
    ),
)

# learning rate scheduler
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=90,
        by_epoch=True,
        begin=10,
        end=100,
        convert_to_iter_based=True,
    ),
]

# runtime settings
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=100)

# runtime settings
# only keeps the latest 3 checkpoints
default_hooks = dict(
    checkpoint=dict(max_keep_ckpts=3),
    logger=dict(type="LoggerHook", interval=100),
)
visualizer = dict(
    type="SelfSupVisualizer",
    vis_backends=[
        dict(type="LocalVisBackend", name="selfsup_local"),
        dict(type="WandbVisBackend", name="selfsup"),
    ],
)
