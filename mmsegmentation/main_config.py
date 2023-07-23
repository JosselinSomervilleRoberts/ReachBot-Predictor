# =========== TO CHANGE =========== #

# Define the model here (2 modifications!) (from custom_configs/models)
# string
MODEL = "vit_vit-b16_mln_upernet"
_base_ = ["custom_configs/models/vit-b16_mln_upernet.py",]

# Define the dataset (from custom_configs/datasets)
# string
DATASET = "cracks_full"

# Define the loss (from custom_configs/losses)
# dictionnary mapping loss to coefficient
LOSSES = {
    "skil": 3.0,
    "dice": 1.0,
}

# Define the learning rate schedule (from custom_configs/schedules)
# string
LR = "aggressive"

# Define either the path of a custom checkpoint or None to use the default checkpoint
# string or None
CHECKPOINT = None

# Number of epochs to train for
# int
num_epochs = 100
# ================================= #


# Associates a model name to its default weights
default_checkpoint = {
    "vit_vit-b16_mln_upernet": "pretrain/upernet_vit-b16_mln_512x512_80k_ade20k_20210624_130547-0403cee1_fix.pth"
}

# Associates a loss to its class and name
losses_mappings = {
    "skil": ("SkilLoss", "loss_skill"),
    "dice": ("DiceLoss", "loss_dice"),
}



# Auto batch size and crop size
from custom_configs.memory import *


# Dataset
if DATASET == "cracks_cropped":
    from custom_configs.datasets.cracks_cropped import *
elif DATASET == "cracks_full":
    from custom_configs.datasets.cracks_full import *
elif DATASET == "boulders_cropped":
    from custom_configs.datasets.boulders_cropped import *
elif DATASET == "boulders_full":
    from custom_configs.datasets.boulders_full import *
else:
    raise Exception(f"Unknown dataset: {DATASET}")
(
    train_dataloader,
    test_dataloader,
    val_dataloader,
    train_pipeline,
    test_pipeline,
) = get_dataset(crop_size, train_batch_size)


# Learning rate
if LR == "aggressive":
    from custom_configs.schedules.aggressive import *
else:
    raise Exception(f"Unknown LR schedule: {LR}")
param_scheduler = get_scheduler(num_epochs * num_training_samples // train_batch_size)


# Metrics
val_evaluator = dict(type="ReachbotMetric")
test_evaluator = val_evaluator


# Make sure to evaluate every 1 epoch
train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=num_epochs * num_training_samples // train_batch_size,
    val_interval=num_training_samples // train_batch_size
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")


# Model cfg
if CHECKPOINT is None:
    CHECKPOINT = default_checkpoint[MODEL]
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=CHECKPOINT,
    decode_head=dict(
        num_classes=2,
        loss_decode=[
            dict(type=losses_mappings[loss_name][0], loss_name=losses_mappings[loss_name][1], loss_weight=LOSSES[loss_name])
            for loss_name in LOSSES
        ],
    ),
    auxiliary_head=dict(
        num_classes=2,
        loss_decode=[
            dict(type=losses_mappings[loss_name][0], loss_name=losses_mappings[loss_name][1], loss_weight=LOSSES[loss_name] * 0.4)
            for loss_name in LOSSES
        ],
    ),
)


# Hooks
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=1, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=num_training_samples // train_batch_size),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook", draw=True, interval=1),
)


# WANDB logging
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


# Visualization
visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer", alpha=0.6
)