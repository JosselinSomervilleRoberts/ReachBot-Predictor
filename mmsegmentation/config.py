# =========== TO CHANGE =========== #

# Name of the run
# string
WANDB_RUN_NAME = "Skill"

# Define the model here (2 modifications!) (from custom_configs/models)
# string
MODEL = "vit_vit-b16_mln_upernet"
_base_ = ["custom_configs/models/vit-b16_mln_upernet.py",]

# Define the dataset (from custom_configs/datasets)
# string
DATASET = "cracks_full"

# Define the loss (from custom_configs/main)
# dictionnary mapping loss to coefficient
LOSSES = {
    #"skil": 3.0,
    "dice": 3.0,
    "cross_entropy": 1.0,
    "cl_dice": 3.0,
}

# Metric name
# string
METRIC = "ReachbotMetric"

# Define the learning rate schedule (from custom_configs/schedules)
# string
LR = "aggressive"

# Define either the path of a custom checkpoint or None to use the default checkpoint
# string or None
CHECKPOINT = None

# Number of epochs to train for
# int
num_epochs = 100

# Debug path
# string or None
DEBUG_PATH = "/media/jsomerviller/SSD2/output"

# Logging config, uncomment the one you want

# EXPERIMENTAL (to see results fast)
# No visualization, no logging, no saving, metrics every few epochs
LOG_STEP_INTERVAL = -1
DEBUG_STEP_INTERVAL = -1
SAVE_EPOCH_INTERVAL = -1
VISUALIZE_ONE_OUT_OF = -1
EVAL_EPOCH_INTERVAL = 5
USE_WANDB = False

# # DEBUGGING (to see what is happening)
# # Debugging, logging, few visualization, no saving, metrics every few epochs
# LOG_STEP_INTERVAL = 5
# DEBUG_STEP_INTERVAL = 10
# SAVE_EPOCH_INTERVAL = -1
# VISUALIZE_ONE_OUT_OF = 5
# EVAL_EPOCH_INTERVAL = 3
# USE_WANDB = False

# # PRODUCTION (to train a model and save the results)
# # No debugging but logging, visualization, saving, metrics every epoch
# LOG_STEP_INTERVAL = 1
# DEBUG_STEP_INTERVAL = -1
# SAVE_EPOCH_INTERVAL = 1
# VISUALIZE_ONE_OUT_OF = 1
# EVAL_EPOCH_INTERVAL = 1
# USE_WANDB = True

# ================================= #


# Auto batch size and crop size
from custom_configs.memory import *


# Imports
from custom_configs.main import default_checkpoint, losses_mappings, datasets_mappings, schedules_mappings
from toolbox.printing import warn
import importlib


# Dataset
if DATASET in datasets_mappings:
    dataset_module = importlib.import_module(datasets_mappings[DATASET])
else:
    raise Exception(f"Unknown dataset: {DATASET}")
(
    train_dataloader,
    test_dataloader,
    val_dataloader,
    train_pipeline,
    test_pipeline,
    dataset_type,
    data_root,
    num_training_samples,
) = dataset_module.get_dataset(crop_size, train_batch_size)
del dataset_module


# Learning rate
if LR in schedules_mappings:
    lr_module = importlib.import_module(schedules_mappings[LR])
else:
    raise Exception(f"Unknown learning rate schedule: {LR}")
param_scheduler = lr_module.get_scheduler(num_epochs * num_training_samples // train_batch_size)
del lr_module


# Metrics
val_evaluator = dict(type=METRIC)
test_evaluator = val_evaluator


# Make sure to evaluate every EVAL_EPOCH_INTERVAL epochs
assert EVAL_EPOCH_INTERVAL > 1, "EVAL_EPOCH_INTERVAL must be > 1"
train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=num_epochs * num_training_samples // train_batch_size,
    val_interval=EVAL_EPOCH_INTERVAL * num_training_samples // train_batch_size
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
            dict(type=losses_mappings[loss_name][0], loss_name=losses_mappings[loss_name][1], loss_weight=LOSSES[loss_name], debug_every=DEBUG_STEP_INTERVAL, debug_path=DEBUG_PATH)
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
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)
if LOG_STEP_INTERVAL > 0:
    default_hooks["logger"] = dict(type="LoggerHook", interval=LOG_STEP_INTERVAL, log_metric_by_epoch=False)
else:
    warn("No logging will be performed.")

if SAVE_EPOCH_INTERVAL > 0:
    default_hooks["checkpoint"] = dict(type="CheckpointHook", by_epoch=False, interval=SAVE_EPOCH_INTERVAL * num_training_samples // train_batch_size)
else:
    warn("No checkpoint will be saved.")

if VISUALIZE_ONE_OUT_OF > 0:
    default_hooks["visualization"] = dict(type="SegVisualizationHook", draw=True, interval=VISUALIZE_ONE_OUT_OF)
else:
    warn("No visualization will be performed.")



# WANDB logging
vis_backends = [
    dict(type="LocalVisBackend"),
]
if USE_WANDB:
    vis_backends.append(
        dict(
            type="WandbVisBackend",
            init_kwargs=dict(
                project=DATASET + "_segmentation",
                entity="single-shot-robot",
                name=WANDB_RUN_NAME,
            ),
        ),
    )
else:
    warn("No WANDB logging will be performed.")


# Visualization
visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer", alpha=0.6
)