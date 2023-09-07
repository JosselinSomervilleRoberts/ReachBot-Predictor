_base_ = "../../../configs/unet/unet-s5-d16_deeplabv3_4xb4-40k_stare-128x128.py"

# Metrics
METRIC = "ReachbotMetric"
val_evaluator = dict(type=METRIC)
test_evaluator = val_evaluator


LOG_STEP_INTERVAL = 50
DEBUG_STEP_INTERVAL = -1
SAVE_STEP_INTERVAL = -1
VISUALIZE_ONE_OUT_OF = 1


# Hooks
from toolbox.printing import warn

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)
if LOG_STEP_INTERVAL > 0:
    default_hooks["logger"] = dict(
        type="LoggerHook", interval=LOG_STEP_INTERVAL, log_metric_by_epoch=False
    )
else:
    warn("No logging will be performed.")
    if "logger" in default_hooks:
        del default_hooks["logger"]

if SAVE_STEP_INTERVAL > 0:
    default_hooks["checkpoint"] = dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=SAVE_STEP_INTERVAL,
        max_keep_ckpts=1,
        save_last=False,
        save_best="avg",
        rule="greater",
    )
else:
    warn("No checkpoint will be saved.")
    if "checkpoint" in default_hooks:
        del default_hooks["checkpoint"]

if VISUALIZE_ONE_OUT_OF > 0:
    default_hooks["visualization"] = dict(
        type="SegVisualizationHook", draw=True, interval=VISUALIZE_ONE_OUT_OF
    )
else:
    warn("No visualization will be performed.")
    if "visualization" in default_hooks:
        del default_hooks["visualization"]
