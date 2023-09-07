_base_ = "../base/unet_drive.py"

# Loss definition
model = dict(
    decode_head=dict(
        loss_decode=[
            dict(
                type="SkilLossProduct",
                loss_name="loss_skill_product",
                loss_weight=1.0,
                epsilon=0.01,
                border_factor=0.82,
                border_size=20,
                iterations=50,
            ),
            dict(type="DiceLoss", loss_name="loss_dice", loss_weight=3.0),
        ]
    )
)


# Logging
WANDB_RUN_NAME = "3-DICE-1-Skil-Product"
USE_WANDB = True
WANDB_PROJECT_SUFFIX = "_prod"
DATASET = "DRIVE_DB1"


# WANDB logging
from toolbox.printing import warn

vis_backends = [
    dict(type="LocalVisBackend"),
]
if USE_WANDB:
    # Generate a 10 character long random string
    import random
    import string

    tag = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
    vis_backends.append(
        dict(
            type="WandbVisBackend",
            init_kwargs=dict(
                project=DATASET + "_segmentation" + WANDB_PROJECT_SUFFIX,
                entity="single-shot-robot",
                name=WANDB_RUN_NAME + "_" + tag,
                group=WANDB_RUN_NAME,
            ),
        ),
    )
else:
    warn("No WANDB logging will be performed.")

# Visualization
visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer", alpha=0.6
)
