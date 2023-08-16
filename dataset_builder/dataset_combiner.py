import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

from annotation_modifiers import (
    ShiftedAnnotationModifier,
    RandomBranchRemovalModifier,
    RandomWidthAnnotationModifier,
    CombinedAnnotationModifier,
)


# The OUTPOUT_FOLDER will have this format:

# OUTPUT_FOLDER
# ├── ann_dir
# │   ├── train
# │   │   ├── 0.png
# │   │   ├── 1.png
# │   │   ├── ...
# │   │   └── {{num_train}}.png
# │   └── val
# │       ├── 0.png
# │       ├── 1.png
# │       ├── ...
# │       └── {{num_val}}.png
# └── img_dir
#     ├── train
#     │   ├── 0.png
#     │   ├── 1.png
#     │   ├── ...
#     │   └── {{num_train}}.png
#     └── val
#         ├── 0.png
#         ├── 1.png
#         ├── ...
#         └── {{num_val}}.png


# Each folder input should have this format:

# INPUTS[i]["path"]
# ├── images
# │   ├── 0.png
# │   ├── 1.png
# │   ├── ...
# │   └── {{num_images}}.png
# └── masks
#     ├── 0.png
#     ├── 1.png
#     ├── ...
#     └── {{num_masks}}.png


# The object INPUTS[i] should have this format:
# {
#     "path": PATH_TO_FOLDER,
#     "for_eval": LIST_OF_INDICES_FOR_EVAL,
#     "for_train": LIST_OF_INDICES_FOR_TRAIN,
# }

RESIZE_LONGER_SIDE = 1024
OUTPUT_FOLDER = "../datasets/vessels/combined_degraded"
INPUTS = [
    {
        "path": "../datasets/original_data/drive",
        "for_eval": list(range(0, 5)),
        "for_train": list(range(5, 20)),
    },
    {
        "path": "../datasets/original_data/hrf",
        "for_eval": list(range(0, 11)),
        "for_train": list(range(11, 45)),
    },
    {
        "path": "../datasets/original_data/chase_db1",
        "for_eval": list(range(0, 7)),
        "for_train": list(range(7, 28)),
    },
    {
        "path": "../datasets/original_data/stare",
        "for_eval": list(range(0, 5)),
        "for_train": list(range(5, 20)),
    },
]
MODIFIERS = [
    ShiftedAnnotationModifier("ShiftedAnnotationModifier", shift_range=(-10, 10)),
    RandomWidthAnnotationModifier("RandomWidthAnnotationModifier", (0.1, 2.0)),
    RandomBranchRemovalModifier(
        "RandomBranchRemovalModifier", prob_removal=0.35, selectiveness=0.2
    ),
]
PROB_MODIFIERS = 0.5

modifier = None
if len(MODIFIERS) > 0:
    modifier = CombinedAnnotationModifier(
        "CombinedAnnotationModifier",
        MODIFIERS,
        prob=PROB_MODIFIERS,
    )


# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

idx_train = 0
idx_val = 0
for input in INPUTS:
    # Create ann_dir and img_dir if they don't exist
    ann_dir = os.path.join(OUTPUT_FOLDER, "ann_dir")
    img_dir = os.path.join(OUTPUT_FOLDER, "img_dir")
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # Create train and val folders if they don't exist
    for root_folder in [ann_dir, img_dir]:
        train_folder = os.path.join(root_folder, "train")
        val_folder = os.path.join(root_folder, "val")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

    # Copy images and masks to the output folder
    for i in tqdm(input["for_train"], desc="Copying images and masks for training"):
        # Copy image and resize longer side to RESIZE_LONGER_SIDE
        src = os.path.join(input["path"], "images", str(i) + ".png")
        dst = os.path.join(img_dir, "train", str(idx_train) + ".png")
        shutil.copyfile(src, dst)

        # Copy mask and apply modifier
        src = os.path.join(input["path"], "masks", str(i) + ".png")
        dst = os.path.join(ann_dir, "train", str(idx_train) + ".png")
        mask = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if np.max(mask) > 1:
            mask = mask / 255
        if modifier is not None:
            mask = modifier(mask)
        cv2.imwrite(dst, mask * 255)

        idx_train += 1

    for i in tqdm(input["for_eval"], desc="Copying images and masks for evaluation"):
        # Copy image
        src = os.path.join(input["path"], "images", str(i) + ".png")
        dst = os.path.join(img_dir, "val", str(idx_val) + ".png")
        shutil.copyfile(src, dst)

        # Copy mask without modification
        src = os.path.join(input["path"], "masks", str(i) + ".png")
        dst = os.path.join(ann_dir, "val", str(idx_val) + ".png")
        shutil.copyfile(src, dst)

        idx_val += 1

print(f"Successfuly created dataset in {OUTPUT_FOLDER}")
print(f" - Number of training images: {idx_train}")
print(f" - Number of validation images: {idx_val}")
