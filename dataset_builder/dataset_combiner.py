import os
import shutil


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

OUTPUT_FOLDER = "../datasets/combined"
INPUTS = [
    {
        "path": "../datasets/original_data/cracks_real",
        "for_eval": list(range(0, 50)),
        "for_train": list(range(50, 100)),
    },
    {
        "path": "../datasets/original_data/cracks_generated",
        "for_eval": [],
        "for_train": list(range(0, 136)),
    },
]


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
    for i in input["for_train"]:
        # Copy image
        src = os.path.join(input["path"], "images", str(i) + ".png")
        dst = os.path.join(img_dir, "train", str(idx_train) + ".png")
        shutil.copyfile(src, dst)

        # Copy mask
        src = os.path.join(input["path"], "masks", str(i) + ".png")
        dst = os.path.join(ann_dir, "train", str(idx_train) + ".png")
        shutil.copyfile(src, dst)

        idx_train += 1

    for i in input["for_eval"]:
        # Copy image
        src = os.path.join(input["path"], "images", str(i) + ".png")
        dst = os.path.join(img_dir, "val", str(idx_val) + ".png")
        shutil.copyfile(src, dst)

        # Copy mask
        src = os.path.join(input["path"], "masks", str(i) + ".png")
        dst = os.path.join(ann_dir, "val", str(idx_val) + ".png")
        shutil.copyfile(src, dst)

        idx_val += 1

print(f"Successfuly created dataset in {OUTPUT_FOLDER}")
print(f" - Number of training images: {idx_train}")
print(f" - Number of validation images: {idx_val}")
