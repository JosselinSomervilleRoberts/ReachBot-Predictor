import configparser
import glob
import os
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from skimage.morphology import label
from toolbox.printing import warn
from tqdm import tqdm

from generate_labelbox_dataset import (
    check_if_labelbox_dataset_exists,
    generate_labelbox_dataset,
)

# config
config = configparser.ConfigParser()
config.read("config.ini")

CLASSES = ["edge", "boulder", "crack", "rough_patch"]
LABELBOX_DATASET_FOLDER = config["PATHS"]["LABELBOX_DATASET"]
FINETUNE_DATASET_FOLDER = config["PATHS"]["FINETUNE_DATASET"]


def load_gt_masks(class_name: str) -> dict:  # -> Dict[int, Image]:
    """Loads the ground truth masks from the FINETUNE_DATASET_FOLDER folder."""
    ground_truth_masks = {}
    masks_paths = sorted(
        glob.glob(os.path.join(FINETUNE_DATASET_FOLDER, class_name, "masks/*.png"))
    )
    for k, mask_path in enumerate(masks_paths):
        gt_grayscale = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        ground_truth_masks[int(k)] = gt_grayscale == 0
    return ground_truth_masks


def load_bbox_coords(class_name: str) -> List[np.ndarray]:
    """Loads the bounding box coordinates from the FINETUNE_DATASET_FOLDER folder."""
    bbox_coords = {}
    bboxes_path = sorted(
        glob.glob(os.path.join(FINETUNE_DATASET_FOLDER, class_name, "bboxes/*.txt"))
    )
    for k, bbox_path in enumerate(bboxes_path):
        with open(bbox_path, "r") as f:
            bbox = np.array([int(x) for x in f.read().split()])
        bbox_coords[int(k)] = bbox
    return bbox_coords


# Helper function provided in https://github.com/facebookresearch/segment-anything/blob/9e8f1309c94f1128a6e5c047a10fdcb02fc8d651/notebooks/predictor_example.ipynb
def show_mask(mask: Image, ax: plt.Axes, color: tuple = None) -> None:
    if color is None:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# Helper function provided in https://github.com/facebookresearch/segment-anything/blob/9e8f1309c94f1128a6e5c047a10fdcb02fc8d651/notebooks/predictor_example.ipynb
def show_box(box: np.ndarray, ax: plt.Axes) -> None:
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_image_with_mask(
    class_name: str,
    image_number: int,
    bbox_coords: List[np.ndarray],
    ground_truth_masks: dict,
) -> None:
    image = cv2.imread(
        f"{FINETUNE_DATASET_FOLDER}/{class_name}/images/{image_number}.png"
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_box(bbox_coords[image_number], plt.gca())
    show_mask(ground_truth_masks[image_number], plt.gca())
    plt.axis("off")
    plt.show()


def convert_mask_to_binary(mask: Image) -> Image:
    """Converts a mask to a binary mask."""
    mask = mask.convert("L")
    mask = mask.point(lambda x: 0 if x >= 128 else 255, "1")
    return mask


def mask_to_bbox(mask: Image) -> np.ndarray:
    """Converts a mask to a bounding box such that the bounding box contains the mask.
    The bounding box is generated with some noise and padding."""
    # Get the bounding box
    bbox = mask.getbbox()

    # Add some noise and padding to the bounding box
    bbox = (bbox[0] - 10, bbox[1] - 10, bbox[2] + 10, bbox[3] + 10)

    # Crop the bounding box so that it is within the image
    bbox = (
        max(bbox[0], 0),
        max(bbox[1], 0),
        min(bbox[2], mask.width),
        min(bbox[3], mask.height),
    )

    # Convert the bounding box to a numpy array
    bbox = np.array(bbox)

    return bbox


def check_if_finetuning_dataset_exists() -> bool:
    """Checks if the finetuning dataset exists."""
    return os.path.exists(FINETUNE_DATASET_FOLDER)


def separate_masks(mask: Image) -> List[np.ndarray]:
    """If they are several convex components in the mask, separate them into individual masks."""
    mask = np.array(mask).astype(bool)
    separated_mask = label(mask)
    blobs = []
    for i in np.unique(separated_mask):
        if i == 0:  # background
            continue
        blobs.append((separated_mask == i).astype(int))
    return blobs


def generate_finetuning_dataset() -> None:
    """Generates the finetuning dataset from the dataset folder."""

    # Check if the dataset folder exists
    if not check_if_labelbox_dataset_exists():
        warn("The dataset folder does not exist. It will be generated now.")
        generate_labelbox_dataset()

    # Get the number of images in the dataset
    num_images = len(os.listdir(LABELBOX_DATASET_FOLDER))
    print(f"Found {num_images} images")

    # Create the output folder
    if not os.path.exists(FINETUNE_DATASET_FOLDER):
        os.mkdir(FINETUNE_DATASET_FOLDER)
        for class_mask in CLASSES:
            os.mkdir(os.path.join(FINETUNE_DATASET_FOLDER, class_mask))
            os.mkdir(os.path.join(FINETUNE_DATASET_FOLDER, class_mask, "images"))
            os.mkdir(os.path.join(FINETUNE_DATASET_FOLDER, class_mask, "masks"))
            os.mkdir(os.path.join(FINETUNE_DATASET_FOLDER, class_mask, "bboxes"))

    # For each image folder, open classes.txt and get the classes
    # For each line (i.e. mask) in classes.txt, load the mask if the class is in CLASSES
    # Then generate a bounding box for the mask and save it in the output folder
    # The bounding box is saved as a binary mask
    # It if generated with some noise and padding

    # The final architecture of the dataset folder is:
    # <FINETUNE_DATASET_FOLDER/class_name>
    # ├── images
    # │   ├── 0.png
    # │   ├── 1.png
    # │   ├── ...
    # │   └── N.png
    # │── masks
    # │   ├── 0.png
    # │   ├── 1.png
    # │   ├── ...
    # │   └── N.png
    # │── bboxes
    # │   ├── 0.png
    # │   ├── 1.png
    # │   ├── ...
    # │   └── N.png

    num_masks_added = {}
    num_masks_failed = {}
    # list directories in the dataset folder
    list_dirs = os.listdir(LABELBOX_DATASET_FOLDER)
    dict_images = {i: list_dirs[i] for i in range(len(list_dirs))}
    for i in tqdm(dict_images):
        img_folder = os.path.join(LABELBOX_DATASET_FOLDER, str(dict_images[i]))
        classes_file = os.path.join(img_folder, "classes.txt")

        for class_mask in CLASSES:
            num_masks_added[class_mask] = 0
            num_masks_failed[class_mask] = 0
        # Get the classes
        with open(classes_file, "r") as f:
            classes = f.read().splitlines()

        # Get the masks
        for j, class_mask in enumerate(classes):
            if class_mask in CLASSES:
                try:
                    # Load the mask
                    mask_path: str = os.path.join(img_folder, "mask_" + str(j) + ".png")
                    original_mask: Image = Image.open(mask_path)
                    original_mask = convert_mask_to_binary(original_mask)
                    masks: list = separate_masks(original_mask)

                    if len(masks) > 1:
                        print(f"Found {len(masks)} masks in image {i} mask {j}")
                    elif len(masks) == 0:
                        print(f"Found 0 masks in image {i} mask {j}")
                        continue

                    folder = os.path.join(FINETUNE_DATASET_FOLDER, class_mask)
                    for mask in masks:
                        # Convert mask from np.ndarray to Image
                        mask = Image.fromarray(mask.astype(np.uint8) * 255)

                        bbox: np.ndarray = mask_to_bbox(mask)

                        # Invert the mask
                        mask = Image.fromarray(np.invert(np.array(mask)))

                        # Save the mask
                        mask.save(
                            os.path.join(
                                folder,
                                "masks",
                                f"{i}_{str(num_masks_added[class_mask])}" + ".png",
                            )
                        )

                        # Copy the image
                        img_path: str = os.path.join(img_folder, "image.png")
                        img: Image = Image.open(img_path)
                        img.save(
                            os.path.join(
                                folder,
                                "images",
                                f"{i}_{str(num_masks_added[class_mask])}" + ".png",
                            )
                        )

                        # Save the bbox with numpy savetxt
                        np.savetxt(
                            os.path.join(
                                folder,
                                "bboxes",
                                f"{i}_{str(num_masks_added[class_mask])}" + ".txt",
                            ),
                            bbox,
                            fmt="%d",
                        )

                        num_masks_added[class_mask] += 1
                except Exception as e:
                    num_masks_failed[class_mask] += 1
                    print(f"Error processing image {i} mask {j}: {e}")
                    continue

    for class_mask in CLASSES:
        print(f"Added {num_masks_added[class_mask]} masks for class {class_mask}")
        print(
            f"Failed to add {num_masks_failed[class_mask]} masks for class {class_mask}"
        )


if __name__ == "__main__":
    if not check_if_finetuning_dataset_exists():
        print("Finetuning dataset does not exist. Generating...")
        generate_finetuning_dataset()
    else:
        print("Finetuning dataset already exists. Skipping generation.")
        class_name = "boulder"
        gt_masks = load_gt_masks(class_name)
        bbox_coords = load_bbox_coords(class_name)
        show_image_with_mask(class_name, 0, bbox_coords, gt_masks)
