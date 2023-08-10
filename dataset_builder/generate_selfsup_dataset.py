import configparser
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from tqdm import tqdm

from generate_finetuning_dataset import convert_mask_to_binary, load_gt_masks

# config
config = configparser.ConfigParser()
config.read("config.ini")  # TODO: Fix config path and variables
SELFSUP_SOURCE_DATASET_FOLDER = config["PATHS"]["SELFSUP_SOURCE_DATASET"]
SELFSUP_DATASET_FOLDER = config["PATHS"]["SELFSUP_DATASET"]


def get_images_paths():
    """
    Returns a list of all the images paths in the source dataset
    """
    images_paths = (
        glob.glob(os.path.join(SELFSUP_SOURCE_DATASET_FOLDER, "*.jpg"))
        + glob.glob(os.path.join(SELFSUP_SOURCE_DATASET_FOLDER, "*.png"))
        + glob.glob(os.path.join(SELFSUP_SOURCE_DATASET_FOLDER, "*.JPG"))
        + glob.glob(os.path.join(SELFSUP_SOURCE_DATASET_FOLDER, "*.jpeg"))
    )
    return images_paths


if __name__ == "__main__":
    if not os.path.exists(os.path.join(SELFSUP_DATASET_FOLDER, "images")):
        os.makedirs(os.path.join(SELFSUP_DATASET_FOLDER, "images"))

    patch_size = 256
    nb_images = 0
    images_paths = get_images_paths()
    for image_path in tqdm(images_paths):
        image = Image.open(image_path)
        nb_x_steps = image.size[0] // (patch_size) - 1
        nb_y_steps = image.size[1] // (patch_size) - 1
        for x in range(nb_x_steps):
            for y in range(nb_y_steps):
                left = x * (patch_size)
                top = y * (patch_size)
                right = left + patch_size
                bottom = top + patch_size
                patch = image.crop((left, top, right, bottom))
                patch.save(
                    os.path.join(SELFSUP_DATASET_FOLDER, "images", f"{nb_images}.png")
                )
                nb_images += 1
