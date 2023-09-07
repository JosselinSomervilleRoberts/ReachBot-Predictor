import configparser
import glob
import os
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# config
config = configparser.ConfigParser()
config.read("config.ini")

LABELBOX_DATASET_FOLDER = config["PATHS"]["LABELBOX_DATASET"]
CRACKS_SEG_FULL_DATASET = config["PATHS"]["CRACKS_SEG_FULL_DATASET"]


def generate_cracks_dataset(image_scale=1024):
    # Get the number of images in the dataset
    num_images = len(os.listdir(LABELBOX_DATASET_FOLDER))
    print(f"Found {num_images} images")

    # Create the output folder
    if not os.path.exists(CRACKS_SEG_FULL_DATASET):
        os.makedirs(os.path.join(CRACKS_SEG_FULL_DATASET, "images"))
        os.makedirs(os.path.join(CRACKS_SEG_FULL_DATASET, "masks"))

    list_dirs = os.listdir(LABELBOX_DATASET_FOLDER)
    dict_images = {i: list_dirs[i] for i in range(len(list_dirs))}

    for i in tqdm(dict_images):
        img_folder = os.path.join(LABELBOX_DATASET_FOLDER, str(dict_images[i]))
        classes_file = os.path.join(img_folder, "classes.txt")
        # Get the classes
        with open(classes_file, "r") as f:
            classes = f.read().splitlines()

        # Get the masks
        if "crack" in classes:
            # Check if the subfolder contains an 'image.png' file
            image_path = os.path.join(img_folder, "image.png")
            # if os.path.isfile(image_path):
            # Copy the 'image.png' file to the 'images' subfolder in the new dataset
            image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # resize image so that the largest dimension is image_scale
            max_dim = max(image.shape)
            scale = image_scale / max_dim
            image = cv2.resize(image, None, fx=scale, fy=scale)

            cv2.imwrite(
                os.path.join(
                    CRACKS_SEG_FULL_DATASET,
                    "images",
                    f"{i}.png",
                ),
                image,
            )

            # Each mask .png file will become one channel of the 'p' (palette) type PIL .png image
            all_masks = []
            mask_files = glob.glob(os.path.join(img_folder, "mask*.png"))
            for j, mask_path in enumerate(mask_files):
                if classes[j] == "crack":
                    mask_image = Image.open(mask_path)
                    mask_image = mask_image.convert("L")
                    mask_image = np.array(mask_image)
                    # invert the mask (boulders are white)
                    mask_image = np.where(mask_image == 0, 255, 0)

                    all_masks.append(mask_image)

            mask_summed = sum(all_masks)
            mask_summed = (mask_summed != 0).astype(np.uint8)
            mask_summed = cv2.resize(mask_summed, None, fx=scale, fy=scale)
            cv2.imwrite(
                os.path.join(
                    CRACKS_SEG_FULL_DATASET,
                    "masks",
                    f"{i}.png",
                ),
                mask_summed,
            )


if __name__ == "__main__":
    generate_cracks_dataset()
