import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import configparser
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List
from toolbox.printing import warn, ldebug
import shutil



# config
config = configparser.ConfigParser()
config.read('./config.ini')
LABELBOX_DATASET_FOLDER = config["PATHS"]["LABELBOX_DATASET"]
MASKRCNN_DATASET_FOLDER = config["PATHS"]["MASKRCNN_DATASET"]
LB_API_KEY = config['SECRETS']['LABELBOX_API_KEY']
LB_PROJECT_ID = config['SECRETS']['LABELBOX_PROJECT_ID']
CLASS_TO_COLOR = {
        "edge": [int(c) for c in config["DISPLAY"]["COLOR_EDGE"].split(' ')],
        "crack": [int(c) for c in config["DISPLAY"]["COLOR_CRACK"].split(' ')],
        "boulder": [int(c) for c in config["DISPLAY"]["COLOR_BOULDER"].split(' ')],
        "rough_patch": [int(c) for c in config["DISPLAY"]["COLOR_ROUGH_PATCH"].split(' ')],
    }




def generate_maskrcnn_dataset():
    # Check if the mask r-cnn dataset exists
    # if check_if_masckrcnn_dataset_exists():
    #     print('Mask R-CNN dataset already exists')
    #     return

    # Create an 'images' subfolder containing all 'image.png' files from the labelbox dataset.
    # Images are saved as 'image_index.png' with image_index being the index of the image.

    # Create the 'images' subfolder in the new dataset folder
    new_images_folder = os.path.join(MASKRCNN_DATASET_FOLDER, "images")
    os.makedirs(new_images_folder, exist_ok=True)


    # Create a 'masks' subfolder containing one 'p' (palette) type 'mask_image_index.png' mask for each image.
    new_masks_folder = os.path.join(MASKRCNN_DATASET_FOLDER, "masks")
    os.makedirs(new_masks_folder, exist_ok=True)


    # Iterate over subfolders in the existing dataset
    for subfolder_name in tqdm(os.listdir(LABELBOX_DATASET_FOLDER), desc='Processing images and masks'):
        subfolder_path = os.path.join(LABELBOX_DATASET_FOLDER, subfolder_name)

        # Load the class information from the text file
        classes_file_path = os.path.join(subfolder_path, "classes.txt")
        with open(classes_file_path, "r") as f:
            classes = f.read().split("\n")


        # Check if the 'boulder' class is present in the image
        if "boulder" in classes:
            # Check if the subfolder contains an 'image.png' file
            image_path = os.path.join(subfolder_path, "image.png")
            if os.path.isfile(image_path):
                
                # Copy the 'image.png' file to the 'images' subfolder in the new dataset
                new_image_path = os.path.join(new_images_folder, f"image_{subfolder_name}.png")
                shutil.copy2(image_path, new_image_path)

            # Each mask .png file will become one channel of the 'p' (palette) type PIL .png image
            mask_channels = []
            mask_files = os.listdir(subfolder_path)
            # Get the index of each mask containded in the file name
            mask_indices = [int(file_name.split("_")[1].split(".")[0]) for file_name in mask_files if file_name.startswith("mask_") and file_name.endswith(".png")]
            # Sort the mask files by index
            mask_files = ['mask_' + str(i) + '.png' for i in sorted(mask_indices)]

            for i, file_name in enumerate(mask_files):
                if file_name.startswith('mask_') and file_name.endswith('.png'):
                    if classes[i] == "boulder":
                        mask_path = os.path.join(subfolder_path, file_name)
                        mask_image = Image.open(mask_path)
                        mask_image = mask_image.convert('L')
                        mask_image = np.array(mask_image)
                        # invert the mask (boulders are white)
                        mask_image = np.where(mask_image == 0, 255, 0)
                        
                        mask_channels.append(mask_image)

                
            image_palette = np.zeros(mask_image.shape, dtype=np.uint8)
            for i, mask in enumerate(mask_channels):
                # Get the instance number
                instance_number = i + 1
                # Get the indices of where the mask is not 0
                indices = np.where(mask != 0)
                # Set the palette image to the instance number at the indices
                image_palette[indices] = instance_number
            
            # Save the palette image as a type 'p' image from PIL
            new_mask_path = os.path.join(new_masks_folder, f"mask_{subfolder_name}.png")
            #print('image palette numpy:', image_palette.shape)
            image_palette = Image.fromarray(np.uint8(image_palette))
            #print('image palette PIL:', image_palette)
            image_palette.save(new_mask_path)




if __name__ == '__main__':
    generate_maskrcnn_dataset()