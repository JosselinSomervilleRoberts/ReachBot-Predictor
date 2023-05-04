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

# config
config = configparser.ConfigParser()
config.read('config.ini')
DATASET_FOLDER = config["PATHS"]["LABELBOX_DATASET"]
LB_API_KEY = config['SECRETS']['LABELBOX_API_KEY']
LB_PROJECT_ID = config['SECRETS']['LABELBOX_PROJECT_ID']
CLASS_TO_COLOR = {
        "edge": [int(c) for c in config["DISPLAY"]["COLOR_EDGE"].split(' ')],
        "crack": [int(c) for c in config["DISPLAY"]["COLOR_CRACK"].split(' ')],
        "boulder": [int(c) for c in config["DISPLAY"]["COLOR_BOULDER"].split(' ')],
        "rough_patch": [int(c) for c in config["DISPLAY"]["COLOR_ROUGH_PATCH"].split(' ')],
    }

def check_if_dataset_exists() -> bool:
    """Checks if the dataset already exists."""
    return os.path.exists(DATASET_FOLDER)

def get_masks(image_number: int) -> list:
    """Gets the masks for a given image number."""
    masks = []
    img_folder = os.path.join(DATASET_FOLDER, str(image_number))
    for mask_path in os.listdir(img_folder):
        if mask_path.endswith(".png") and mask_path.startswith("mask"):
            mask = Image.open(os.path.join(img_folder, mask_path))
            masks.append(mask)
    return masks

def get_classes(image_number: int) -> List[str]:
    """Gets the classes for a given image number."""
    classes = []
    img_folder = os.path.join(DATASET_FOLDER, str(image_number))
    with open(os.path.join(img_folder, "classes.txt"), "r") as f:
        classes = f.read().split("\n")
    return classes

def generate_labelbox_dataset() -> None:
    import labelbox

    # Get the labelbox project
    lb = labelbox.Client(api_key=LB_API_KEY)
    project = lb.get_project(LB_PROJECT_ID)

    print("Downloading labels...")
    labels = project.label_generator()
    labels = project.export_labels(download = True)
    print("Done")

    # Create folder name "dataset" if it doesn't exist
    import os
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)

    # Download all images and save them in the "dataset" folder
    print("Downloading images...")
    for i in tqdm(range(len(labels))):
        x = labels[i]['Labeled Data']
        response = requests.get(x)
        img = Image.open(BytesIO(response.content))

        # Create folder named i if it doesn't exist
        img_folder = os.path.join(DATASET_FOLDER, str(i))
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        else:
            print(f"Image {i} already exists")
            continue

        # Save image
        img.save(img_folder + "/image.png")

        # Get all masks
        classes = []
        objects = labels[i]['Label']['objects']
        for j, object in enumerate(objects):
            instanceURI = object['instanceURI']
            response = requests.get(instanceURI)
            img = Image.open(BytesIO(response.content))
            img = Image.eval(img, lambda a: 255 - a)
            img.putalpha(255)
            img.save(img_folder + "/mask_" + str(j) + ".png")
            classes.append(objects[j]['title'])

        # Save classes
        with open(img_folder + "/classes.txt", "w") as f:
            f.write("\n".join(classes))
    print("Done")

def show_img_with_all_masks(image_number: int) -> None:
    """Shows the image with all the masks on top of it."""
    # Load the image
    img_folder = os.path.join(DATASET_FOLDER, str(image_number))
    img = cv2.imread(img_folder + "/image.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load the masks
    masks = get_masks(image_number)
    classes = get_classes(image_number)

    for i, mask in enumerate(masks):
        # Draw the mask on top of the image as class_to_color[classes[i]] with an alpha of 0.5 where
        # the mask is not 0.
        mask_img = np.array(mask)[:,:,0]
        color = CLASS_TO_COLOR[classes[i]]
        color.append(0.5)
        
        # Apply the color where mask_img is 0 with alpha blending
        img[mask_img == 0] = img[mask_img == 0] * (1 - color[3]) + np.array(color[:3]) * color[3]
        

    # Show the image with all the masks on top of it
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    if not check_if_dataset_exists():
        print("Dataset does not exist")
        generate_labelbox_dataset()
    else:
        print("Dataset already exists")
        show_img_with_all_masks(0)