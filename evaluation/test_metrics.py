"""
Loads an image from the dataset and its corresponding ground truth segmentation.
Generate a fake predictions by applying a small random translation and some noise.
Then show the image, the two masks on top of the image and the two masks side by side.
Compute all the metrics and display them."""

import os
import configparser
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy
import time

from compute_metrics import compute_all_metrics, to_np

# Load the image and the ground truth segmentation
# config
config = configparser.ConfigParser()
config.read('config.ini')
FINETUNE_DATASET_FOLDER = config["PATHS"]["FINETUNE_DATASET"]
path_image = os.path.join(FINETUNE_DATASET_FOLDER, "crack", "images", "0.png")
path_mask = os.path.join(FINETUNE_DATASET_FOLDER, "crack", "masks", "0.png")
path_bbox = os.path.join(FINETUNE_DATASET_FOLDER, "crack", "bboxes", "0.txt")

# Load the image and the ground truth segmentation and crop them to the bounding box
image = Image.open(path_image)
mask = Image.open(path_mask)

# The bounding box is saved as coordinates in a binary mask
with open(path_bbox, "r") as f:
    bbox = np.array([int(x) for x in f.read().split()])

# Crop the image and the mask to the bounding box
image = image.crop(bbox)
mask = mask.crop(bbox)
mask = 1. - to_np(mask)

# Generate a fake prediction
prediction = mask.copy()
prediction = np.roll(prediction, 10, axis=0)
prediction = np.roll(prediction, 10, axis=1)
# Add noise
prediction = prediction + np.abs(np.random.normal(0, 2.0, prediction.shape)) / (1+scipy.ndimage.distance_transform_edt(prediction <= 0.5))**2
prediction = prediction >= 0.5


# Compute all the metrics
# Time the computation
print("Computing metrics...")
start_time = time.time()
metrics = compute_all_metrics(ground_truth=mask, prediction_binary=prediction)
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
print("\nMetrics:")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value}")

# Test what happends if the prediction is the same as the ground truth
prediction_1 = mask.copy()
metrics_1 = compute_all_metrics(ground_truth=mask, prediction_binary=prediction_1)
print("\nMetrics when the prediction is the same as the ground truth:")
for metric_name, metric_value in metrics_1.items():
    print(f"{metric_name}: {metric_value}")

# Test what happends if the prediction is the opposite of the ground truth
prediction_2 = 1 - mask.copy()
metrics_2 = compute_all_metrics(ground_truth=mask, prediction_binary=prediction_2)
print("\nMetrics when the prediction is the opposite of the ground truth:")
for metric_name, metric_value in metrics_2.items():
    print(f"{metric_name}: {metric_value}")

# Test what happens if the prediction is full of 0s
prediction_3 = np.zeros_like(mask)
metrics_3 = compute_all_metrics(ground_truth=mask, prediction_binary=prediction_3)
print("\nMetrics when the prediction is full of 0s:")
for metric_name, metric_value in metrics_3.items():
    print(f"{metric_name}: {metric_value}")

# Test what happens if the prediction is full of 1s
prediction_4 = np.ones_like(mask)
metrics_4 = compute_all_metrics(ground_truth=mask, prediction_binary=prediction_4)
print("\nMetrics when the prediction is full of 1s:")
for metric_name, metric_value in metrics_4.items():
    print(f"{metric_name}: {metric_value}")

# Test what happens if the prediction is random
prediction_5 = np.random.rand(*mask.shape) >= 0.5
metrics_5 = compute_all_metrics(ground_truth=mask, prediction_binary=prediction_5)
print("\nMetrics when the prediction is random:")
for metric_name, metric_value in metrics_5.items():
    print(f"{metric_name}: {metric_value}")

intersection = np.logical_and(mask, prediction)
union = np.logical_or(mask, prediction)

# Show the image, the ground truth segmentation and the prediction in one image
# Below show the image with ground truth mask, predict mask, ground trith + predict mask
fig, ax = plt.subplots(3, 3, figsize=(15, 5))
ax[0,0].imshow(image)
ax[0,0].set_title("Image")
ax[0,1].imshow(mask)
ax[0,1].set_title("Ground truth segmentation")
ax[0,2].imshow(prediction)
ax[0,2].set_title("Prediction")
ax[1,0].imshow(image)
# Show the mask in red. Since this is one channel, expand to 3 channels and set the red channel to the mask
ax[1,0].imshow(mask, alpha=0.5)
ax[1,0].set_title("Ground truth segmentation")
ax[1,1].imshow(image)
ax[1,1].imshow(prediction, alpha=0.5)
ax[1,1].set_title("Prediction")
ax[1,2].imshow(image)
ax[1,2].imshow(mask, alpha=0.5)
ax[1,2].imshow(prediction, alpha=0.5)
ax[1,2].set_title("Ground truth segmentation and prediction")
ax[2,0].imshow(intersection)
ax[2,0].set_title("Intersection")
ax[2,1].imshow(union)
ax[2,1].set_title("Union")

plt.show()
