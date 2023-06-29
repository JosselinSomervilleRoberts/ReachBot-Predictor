import configparser
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from tqdm import tqdm

import wandb
from evaluation.compute_metrics import compute_all_metrics, log_metrics
from generate_cracks_dataset import get_bboxes_for_mask, get_square_bboxes
from generate_finetuning_dataset import convert_mask_to_binary, load_gt_masks

# config
config = configparser.ConfigParser()
config.read("config.ini")
CLASSES = ["edge", "boulder", "crack", "rough_patch"]
LABELBOX_DATASET_FOLDER = config["PATHS"]["LABELBOX_DATASET"]
FINETUNE_DATASET_FOLDER = config["PATHS"]["FINETUNE_DATASET"]
CRACKS_DATASET_FOLDER = config["PATHS"]["CRACKS_DATASET"]
CRACKS_SEG_HM = config["PATHS"]["CRACKS_SEG_HM"]
CRACKS_SEG_FULL_DATASET = config["PATHS"]["CRACKS_SEG_FULL_DATASET"]


def load_hm_mask() -> dict:  # -> Dict[int, Image]:
    """Loads the ground truth masks from the FINETUNE_DATASET_FOLDER folder."""
    ground_truth_masks = {}
    masks_paths = sorted(
        glob.glob(os.path.join(CRACKS_SEG_HM, "masks/*.png")),
        key=lambda x: int(x.split("/")[-1].split(".")[0] + "_0"),
    )
    for k, mask_path in enumerate(masks_paths):
        gt_grayscale = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        ground_truth_masks[k] = gt_grayscale

    return ground_truth_masks


def get_human_eval_metrics(output_size=128, max_size=128, min_size=32):
    class_name = "crack"
    gt_masks = load_gt_masks(class_name)
    hm_masks = load_hm_mask()
    curr_image = None
    curr_total_mask = None
    metrics = []
    for k, image_path in enumerate(
        tqdm(sorted(glob.glob(f"{FINETUNE_DATASET_FOLDER}/{class_name}/images/*.png")))
    ):
        image_number = image_path.split("/")[-1].split("_")[0]
        if image_number != curr_image:
            curr_image = image_number
            total_mask_curr_image = len(
                glob.glob(
                    f"{FINETUNE_DATASET_FOLDER}/{class_name}/images/{curr_image}_*.png"
                )
            )
            curr_total_mask = sum(
                [
                    gt_masks[k + i].astype(np.uint8) * 255
                    for i in range(total_mask_curr_image)
                ]
            )
        hm_mask = cv2.resize(
            hm_masks[int(image_number)].copy().astype(np.uint8) * 255,
            (curr_total_mask.shape[1], curr_total_mask.shape[0]),
        )

        mask = gt_masks[k]
        bboxes = get_bboxes_for_mask(mask, max_size=max_size)
        square_bboxes = get_square_bboxes(bboxes, mask, min_size=min_size)
        mask_boxes = mask.copy().astype(np.uint8) * 255
        for bbox in square_bboxes:
            mask_boxes = cv2.rectangle(
                mask_boxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2
            )

            mask_cropped = curr_total_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            mask_resized = cv2.resize(mask_cropped, (output_size, output_size))
            mask_palette = np.zeros((output_size, output_size))
            mask_palette[mask_resized > 0] = 1
            hm_mask_cropped = hm_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            hm_mask_resized = cv2.resize(
                hm_mask_cropped,
                (output_size, output_size),
            )
            hm_mask_palette = np.zeros((output_size, output_size))
            hm_mask_palette[hm_mask_resized > 0] = 1
            metrics.append(
                compute_all_metrics(
                    ground_truth=mask_palette, prediction_binary=hm_mask_palette
                )
            )

    return metrics


def get_human_eval_metrics_full_images():
    class_name = "crack"
    gt_masks = load_gt_masks(class_name)
    hm_masks = load_hm_mask()
    curr_image = None
    curr_total_mask = None
    metrics = []
    for k, image_path in enumerate(
        tqdm(sorted(glob.glob(f"{FINETUNE_DATASET_FOLDER}/{class_name}/images/*.png")))
    ):
        image_number = image_path.split("/")[-1].split("_")[0]
        if image_number != curr_image:
            curr_image = image_number
            total_mask_curr_image = len(
                glob.glob(
                    f"{FINETUNE_DATASET_FOLDER}/{class_name}/images/{curr_image}_*.png"
                )
            )
            curr_total_mask = sum(
                [
                    gt_masks[k + i].astype(np.uint8) * 255
                    for i in range(total_mask_curr_image)
                ]
            )

            hm_mask = cv2.resize(
                hm_masks[int(image_number)].copy().astype(np.uint8) * 255,
                (curr_total_mask.shape[1], curr_total_mask.shape[0]),
            )

            mask_palette = np.zeros(curr_total_mask.shape)
            mask_palette[curr_total_mask > 0] = 1
            hm_mask_palette = np.zeros(hm_mask.shape)
            hm_mask_palette[hm_mask > 0] = 1
            metrics.append(
                compute_all_metrics(
                    ground_truth=mask_palette, prediction_binary=hm_mask_palette
                )
            )

    return metrics


if __name__ == "__main__":
    wandb.init(
        project="Reachbot-cracks",
        name="human-eval-full-images",
    )
    metrics = get_human_eval_metrics_full_images()
    log_metrics(metrics)
