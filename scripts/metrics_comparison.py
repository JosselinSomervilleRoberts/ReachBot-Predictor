import matplotlib.pyplot as plt
from tqdm import tqdm

from annotation_modifiers import (
    ShiftedAnnotationModifier,
    RandomWidthAnnotationModifier,
    RandomBranchRemovalModifier,
    CombinedAnnotationModifier,
    dilation,
)

from mmseg.evaluation.metrics.metrics import crack_metrics

import numpy as np
import cv2
import argparse
import os
import torch


def dice_score_func(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    intersection = np.sum(ground_truth * prediction)
    union = np.sum(ground_truth) + np.sum(prediction)
    return (2 * intersection + 1e-8) / (union + 1e-8)


def crack_metric_func(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    metrics = crack_metrics(
        ground_truth,
        prediction,
    )
    return metrics["avg"], metrics["dice"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../datasets/combined")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--desired_dice", type=float, default=0.4)
    parser.add_argument("--desired_crack_metric_diff", type=float, default=0.1)
    parser.add_argument("--output_path", type=str, default="metric_images")
    return parser.parse_args()


modifiers = [
    ShiftedAnnotationModifier("ShiftedAnnotationModifier", shift_range=(-100, 100)),
    ShiftedAnnotationModifier("ShiftedAnnotationModifier", shift_range=(-50, 50)),
    ShiftedAnnotationModifier("ShiftedAnnotationModifier", shift_range=(-25, 25)),
    RandomWidthAnnotationModifier("RandomWidthAnnotationModifier", (0.1, 2.0)),
    RandomWidthAnnotationModifier("RandomWidthAnnotationModifier", (0.1, 0.5)),
    RandomBranchRemovalModifier(
        "RandomBranchRemovalModifier", prob_removal=0.3, selectiveness=0.5
    ),
    RandomBranchRemovalModifier(
        "RandomBranchRemovalModifier", prob_removal=0.2, selectiveness=0.2
    ),
    RandomBranchRemovalModifier(
        "RandomBranchRemovalModifier", prob_removal=0.2, selectiveness=0.5
    ),
    RandomBranchRemovalModifier(
        "RandomBranchRemovalModifier", prob_removal=0.4, selectiveness=0.2
    ),
    RandomBranchRemovalModifier(
        "RandomBranchRemovalModifier", prob_removal=0.3, selectiveness=0.5
    ),
]
combined = CombinedAnnotationModifier(
    "CombinedAnnotationModifier",
    modifiers,
    prob=0.3,
)
modifiers.append(combined)


def load_random_image(dataset_path: str):
    train_ann_path = os.path.join(dataset_path, "ann_dir/train/")
    # Choose a random image
    ann_name = np.random.choice(os.listdir(train_ann_path))
    ann_path = os.path.join(train_ann_path, ann_name)
    annotation = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
    if np.max(annotation) > 1:
        annotation = annotation / 255
    img_path = os.path.join(dataset_path, "img_dir/train/", ann_name)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if np.max(image) <= 1:
        image = image * 255
    return image, annotation


def deform_with_desired_diff(
    image: np.ndarray,
    dice_score: float,
    diff_crack_metric: float,
    tolerance: float = 0.01,
):
    dice_score_1 = 1.0
    image_1 = image.copy()
    image_2 = image.copy()
    deformed_image_1 = image_1

    idx = 0
    best_dice = 1.0
    print(f"\nTrying to find the desired dice score of {dice_score}")
    while abs(dice_score_1 - dice_score) > tolerance:
        deformed_image_1 = combined(image_1)
        dice_score_1 = dice_score_func(deformed_image_1, image)
        dice_diff_obj = abs(dice_score_1 - dice_score)
        if dice_diff_obj < best_dice:
            best_dice = dice_diff_obj
        idx += 1
        print(f"Try {idx}   Dice: {dice_diff_obj} (best: {best_dice})")
    crack_metric_1, dice_score_1 = crack_metric_func(deformed_image_1, image)

    crack_metric_2 = crack_metric_1
    deformed_image_2 = deformed_image_1.copy()
    dice_score_2 = dice_score_1

    idx = 0
    best_prod = 1.0
    print(
        f"\nTrying to find the desired crack metric difference of {diff_crack_metric}"
    )
    while (
        abs(dice_score_1 - dice_score_2) > tolerance
        or abs(crack_metric_2 - crack_metric_1) < diff_crack_metric
    ):
        deformed_image_2 = combined(image_2)
        crack_metric_2, dice_score_2 = crack_metric_func(deformed_image_2, image)
        diff_dice_obj = abs(dice_score_1 - dice_score_2)
        diff_crack_obj = abs(abs(crack_metric_2 - crack_metric_1) - diff_crack_metric)
        prod = diff_dice_obj * diff_crack_obj
        if prod < best_prod:
            best_prod = prod
        idx += 1
        print(
            f"Try {idx}   Dice: {diff_dice_obj} Crack: {diff_crack_obj} (best: {best_prod})"
        )

    # Save both images
    return (
        deformed_image_1,
        deformed_image_2,
        crack_metric_1,
        crack_metric_2,
        dice_score_1,
        dice_score_2,
    )


if __name__ == "__main__":
    args = parse_args()
    # Current path of the file
    save_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(save_dir, args.output_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx in tqdm(range(args.num_images)):
        print(f"\n\nImage {idx}")
        image, annotation = load_random_image(args.dataset_path)
        print("Image loaded!")
        (
            deformed_image_1,
            deformed_image_2,
            crack_metric_1,
            crack_metric_2,
            dice_score_1,
            dice_score_2,
        ) = deform_with_desired_diff(
            annotation,
            args.desired_dice,
            args.desired_crack_metric_diff,
            args.tolerance,
        )

        # Save the original annotation
        save_example_path = os.path.join(save_dir, str(idx))
        if not os.path.exists(save_example_path):
            os.makedirs(save_example_path)

        cv2.imwrite(os.path.join(save_example_path, "original.png"), image)
        cv2.imwrite(os.path.join(save_example_path, "annotation.png"), annotation * 255)
        cv2.imwrite(
            os.path.join(save_example_path, "deformed_1.png"), deformed_image_1 * 255
        )
        cv2.imwrite(
            os.path.join(save_example_path, "deformed_2.png"), deformed_image_2 * 255
        )

        # Save the metrics in a text file
        with open(os.path.join(save_example_path, "metrics.txt"), "w") as f:
            f.write(f"Deformed 1 crack metric: {crack_metric_1}\n")
            f.write(f"Deformed 1 dice score: {dice_score_1}\n")
            f.write(f"Deformed 2 crack metric: {crack_metric_2}\n")
            f.write(f"Deformed 2 dice score: {dice_score_2}\n")
