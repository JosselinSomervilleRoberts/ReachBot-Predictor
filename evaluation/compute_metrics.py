"""
This file contains all the metrics used to evaluate our segmentation models.
"""

import numpy as np
import scipy
import torch
from PIL import Image

from typing import Tuple, Union, Dict, Optional, List
from metrics import iou_score, dice_score
from skimage.morphology import label
from toolbox.printing import print_color
import wandb


def to_np(image: Union[np.ndarray, torch.Tensor, Image.Image], dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Converts a PIL image, a numpy array or a PyTorch tensor to a numpy array.
    Converts to an array from 0 to 1 by default.
    The output shape is (height, width, channels).

    Args:
        image: The image to convert.

    Returns:
        The converted image.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    elif not isinstance(image, np.ndarray):
        raise ValueError(
            f"Expected image to be a PIL image, a numpy array or a PyTorch "
            f"tensor. Got {type(image)}."
        )

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    if image.dtype != dtype:
        # Secial case for binary images. Uses a threshold to prevent rounding errors
        if dtype == bool:
            image = image > 0.5
        else:
            image = image.astype(dtype)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    return image


def preprocess_for_comparison(
    ground_truth: Union[np.ndarray, torch.Tensor, Image.Image],
    prediction: Union[np.ndarray, torch.Tensor, Image.Image],
    dtype: np.dtype = np.float32,
    diffuse_sigma_factor: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses the ground truth and the prediction for comparison.
    Returns both images as numpy arrays of type np.float32 (from 0 to 1).
    Check that the shapes are the same.
    """
    ground_truth = to_np(ground_truth, dtype=dtype)
    prediction = to_np(prediction, dtype=dtype)
    assert ground_truth.shape == prediction.shape, \
        f"Expected ground truth and prediction to have the same shape. " \
        f"Got {ground_truth.shape} and {prediction.shape}."
    
    # Applies a gaussian diffusion to the images
    if diffuse_sigma_factor > 0:
        assert dtype == np.float32, \
            f"Expected dtype to be np.float32 when using a gaussian diffusion. " \
            f"Got {dtype}."
        ground_truth = apply_gaussian_diffusion(ground_truth, diffuse_sigma_factor)
        prediction = apply_gaussian_diffusion(prediction, diffuse_sigma_factor)

    return ground_truth, prediction
        

def apply_gaussian_diffusion(
    image: Union[np.ndarray, torch.Tensor, Image.Image],
    sigma_factor: float = 0.1) -> np.ndarray:
    """
    Applies a gaussian diffusion to an image.
    """
    image = to_np(image)
    sigma = sigma_factor * np.sqrt(image.shape[0] * image.shape[1])
    distance = scipy.ndimage.distance_transform_edt(image == 0)
    return np.exp(-(distance ** 2 / (2 * sigma ** 2)))


def mask_to_bbox(mask: np.ndarray) -> np.ndarray:
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


def separate_masks(mask: Union[np.ndarray, torch.Tensor, Image.Image]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """If they are several convex components in the mask, separate them into individual masks.
    This returns a list of coordinates of the bounding boxes of the masks."""
    mask = to_np(mask, dtype=bool)
    separated_mask = label(mask)
    bboxes: List[np.ndarray] = []
    blobs: List[np.ndarray] = []
    for i in np.unique(separated_mask):
        if i == 0:  # background
            continue
        blob = (separated_mask == i).astype(int)
        blobs.append(blob)
        # Get the bounding box
        bbox = np.array((Image.fromarray(blob.squeeze(-1).astype(np.uint8) * 255)).getbbox())

        # Add some padding to the bounding box
        bbox = (bbox[0] - 10, bbox[1] - 10, bbox[2] + 10, bbox[3] + 10)

        # Crop the bounding box so that it is within the image
        bbox = (
            max(bbox[0], 0),
            max(bbox[1], 0),
            min(bbox[2], mask.shape[1]),
            min(bbox[3], mask.shape[0]),
        )
        bbox = np.array(bbox)
        bboxes.append(bbox)

    return blobs, bboxes


def separate_masks_from_bbox(mask: Union[np.ndarray, torch.Tensor, Image.Image], bboxes: List[np.ndarray]) -> List[np.ndarray]:
    """Given a mask and some bounding boxes, crop the mask to each bounding box."""
    mask = to_np(mask, dtype=bool)
    blobs: List[np.ndarray] = []

    for bbox in bboxes:
        blob = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        blobs.append(blob)

    return blobs


def binary_iou(
    ground_truth: Union[np.ndarray, torch.Tensor, Image.Image],
    prediction: Union[np.ndarray, torch.Tensor, Image.Image]) -> float:
    """
    Computes the Intersection over Union (IoU) score between a ground truth
    segmentation and a predicted segmentation.
    Both the ground truth and the prediction must be binary images.

    Args:
        ground_truth: The ground truth segmentation.
        prediction: The predicted segmentation.

    Returns:
        The IoU score.
    """
    ground_truth, prediction = preprocess_for_comparison(ground_truth, prediction, dtype=bool)

    # Computes the intersection and the union
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)

    # Computes the IoU score
    return iou_score(intersection=np.sum(intersection), union=np.sum(union))


def float_iou(
    ground_truth: Union[np.ndarray, torch.Tensor, Image.Image],
    prediction: Union[np.ndarray, torch.Tensor, Image.Image]) -> float:
    """
    Computes the Intersection over Union (IoU) score between a ground truth
    segmentation and a predicted segmentation.
    The ground truth image must be a binary image and the predicted image represents
    the probability of each pixel to belong to the class, it it therefore a float.

    The intersection is defined as the sum of the probabilities of the pixels
    that are correctly classified.

    The union is defined as the sum of the probabilities of the entire prediction.

    Args:
        ground_truth: The ground truth segmentation.
        prediction: The predicted segmentation.
    
    Returns:
        The IoU score.
    """
    ground_truth, prediction = preprocess_for_comparison(ground_truth, prediction)

    # Computes the intersection and the union
    intersection = np.sum(ground_truth * prediction)
    union = np.sum(prediction)

    # Computes the IoU score
    return iou_score(intersection=intersection, union=union)


def binary_dice(
    ground_truth: Union[np.ndarray, torch.Tensor, Image.Image],
    prediction: Union[np.ndarray, torch.Tensor, Image.Image]) -> float:
    """
    Computes the Dice score between a ground truth segmentation and a predicted
    segmentation.
    Both the ground truth and the prediction must be binary images.

    Args:
        ground_truth: The ground truth segmentation.
        prediction: The predicted segmentation.

    Returns:
        The Dice score.
    """
    ground_truth, prediction = preprocess_for_comparison(ground_truth, prediction, dtype=bool)

    # Computes the intersection and the union
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)

    # Computes the Dice score
    return dice_score(intersection=np.sum(intersection), union=np.sum(union))


def float_dice(
    ground_truth: Union[np.ndarray, torch.Tensor, Image.Image],
    prediction: Union[np.ndarray, torch.Tensor, Image.Image]) -> float:
    """
    Computes the Dice score between a ground truth segmentation and a predicted
    segmentation.
    The ground truth image must be a binary image and the predicted image represents
    the probability of each pixel to belong to the class, it it therefore a float.

    The intersection is defined as the sum of the probabilities of the pixels
    that are correctly classified.

    The union is defined as the sum of the probabilities of the entire prediction.

    Args:
        ground_truth: The ground truth segmentation.
        prediction: The predicted segmentation.

    Returns:
        The Dice score.
    """

    # Converts to numpy array without the channel dimension
    ground_truth = to_np(ground_truth)[:, :, 0]
    prediction = to_np(prediction)[:, :, 0]

    # Computes the intersection and the union
    intersection = np.sum(np.minimum(ground_truth, prediction))
    union = np.sum(np.maximum(ground_truth, prediction))

    # Computes the Dice score
    return dice_score(intersection=intersection, union=union)


def binary_distance_proximity(
    ground_truth: Union[np.ndarray, torch.Tensor, Image.Image],
    prediction: Union[np.ndarray, torch.Tensor, Image.Image],
    sigma_factor: float = 0.1) -> float:
    """
    Computes the distance score defined by:
    score = ratio * sqrt(distance_factor_gt * distance_factor_pred)
    where:
    - ratio = exp(-|nb_pixels_gt - nb_pixels_pred| / nb_pixels_gt)
    - distance_factor_gt = (1/nb_pixels_gt) * sum_{x in ground_truth} exp(-distance(x, closest pixel==1 in prediction) / (2 * sigma^2))
    - distance_factor_pred = (1/nb_pixels_pred) * sum_{x in prediction} exp(-distance(x, closest pixel==1 in ground_truth) / (2 * sigma^2))
    where sigma is a function of the size of the image.
    - sigma = sigma_factor * sqrt(height * width)
    For each pixel of the prediction computes

    Args:
        ground_truth: The ground truth segmentation.
        prediction: The predicted segmentation.
        sigma_factor: The factor used to compute sigma.

    Returns:
        The distance score.
    """
    ground_truth, prediction = preprocess_for_comparison(ground_truth, prediction, dtype=bool)

    if np.sum(prediction) == 0:
        return 0.0

    # Computes the distance between the two images
    sigma = sigma_factor * np.sqrt(ground_truth.shape[0] * ground_truth.shape[1])
    nb_pixels_gt = np.sum(ground_truth)
    nb_pixels_pred = np.sum(prediction)
    ratio = np.exp(-np.abs(nb_pixels_gt - nb_pixels_pred) / (1+nb_pixels_gt))
    distances_gt = scipy.ndimage.distance_transform_edt(ground_truth == 0)
    distances_pred = scipy.ndimage.distance_transform_edt(prediction == 0)
    distance_factor_gt = np.sum(prediction * np.exp(-distances_gt ** 2 / (2 * sigma ** 2))) / (1+nb_pixels_gt)
    distance_factor_pred = np.sum(ground_truth * np.exp(-distances_pred ** 2 / (2 * sigma ** 2))) / (1+nb_pixels_pred)

    # Computes the distance score
    return ratio * np.sqrt(distance_factor_gt * distance_factor_pred)


def float_distance_proximity(
    ground_truth: Union[np.ndarray, torch.Tensor, Image.Image],
    prediction: Union[np.ndarray, torch.Tensor, Image.Image],
    sigma_factor: float = 0.1,
    threshold: float = 0.5) -> float:
    """
    Generalization of the distance score.

    For the distance_factor_pred, we compute the distance for pixel of the
    prediction with the closest pixel in the ground truth times the probability
    of the pixel in the prediction.
    The distance is weighted by the probability of the pixel in the prediction.

    For the distance_factor_gt, we compute the distance for pixel of the
    ground truth with the closest pixel in the prediction bigger than a threshold.
    The distance is weughted by the number of pixels in the ground truth.

    score = ratio * sqrt(distance_factor_gt * distance_factor_pred)
    where:
    - ratio = exp(- |nb_pixels_gt - nb_pixels_pred| / nb_pixels_gt)
    - distance_factor_gt = (1/nb_pixels_gt) * sum_{x in ground_truth} exp(-distance(x, closest pixel>=threshold in prediction) / (2 * sigma^2))
    - distance_factor_pred = (1/sum(prob_x_pred)) * sum_{x for all prediction} prob_x * exp(-distance(x, closest pixel==1 in ground_truth) / (2 * sigma^2))
    where sigma is a function of the size of the image.
    - sigma = sigma_factor * sqrt(height * width)

    Args:
        ground_truth: The ground truth segmentation.
        prediction: The predicted segmentation.
        sigma_factor: The factor used to compute sigma.
        threshold: The threshold used to binarize the prediction.

    Returns:
        The distance score.
    """
    ground_truth, prediction = preprocess_for_comparison(ground_truth, prediction)

    if np.sum(prediction >= threshold) == 0:
        return 0.0

    # Computes the distance between the two images
    sigma = sigma_factor * np.sqrt(ground_truth.shape[0] * ground_truth.shape[1])
    nb_pixels_gt = np.sum(ground_truth)         # Number of pixels in the ground truth mask
    nb_pixels_pred = np.sum(prediction)         # Sum of the probabilities
    ratio = np.exp(-np.abs(nb_pixels_gt - nb_pixels_pred) / nb_pixels_gt)
    distances_gt = scipy.ndimage.distance_transform_edt(prediction < threshold)
    distances_pred = scipy.ndimage.distance_transform_edt(ground_truth == 0)
    distance_factor_gt = np.sum(prediction * np.exp(-distances_gt ** 2 / (2 * sigma ** 2))) / nb_pixels_gt
    distance_factor_pred = np.sum(ground_truth * np.exp(-distances_pred ** 2 / (2 * sigma ** 2))) / nb_pixels_pred

    # Computes the distance score
    return ratio * np.sqrt(distance_factor_gt * distance_factor_pred)

        
def gaussian_binary_iou(
    ground_truth: Union[np.ndarray, torch.Tensor, Image.Image],
    prediction: Union[np.ndarray, torch.Tensor, Image.Image],
    sigma_factor: float = 0.1) -> float:
    """
    Computes the IoU score between a ground truth segmentation and a predicted
    segmentation after applying a gaussian diffusion to both images.
    """
    diffuse_gt, diffuse_pred = preprocess_for_comparison(ground_truth, prediction, diffuse_sigma_factor=sigma_factor)
    return float_iou(ground_truth=diffuse_gt, prediction=diffuse_pred)


def gaussian_binary_dice(
    ground_truth: Union[np.ndarray, torch.Tensor, Image.Image],
    prediction: Union[np.ndarray, torch.Tensor, Image.Image],
    sigma_factor: float = 0.1) -> float:
    """
    Computes the Dice score between a ground truth segmentation and a predicted
    segmentation after applying a gaussian diffusion to both images.
    """
    diffuse_gt, diffuse_pred = preprocess_for_comparison(ground_truth, prediction, diffuse_sigma_factor=sigma_factor)
    return float_dice(ground_truth=diffuse_gt, prediction=diffuse_pred)


def compute_all_metrics_on_single_image(
    ground_truth: Union[np.ndarray, torch.Tensor, Image.Image],
    prediction: Optional[Union[np.ndarray, torch.Tensor, Image.Image]] = None,
    prediction_binary: Optional[Union[np.ndarray, torch.Tensor, Image.Image]] = None,
    sigma_factor: float = 0.02,
    threshold: float = 0.5) -> Dict[str, float]:
    """
    Computes all the metrics.
    """
    # Checks that either prediction or prediction_binary is not None
    assert prediction is not None or prediction_binary is not None, \
        f"Expected either prediction or prediction_binary to be not None. " \
        f"Got prediction={prediction} and prediction_binary={prediction_binary}."

    # Converts to numpy array.
    # This is done in all the metrics but the functions is very fast
    # if the image is already a numpy array.
    ground_truth = to_np(ground_truth)
    if prediction is not None: prediction = to_np(prediction)
    if prediction_binary is not None: prediction_binary = to_np(prediction_binary)

    # Computes the metrics
    metrics = {}
    if prediction is not None:
        metrics["float_intersection"] = np.sum(np.minimum(ground_truth, prediction))
        metrics["float_union"] = np.sum(np.maximum(ground_truth, prediction))
        metrics["float_iou"] = iou_score(intersection=metrics["float_intersection"], union=metrics["float_union"])
        metrics["float_dice"] = dice_score(intersection=metrics["float_intersection"], union=metrics["float_union"])
        metrics["float_distance_proximity"] = float_distance_proximity(ground_truth=ground_truth, prediction=prediction, sigma_factor=sigma_factor, threshold=threshold)
        metrics["average_float"] = (metrics["float_iou"] + metrics["float_dice"] + metrics["float_distance_proximity"]) / 3
    if prediction_binary is not None:
        diffuse_gt, diffuse_pred = preprocess_for_comparison(ground_truth, prediction_binary, diffuse_sigma_factor=sigma_factor)
        diffuse_intersection = np.sum(np.minimum(diffuse_gt, diffuse_pred))
        diffuse_union = np.sum(np.maximum(diffuse_gt, diffuse_pred))
        metrics["binary_intersection"] = np.sum(np.logical_and(ground_truth, prediction_binary))
        metrics["binary_union"] = np.sum(np.logical_or(ground_truth, prediction_binary))
        metrics["binary_iou"] = iou_score(intersection=metrics["binary_intersection"], union=metrics["binary_union"])
        metrics["binary_dice"] = dice_score(intersection=metrics["binary_intersection"], union=metrics["binary_union"])
        metrics["binary_distance_proximity"] = binary_distance_proximity(ground_truth=ground_truth, prediction=prediction_binary, sigma_factor=sigma_factor)
        metrics["gaussian_binary_iou"] = iou_score(intersection=diffuse_intersection, union=diffuse_union)
        metrics["gaussian_binary_dice"] = dice_score(intersection=diffuse_intersection, union=diffuse_union)
        metrics["average_binary"] = (metrics["binary_iou"] + metrics["binary_dice"] + metrics["binary_distance_proximity"] + metrics["gaussian_binary_iou"] + metrics["gaussian_binary_dice"]) / 5

    # Compute average metrics
    avg: float = 0.0
    count: int = 0
    for key, value in metrics.items():
        if not key.startswith("average") and "intersection" not in key and "union" not in key:
            avg += value
            count += 1
    metrics["average"] = avg / count

    return metrics
    

def compute_all_metrics(
    ground_truth: Union[np.ndarray, torch.Tensor, Image.Image],
    prediction: Optional[Union[np.ndarray, torch.Tensor, Image.Image]] = None,
    prediction_binary: Optional[Union[np.ndarray, torch.Tensor, Image.Image]] = None,
    sigma_factor: float = 0.02,
    threshold: float=0.5,
    separate_images: bool = True) -> dict:
    """
    Computes all the metrics.
    This is done on the global image and on each individual mask.
    """

    # Computes the metrics on the global image
    metrics_full: Dict[str, float] = compute_all_metrics_on_single_image(
        ground_truth=ground_truth,
        prediction=prediction,
        prediction_binary=prediction_binary,
        sigma_factor=sigma_factor,
        threshold=threshold)
    if not separate_images:
        return metrics_full

    # Computes the metrics on each individual mask
    metrics_individual: List[Dict[str, float]] = []

    # Separate the masks of the ground truth
    ground_truth_masks, bboxes = separate_masks(ground_truth)
    # Separate the masks of the prediction
    prediction_masks, prediction_binary_masks = None, None
    if prediction is not None:
        prediction_masks = separate_masks_from_bbox(prediction, bboxes)
    if prediction_binary is not None:
        prediction_binary_masks = separate_masks_from_bbox(prediction_binary, bboxes)
    
    # Compute the metrics for each mask
    for i, ground_truth_mask in enumerate(ground_truth_masks):
        metrics_individual.append(compute_all_metrics_on_single_image(
            ground_truth=ground_truth_mask,
            prediction=prediction_masks[i] if prediction is not None else None,
            prediction_binary=prediction_binary_masks[i] if prediction_binary is not None else None,
            sigma_factor=sigma_factor,
            threshold=threshold))

    # Compute average and std metrics for the individual masks
    metrics_avg_individual: Dict[str, float] = {}
    metrics_std_individual: Dict[str, float] = {}
    for key in metrics_individual[0].keys():
        if "intersection" not in key and "union" not in key:
            metrics_avg_individual[key] = np.mean([x[key] for x in metrics_individual])
            metrics_std_individual[key] = np.std([x[key] for x in metrics_individual])

    metrics = {
        "full_mask": metrics_full,
        "individual_masks": metrics_individual,
        "avg_individual_masks": metrics_avg_individual,
        "std_individual_masks": metrics_std_individual,
    }

    return metrics

        
def log_metrics(metrics: Union[dict, list], log_to_wandb: bool = True, step: Optional[int] = None) -> None:
    """
    Prints the metrics.
    """
    wandb_metrics = {}
    if isinstance(metrics, dict):
        # Check if "full_mask" is in the dict
        if "full_mask" in metrics:
            # This means that we have metrics for full image and individual masks
            print_color(f"Metrics for 1 image (with {len(metrics['individual_masks'])} masks):", color="blue")
            for key, val_full in metrics["full_mask"].items():
                if not "intersection" in key and not "union" in key:
                    val_std_indiv = metrics["std_individual_masks"][key]
                    val_avg_indiv = metrics["avg_individual_masks"][key]
                    wandb_metrics["Avg full mask/" + key] = val_full
                    wandb_metrics["Avg individual masks/" + key] = val_avg_indiv
                    wandb_metrics["Std individual masks/" + key] = val_std_indiv
                    if key == "average":
                        print_color(f" {key}: {val_full:.3f} (Indiv: {val_avg_indiv:.3f} ± {val_std_indiv:.3f})", color="bold")
                    else:
                        print(f" - {key}: {val_full:.3f} (Indiv: {val_avg_indiv:.3f} ± {val_std_indiv:.3f})")
        else:
            # This means that we only have metrics for the fll_image, directly in the dict
            print_color(f"Metrics for 1 image:", color="blue")
            for key, val in metrics.items():
                if not "intersection" in key and not "union" in key:
                    wandb_metrics["Full image/" + key] = val
                    if key == "average":
                        print_color(f" {key}: {val:.3f}", color="bold")
                    else:
                        print(f" - {key}: {val:.3f}")

    elif isinstance(metrics, list):
        # This means that we have a list of metrics. For each of them compute the average and std
        has_full_mask = "full_mask" in metrics[0]
        metrics_avg_full: Dict[str, float] = {}
        metrics_std_full: Dict[str, float] = {}
        metrics_avg_indiv: Dict[str, float] = {}
        metrics_std_indiv: Dict[str, float] = {}
        if has_full_mask:
            for key in metrics[0]["full_mask"].keys():
                if "intersection" not in key and "union" not in key:
                    metrics_avg_full[key] = np.mean([metric["full_mask"][key] for metric in metrics])
                    metrics_std_full[key] = np.std([metric["full_mask"][key] for metric in metrics])
                    metrics_avg_indiv[key] = np.mean([metric["avg_individual_masks"][key] for metric in metrics])
                    metrics_std_indiv[key] = np.mean([metric["std_individual_masks"][key] for metric in metrics])
                    wandb_metrics["Avg full mask/" + key] = metrics_avg_full[key]
                    wandb_metrics["Std full mask/" + key] = metrics_std_full[key]
                    wandb_metrics["Avg individual masks/" + key] = metrics_avg_indiv[key]
                    wandb_metrics["Std individual masks/" + key] = metrics_std_indiv[key]
        else:
            for key in metrics[0].keys():
                if "intersection" not in key and "union" not in key:
                    metrics_avg_full[key] = np.mean([x[key] for x in metrics])
                    metrics_std_full[key] = np.std([x[key] for x in metrics])
                    wandb_metrics["Avg full mask/" + key] = metrics_avg_full[key]
                    wandb_metrics["Std full mask/" + key] = metrics_std_full[key]

        # Print the metrics to the console
        if has_full_mask:
            print_color(f"Metrics for {len(metrics)} images (with {sum([len(metric['individual_masks']) for metric in metrics])} masks):", color="blue")
        else:
            print_color(f"Metrics for {len(metrics)} images:", color="blue")
        for key, val_full in metrics_avg_full.items():
            val_std_full = metrics_std_full[key]
            if not "intersection" in key and not "union" in key:
                color = "bold" if key == "average" else None
                if has_full_mask:
                    val_avg_indiv = metrics_avg_indiv[key]
                    val_std_indiv = metrics_std_indiv[key]
                    print_color(f" {key}: {val_full:.3f} ± {val_std_full:.3f} (Indiv: {val_avg_indiv:.3f} ± {val_std_indiv:.3f})", color=color)
                else:
                    print_color(f" {key}: {val_full:.3f} ± {val_std_full:.3f}", color=color)

    # Log the metrics to wandb
    if log_to_wandb:
        if step is not None:
            wandb.log(wandb_metrics, step=step)
        else:
            wandb.log(wandb_metrics)
