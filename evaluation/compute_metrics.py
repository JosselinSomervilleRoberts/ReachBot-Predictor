"""
This file contains all the metrics used to evaluate our segmentation models.
"""

import numpy as np
import scipy
import torch
from PIL import Image

from typing import Tuple, Union, Dict, Optional
from metrics import iou_score, dice_score


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
    ratio = np.exp(-np.abs(nb_pixels_gt - nb_pixels_pred) / nb_pixels_gt)
    distances_gt = scipy.ndimage.distance_transform_edt(ground_truth == 0)
    distances_pred = scipy.ndimage.distance_transform_edt(prediction == 0)
    distance_factor_gt = np.sum(prediction * np.exp(-distances_gt ** 2 / (2 * sigma ** 2))) / nb_pixels_gt
    distance_factor_pred = np.sum(ground_truth * np.exp(-distances_pred ** 2 / (2 * sigma ** 2))) / nb_pixels_pred

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


def compute_all_metrics(
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
    