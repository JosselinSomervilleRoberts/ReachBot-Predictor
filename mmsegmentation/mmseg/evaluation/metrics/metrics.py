import numpy as np
from PIL import Image
from typing import Tuple, Union, Dict
from skimage.measure import label
import scipy
from skimage.morphology import skeletonize
import torch

from mmseg.evaluation.metrics.utils import crop_as_small_as_possible
from mmseg.evaluation.metrics.smooth_gaussian_diffusion import apply_smooth_gaussian_diffusion


def crack_metrics(
    ground_truth: Union[np.ndarray, Image.Image],
    prediction: Union[np.ndarray, Image.Image],
    sigma_factor: float = 0.1,
    crop: bool = True) -> Dict[str, float]:
    ground_truth, prediction = preprocess_for_comparison(ground_truth, prediction, dtype=bool)
    sigma = sigma_factor * np.sqrt(ground_truth.shape[0] * ground_truth.shape[1])
    if crop: ground_truth, prediction = crop_as_small_as_possible(ground_truth, prediction)
    num_pixels_gt = np.sum(ground_truth)
    num_pixels_pred = np.sum(prediction)

    # Skeletonizes the ground truth and the prediction
    # First crop the images as small as possible to speed up the computation
    skeleton_gt = skeletonize(ground_truth, method='lee')
    skeleton_pred = skeletonize(prediction, method='lee')
    lenght_gt = np.sum(skeleton_gt)
    length_pred = np.sum(skeleton_pred)

    # Computes the ratio of length
    length_ratio = np.exp(- 1.0 * np.abs(lenght_gt.astype(float) - length_pred) / lenght_gt) 

    # Computes the distance between the two skeletons
    if lenght_gt == 0 and length_pred == 0:
        line_distance_score = 1.
    elif lenght_gt == 0 or length_pred == 0:
        line_distance_score = 0.
    else:
        distances_gt = scipy.ndimage.distance_transform_edt(skeleton_gt == 0)
        distances_pred = scipy.ndimage.distance_transform_edt(skeleton_pred == 0)
        distance_factor_gt = np.sum(skeleton_pred * np.exp(-distances_gt ** 2 / (2 * sigma ** 2))) / lenght_gt
        distance_factor_pred = np.sum(skeleton_gt * np.exp(-distances_pred ** 2 / (2 * sigma ** 2))) / length_pred
        line_distance_score = distance_factor_gt * distance_factor_pred

    # Computes the width score
    # A simple approximation is simply the ratio of the number of pixels
    # To decorelate this from the length score
    width_ratio = num_pixels_gt * length_pred / (num_pixels_pred * lenght_gt)
    width_ratio = np.exp( - np.abs(1 - width_ratio))

    # Intersection and union
    intersection = np.sum(np.logical_and(ground_truth, prediction))
    union = np.sum(np.logical_or(ground_truth, prediction))
    iou = intersection / union
    dice = 2 * intersection / (intersection + union)

    coef_avg = [1, 2, 0.5, 0.5, 0.5]
    avg = np.sum([coef_avg[i] * x for i, x in enumerate([length_ratio, line_distance_score, width_ratio, iou, dice])]) / np.sum(coef_avg)

    results = {
        'length': length_ratio,
        'line_distance': line_distance_score,
        'width': width_ratio,
        'iou': iou,
        'dice': dice,
        'avg': avg
    }
    return results


def preprocess_for_comparison(
    ground_truth: Union[np.ndarray, Image.Image],
    prediction: Union[np.ndarray, Image.Image],
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
        ground_truth = apply_smooth_gaussian_diffusion(ground_truth, border_size=3*diffuse_sigma_factor, sigma=diffuse_sigma_factor)
        prediction = apply_smooth_gaussian_diffusion(prediction, border_size=3*diffuse_sigma_factor, sigma=diffuse_sigma_factor)

    return ground_truth, prediction


def to_np(image: Union[np.ndarray, Image.Image], dtype: np.dtype = np.float32) -> np.ndarray:
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
        image = image.detach().cpu().numpy()
    elif not isinstance(image, np.ndarray):
        raise ValueError(
            f"Expected image to be a PIL image, a numpy array or a PyTorch "
            f"tensor. Got {type(image)}."
        )

    if image.dtype == np.uint8 and np.max(image) > 1:
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