"""
This file contains all the metrics used to evaluate our segmentation models.
"""


def iou_score(intersection: float, union: float) -> float:
    """
    Computes the Intersection over Union (IoU) score between a ground truth
    segmentation and a predicted segmentation.

    Args:
        intersection: The intersection between the ground truth and the prediction.
        union: The union between the ground truth and the prediction.

    Returns:
        The IoU score.
    """
    return intersection / union


def dice_score(intersection: float, union: float) -> float:
    """
    Computes the Dice score between a ground truth segmentation and a predicted
    segmentation.

    Args:
        intersection: The intersection between the ground truth and the prediction.
        union: The union between the ground truth and the prediction.

    Returns:
        The Dice score.
    """
    return 2 * intersection / (union + intersection)

