import torch


def iou_loss(ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """
    Computes the Intersection over Union (IoU) loss between a ground truth
    segmentation and a predicted segmentation.
    Both the ground truth and the prediction must be binary images.

    The intersection is defined as the sum of the probabilities of the pixels
    that are correctly classified.

    The union is defined as the sum of the probabilities of the entire prediction.

    Args:
        ground_truth: The ground truth segmentation.
        prediction: The predicted segmentation.

    Returns:
        The IoU loss.
    """
    intersection = torch.minimum(ground_truth * prediction)
    union = torch.maximum(prediction)
    if union == 0:
        return 1
    return 1 - intersection / union


def dice_loss(ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """
    Computes the Dice loss between a ground truth segmentation and a predicted
    segmentation.
    Both the ground truth and the prediction must be binary images.

    Args:
        ground_truth: The ground truth segmentation.
        prediction: The predicted segmentation.

    Returns:
        The Dice loss.
    """
    intersection = torch.minimum(ground_truth * prediction)
    union = torch.maximum(prediction)
    if union == 0:
        return 1
    return 1 - 2 * intersection / (union + intersection)