from typing import Tuple
import torch
import torch.nn.functional as F


# ========== Original soft skeletonization proposed in CLDice ========== #


def minpool(
    img: torch.Tensor,
    size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> torch.Tensor:
    """Minpooling function.

    Args:
        img: Input image.
        **kwargs: Keyword arguments for F.max_pool2d.

    Returns:
        Minpooled image.
    """
    return -F.max_pool2d(-img, size, stride, padding)


# Source: https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py
def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Soft erosion function.

    This function is a soft approximation of the erosion operator. It is
    defined as a minpooling between a pixel and its direct neighbors (cross pattern).
    It is a differentiable approximation of the erosion operator.

    Args:
        img: Input image.

    Returns:
        Eroded image.
    """
    p1 = minpool(img, (3, 1), (1, 1), (1, 0))
    p2 = minpool(img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


# Source: https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py
def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """Soft dilation function.

    This function is a soft approximation of the dilation operator. It is
    defined as a maxpooling between a pixel and its direct neighbors (cross pattern).
    It is a differentiable approximation of the dilation operator.

    Args:
        img: Input image.

    Returns:
        Dilated image.
    """
    p1 = F.max_pool2d(img, (3, 1), (1, 1), (1, 0))
    p2 = F.max_pool2d(img, (1, 3), (1, 1), (0, 1))
    return torch.max(p1, p2)


# Source: https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py
def soft_open(img: torch.Tensor) -> torch.Tensor:
    """Soft opening function.

    Args:
        img: Input image.

    Returns:
        Opened image.
    """
    return soft_dilate(soft_erode(img))


# Source: https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py
def soft_skeletonize(img: torch.Tensor, iter_: int = 10) -> torch.Tensor:
    """Soft skeletonization function.

    This function is a soft approximation of the morphological skeletonization operator.
    It is defined as the successive application of soft erosion and soft opening.
    The number of iterations should be greater than the largest diameter of the objects
    in the image.

    Args:
        img: Input image.
        iter_: Number of iterations.

    Returns:
        Skeletonized image.
    """
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


# ====================================================================== #


# ========== Soft skeletonization with a 2x2 kernel for thiner results ========== #


def max_minpool_thin(T: torch.Tensor, left: bool = True) -> torch.Tensor:
    """Max-minpooling thinning function.

    This version uses a kernel of size 2x2 instead of 3x3.
    This is technically not possible due to padding issues, but we can
    circumvent this by using a maxpooling of size 2x2 with stride 1 and
    padding 1, followed by a minpooling of size 2x2 with stride 1 and
    padding 0 (for left thinning) or 1 (for right thinning).
    This introduces a bias towards the left or right, but this is not
    a problem for our application.

    Args:
        T: Input image.
        left: Whether to use left or right thinning.

    Returns:
        Thinned image.
    """
    T = -torch.nn.MaxPool2d(2, stride=1, padding=int(left))(-T)
    T = torch.nn.MaxPool2d(2, stride=1, padding=1 - int(left))(T)
    return T


def maxpool_thin(T: torch.Tensor, left: bool = True) -> torch.Tensor:
    """Maxpooling thinning function.

    This version uses a kernel of size 2x2 instead of 3x3.
    Similarly to max_minpool_thin, ww crop the image to the correct size.
    This introduces a bias towards the left or right, but this is not
    a problem for our application.

    Args:
        T: Input image.
        left: Whether to use left or right thinning.

    Returns:
        Thinned image.
    """
    # T is of shape (H, W)
    T = torch.nn.MaxPool2d(2, stride=1, padding=1)(T)
    # T is now of shape (H+1, W+1)
    if left:
        T = T[:, :-1, :-1]
    else:
        T = T[:, 1:, 1:]
    return T


def soft_skeletonize_thin(I: torch.Tensor, k: int) -> torch.Tensor:
    """Soft skeletonization.

    Thinner version of soft_skeletonize.
    This version uses a kernel of size 2x2 instead of 3x3.
    It introduces a bias towards the left or right, but this is not
    a problem for our application.

    Args:
        I: Input image.
        k: Number of iterations.

    Returns:
        Skeletonized image.
    """
    relu = torch.nn.ReLU()
    Ip = max_minpool_thin(I, left=True)
    S = relu(I - Ip)
    for i in range(k):
        I = -maxpool_thin(-I, left=(i % 2 == 0))
        Ip = max_minpool_thin(I, left=(i % 2 == 0))
        S += (1 - S) * relu(I - Ip)
    return S


# =============================================================================== #
