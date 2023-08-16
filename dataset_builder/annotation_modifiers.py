# This file defines some Annotation modifiers that modifies the annotation
# to check how robust the model is to the perturbation of the annotation.

import numpy as np
import torch

from typing import Tuple
from skimage.morphology import skeletonize
import scipy.ndimage
from perlin import FractalPerlin2D


def dilation(input_array: torch.Tensor):
    """Given a tensor, dilates the tensor by at each step setting the value of
    pixels equal to 0 to max(val_pixels_around_it) - 1. This is repeated until
    the tensor does not change anymore.
    """
    num_iter = int(torch.max(input_array) - 1)

    output_array = input_array.clone().unsqueeze(0)
    for _ in range(num_iter):
        # perform max pooling
        # max_1 = torch.nn.functional.max_pool2d(
        #     output_array, kernel_size=(1, 3), stride=1, padding=(0, 1)
        # )
        # max_2 = torch.nn.functional.max_pool2d(
        #     output_array, kernel_size=(3, 1), stride=1, padding=(1, 0)
        # )
        # max_arr = torch.max(max_1, max_2) - 1
        max_arr = (
            torch.nn.functional.max_pool2d(
                output_array, kernel_size=(3, 3), stride=1, padding=(1, 1)
            )
            - 1
        )
        new_output_array = torch.max(output_array, max_arr)
        if torch.all(torch.eq(output_array, new_output_array)):
            break
        output_array = new_output_array

    return output_array.squeeze(0).cpu().numpy()


class AnnotationModifier(object):
    def __init__(self, name: str):
        self.name = name

    def __call__(self, annotation: np.ndarray, visualize: bool = False):
        raise NotImplementedError


class ShiftedAnnotationModifier(AnnotationModifier):
    """Randomly translates the annotation."""

    def __init__(self, name: str, shift_range: Tuple[int, int]):
        super().__init__(name)
        self.shift_range = shift_range

    def __call__(self, annotation: np.ndarray, visualize: bool = False):
        modified_annotation = np.zeros_like(annotation)
        # Include both ends of the range
        shift_x = np.random.randint(self.shift_range[0], self.shift_range[1] + 1)
        shift_y = np.random.randint(self.shift_range[0], self.shift_range[1] + 1)
        modified_annotation[
            max(0, shift_y) : min(annotation.shape[0], annotation.shape[0] + shift_y),
            max(0, shift_x) : min(annotation.shape[1], annotation.shape[1] + shift_x),
        ] = annotation[
            max(0, -shift_y) : min(annotation.shape[0], annotation.shape[0] - shift_y),
            max(0, -shift_x) : min(annotation.shape[1], annotation.shape[1] - shift_x),
        ]
        return modified_annotation


class RandomWidthAnnotationModifier(AnnotationModifier):
    """Randomly changes the width of the annotation.

    This is done by skeletonizing the annotation, computing the width of each pixel
    of the skeleton, and modifying this width with some random perlin noise.
    """

    def __init__(
        self, name: str, distortion_range: Tuple[float, float], device: str = "cpu"
    ):
        """Constructor.

        Args:
            distortion_range: Tuple of floats between 0 and infinity,
                The width witll be multiplied by a random number between
                distortion_range[0] and distortion_range[1]. (Must be positive))
        """
        super().__init__(name)
        self.distortion_range = distortion_range
        self.device = device

    def __call__(self, annotation: np.ndarray, visualize: bool = False):
        skeleton = skeletonize(annotation, method="lee")
        distance_to_sk = scipy.ndimage.distance_transform_edt(annotation != 0)
        shape = distance_to_sk.squeeze().shape
        gen = FractalPerlin2D(
            shape=(1, *shape),
            resolution=(3, 3),
            persistence=0.5,
            lacunarity=2,
            octaves=4,
            device=self.device,
        )
        noise = gen().reshape(distance_to_sk.shape)  # Between 0 and 1
        # noise = np.random.rand(*shape).reshape(distance_to_sk.shape)
        # Scale noise be in the distortion range
        noise = self.distortion_range[0] + noise * (
            self.distortion_range[1] - self.distortion_range[0]
        )
        distance_to_sk_noisy = torch.Tensor(distance_to_sk) * noise
        distance_to_sk_noisy[skeleton == 0] = 0
        # distance_to_sk_noisy = torch.round(torch.sqrt(distance_to_sk_noisy))

        # Now we need to dilate the skeleton by a value of distance_to_sk
        dilated_distance = dilation(distance_to_sk_noisy.squeeze()).reshape(
            annotation.shape
        )
        new_annotation = dilated_distance > 0
        new_annotation = new_annotation.astype(np.uint8)

        if visualize:
            import matplotlib.pyplot as plt

            plt.subplot(3, 2, 1)
            plt.imshow(annotation.squeeze())
            plt.title("Original")
            plt.subplot(3, 2, 2)
            plt.imshow(skeleton.squeeze())
            plt.title("Skeleton")
            plt.subplot(3, 2, 3)
            plt.imshow(distance_to_sk.squeeze())
            plt.colorbar()
            plt.title("Distance to skeleton")
            plt.subplot(3, 2, 4)
            plt.imshow(distance_to_sk_noisy.squeeze())
            plt.title("Distance to skeleton with noise")
            plt.colorbar()
            plt.subplot(3, 2, 5)
            plt.imshow(dilated_distance.squeeze())
            plt.title("Dilated")
            plt.colorbar()
            plt.subplot(3, 2, 6)
            plt.imshow(new_annotation.squeeze())
            plt.title("Dilated as uint8")
            plt.savefig("debug.png")
            print("Saved debug.png")

        return new_annotation


class RandomBranchRemovalModifier(AnnotationModifier):
    """Randomly removes branches from the annotation.

    This is done by skeletonizing the annotation, and removing some branches
    randomly.
    """

    def __init__(
        self,
        name: str,
        prob_removal: float = 0.5,
        selectiveness: float = 1.0,
        device: str = "cpu",
    ):
        """Constructor.

        Args:
            prob_removal: Probability of removing a branch.
            selectiveness: Between 0 and 1. The higher it is, the more likely
                it is to remove branches that are thin, the smaller it is, the less
                the thickness of the branch matters.
        """
        super().__init__(name)
        self.prob_removal = prob_removal
        self.selectiveness = selectiveness
        self.device = device

    def __call__(self, annotation: np.ndarray, visualize: bool = False):
        skeleton = skeletonize(annotation, method="lee")
        distance_to_sk = scipy.ndimage.distance_transform_edt(annotation != 0)
        shape = distance_to_sk.squeeze().shape
        gen = FractalPerlin2D(
            shape=(1, *shape),
            resolution=(3, 3),
            persistence=0.5,
            lacunarity=2,
            octaves=4,
            device=self.device,
        )
        noise = gen().reshape(distance_to_sk.shape)  # Between 0 and 1
        distance_skeleton_mean = np.mean(distance_to_sk[skeleton != 0])
        noise *= (
            distance_skeleton_mean / (0.001 + torch.Tensor(distance_to_sk))
        ) ** self.selectiveness
        # noise = np.random.rand(*shape).reshape(distance_to_sk.shape)
        skeleton_cut = skeleton.copy()
        skeleton_cut[noise > 1 - self.prob_removal] = 0
        distance_to_sk_cut = torch.Tensor(distance_to_sk)
        distance_to_sk_cut[skeleton_cut == 0] = 0

        # Now we need to dilate the skeleton by a value of distance_to_sk
        dilated_distance = dilation(distance_to_sk_cut.squeeze()).reshape(
            annotation.shape
        )
        new_annotation = dilated_distance > 0
        new_annotation = new_annotation.astype(np.uint8)

        if visualize:
            import matplotlib.pyplot as plt

            plt.subplot(3, 2, 1)
            plt.imshow(annotation.squeeze())
            plt.title("Original")
            plt.subplot(3, 2, 2)
            plt.imshow(skeleton.squeeze())
            plt.title("Skeleton")
            plt.subplot(3, 2, 3)
            plt.imshow(distance_to_sk.squeeze())
            plt.colorbar()
            plt.title("Distance to skeleton")
            plt.subplot(3, 2, 4)
            plt.imshow(skeleton_cut.squeeze())
            plt.title("Skeleton Cut")
            plt.colorbar()
            plt.subplot(3, 2, 5)
            plt.imshow(dilated_distance.squeeze())
            plt.title("Dilated")
            plt.colorbar()
            plt.subplot(3, 2, 6)
            plt.imshow(new_annotation.squeeze())
            plt.title("Dilated as uint8")
            plt.savefig("debug.png")
            print("Saved debug.png")

        return new_annotation


class CombinedAnnotationModifier(AnnotationModifier):
    """Combines multiple annotation modifiers."""

    def __init__(self, name: str, annotation_modifiers: list, prob: float = 0.5):
        super().__init__(name)
        self.annotation_modifiers = annotation_modifiers
        self.prob = prob

    def __call__(self, annotation: np.ndarray, visualize: bool = False):
        for modifier in self.annotation_modifiers:
            if np.random.rand() < self.prob:
                annotation = modifier(annotation, visualize=visualize)
        return annotation
