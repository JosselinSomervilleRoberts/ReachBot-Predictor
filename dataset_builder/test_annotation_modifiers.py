from annotation_modifiers import (
    ShiftedAnnotationModifier,
    RandomWidthAnnotationModifier,
    RandomBranchRemovalModifier,
    CombinedAnnotationModifier,
    dilation,
)

import numpy as np
import pytest


def test_shifted_annotation_modifier():
    annotation = np.random.randint(0, 2, size=(100, 100, 1))
    modifier = ShiftedAnnotationModifier("test", (10, 10))
    modified_annotation = modifier(annotation)
    assert modified_annotation.shape == annotation.shape
    assert np.allclose(
        annotation[:-10, :-10], modified_annotation[10:, 10:], equal_nan=True
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dilate(device):
    input_array = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 2, 1, 0], [0, 0, 0, 0]])
    output_array = dilation(input_array, device=device)
    assert np.allclose(
        output_array, np.array([[0, 1, 1, 1], [1, 1, 2, 1], [1, 2, 1, 1], [1, 1, 1, 0]])
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_random_width_annotation_modifier(device):
    annotation = np.zeros((10, 10, 1))
    annotation[4:7, 2:9] = 1

    # No change (still gets modified due to skeletonization)
    modifier = RandomWidthAnnotationModifier("test", (1.0, 1.0), device=device)
    modified_annotation = modifier(annotation, visualize=True)
    expected_annotation = np.zeros((10, 10, 1))
    expected_annotation[4:7, 3:8] = 1
    assert modified_annotation.shape == annotation.shape
    assert np.allclose(modified_annotation, expected_annotation, equal_nan=True)

    # Width divided by 2
    modifier_2 = RandomWidthAnnotationModifier("test", (0.5, 0.5), device=device)
    modified_annotation = modifier_2(annotation, visualize=True)
    expected_annotation = np.zeros((10, 10, 1))
    expected_annotation[5:6, 4:7] = 1
    assert modified_annotation.shape == annotation.shape
    assert np.allclose(modified_annotation, expected_annotation, equal_nan=True)


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    annotation = cv2.imread("drive.png", cv2.IMREAD_GRAYSCALE)
    if np.max(annotation) > 1:
        annotation = annotation / 255

    choice = -1
    modifiers = [
        ShiftedAnnotationModifier("ShiftedAnnotationModifier", shift_range=(-50, 50)),
        RandomWidthAnnotationModifier("RandomWidthAnnotationModifier", (0.1, 2.0)),
        RandomBranchRemovalModifier(
            "RandomBranchRemovalModifier", prob_removal=0.35, selectiveness=0.2
        ),
    ]
    combined = CombinedAnnotationModifier(
        "CombinedAnnotationModifier",
        modifiers,
        prob=0.5,
    )
    modifiers.append(combined)

    while choice <= 0 or choice > len(modifiers):
        print("Choose a modifier:")
        for i, modifier in enumerate(modifiers):
            print(f"{i+1}: {modifier.name}")
        try:
            choice = int(input())
        except ValueError:
            print("Invalid choice")

    modifier = modifiers[choice - 1]

    n_cols = 3
    n_rows = 3
    plt.figure(figsize=(n_cols * 5, n_rows * 5))
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(annotation.squeeze())
    plt.title("Original")
    for i in tqdm(range(n_cols * n_rows - 1)):
        modified = modifier(annotation)
        plt.subplot(n_rows, n_cols, i + 2)
        plt.imshow(modified.squeeze())
        plt.title(f"Modified {i+1}")
    plt.savefig("debug_grid.png")
