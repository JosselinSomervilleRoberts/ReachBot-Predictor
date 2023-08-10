import cv2
import glob
import numpy as np
import os
from PIL import Image
import torch
from tqdm import tqdm

from typing import Optional, List
from data_utils.utils import get_device, set_description


def load_images(
    class_name: str,
    train: bool,
    full_images: bool,
    n: int = -1,
    device: Optional[str] = None,
    switch_rgb: bool = True,
) -> list:
    if device is None:
        device = get_device()
    transformed_data = []

    mode: str = "train" if train else "val"
    mode2 = "full" if full_images else "cropped"
    directory_path = os.path.join(f"./datasets/{class_name}/{mode2}/{mode}")
    images_paths = sorted(glob.glob(os.path.join(directory_path, "images/*.png")))
    if len(images_paths) == 0:
        raise ValueError(f"No images found in {os.path.join(directory_path, 'images')}")
    if n > 0:
        images_paths = images_paths[:n]

    description = f"Loading {mode} images on {device}"
    with tqdm(enumerate(images_paths), total=len(images_paths)) as pbar:
        for k, img_path in pbar:
            set_description(pbar, description, k)
            image = cv2.imread(img_path)
            if switch_rgb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed_data.append(image)
    return transformed_data


def load_gt_masks(
    class_name: str,
    train: bool,
    full_images: bool,
    n: int = -1,
    device: Optional[str] = None,
) -> list:
    """Loads the ground truth masks from the FINETUNE_DATASET_FOLDER folder."""
    if device is None:
        device = get_device()
    ground_truth_masks = []

    mode: str = "train" if train else "val"
    mode2 = "full" if full_images else "cropped"
    directory_path = os.path.join(f"./datasets/{class_name}/{mode2}/{mode}")
    masks_paths = sorted(glob.glob(os.path.join(directory_path, "masks/*.png")))
    if len(masks_paths) == 0:
        raise ValueError(f"No masks found in {os.path.join(directory_path, 'masks')}")
    if n > 0:
        masks_paths = masks_paths[:n]

    description = f"Loading {mode} masks on {device}"
    with tqdm(enumerate(masks_paths), total=len(masks_paths)) as pbar:
        for k, mask_path in pbar:
            set_description(pbar, description, k)
            gt_grayscale = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = gt_grayscale == 0
            gt_mask_resized = torch.from_numpy(
                np.resize(gt_mask, (1, 1, gt_mask.shape[0], gt_mask.shape[1]))
            ).to(device)
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
            ground_truth_masks.append(gt_binary_mask)
    return ground_truth_masks


def load_palette_gt_masks(
    class_name: str,
    train: bool,
    full_images: bool,
    n: int = -1,
    device: Optional[str] = None,
    collapse: bool = False,
) -> List[dict]:
    """Loads the ground truth masks from the FINETUNE_DATASET_FOLDER folder."""
    if device is None:
        device = get_device()
    ground_truth_masks = []

    mode: str = "train" if train else "val"
    mode2 = "full" if full_images else "cropped"
    directory_path = os.path.join(f"./datasets/{class_name}/{mode2}/{mode}")
    masks_paths = sorted(glob.glob(os.path.join(directory_path, "masks/*.png")))
    if len(masks_paths) == 0:
        raise ValueError(f"No masks found in {os.path.join(directory_path, 'masks')}")
    if n > 0:
        masks_paths = masks_paths[:n]

    description = f"Loading {mode} masks on {device}"
    with tqdm(enumerate(masks_paths), total=len(masks_paths)) as pbar:
        for k, mask_path in pbar:
            set_description(pbar, description, k)

            mask = Image.open(mask_path)
            mask = np.array(mask)
            # Instances are encoded as different colors
            # The first instance is usually the background so remove it
            # Then for each channel, the instance is encoded as pixel values
            obj_ids = np.unique(mask)
            obj_ids = obj_ids[1:]  # First id is the background, so remove it

            # split the color-encoded mask into a set of binary masks
            masks = mask == obj_ids[:, None, None]

            if collapse:
                # Collapse the masks into one
                masks = np.max(masks, axis=0)

            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i]) if not collapse else np.where(masks)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

            # Convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            if collapse:
                masks = 1 - torch.as_tensor(masks, dtype=torch.float32)
            else:
                masks = torch.as_tensor(masks, dtype=torch.uint8)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            mask_infos = {}
            mask_infos["boxes"] = boxes
            mask_infos["masks"] = masks
            mask_infos["area"] = area
            ground_truth_masks.append(mask_infos)

    return ground_truth_masks
