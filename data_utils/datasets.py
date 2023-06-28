from torch.utils.data import Dataset
import torch
from typing import Optional
import random
from skimage.morphology import label
import numpy as np
from PIL import Image


from data_utils.utils import get_device
from data_utils.load_data import load_images, load_gt_masks, load_palette_gt_masks


class FullImageDataset(Dataset):
    def __init__(self, class_name: str, train: bool, n: int = -1, device: Optional[None] = None, collapse: bool = False, transform=None, mask_transform=None, transforms=None):
        self.transform = transform
        self.mask_transform = mask_transform
        self.transforms = transforms

        self._train = train
        self._class_name = class_name
        self._n = n
        self._device = device if device is not None else get_device()
        self._collapse = collapse

        # Load all image files, sorting them to ensure that they are aligned
        self.imgs = load_images(class_name=self._class_name, train=self._train, full_images=True, n=self._n, device=self._device)
        self.masks_data = load_palette_gt_masks(class_name=self._class_name, train=self._train, full_images=True, n=self._n, device=self._device, collapse=collapse)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        original_image_size = image.shape[:2]
        mask_data = self.masks_data[idx]
        masks = mask_data["masks"]
        boxes = mask_data["boxes"]
        area = mask_data["area"]

        # Additional infos for MakRCNN
        num_objs = masks.shape[0]
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        if self.transform:
            image = self.transform.apply_image(image)
        if self.mask_transform:
            masks = self.mask_transform(masks)

        if self._collapse:
            return image, masks, original_image_size

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.imgs)
    
    def crop(self, image, target) -> list:
        boxes = target["boxes"]
        masks = target["masks"]

        # For each box, crop the image, the mask and return a list of dict
        # containing the cropped images and masks.
        # boxes and masks are torch tensors
        # image is a cv2 image

        result = []
        for i in range(boxes.shape[1]):
            box = boxes[0][i]
            mask = masks[0][i]

            separated_mask = label(mask)
            for i in np.unique(separated_mask):
                if i == 0:  # background
                    continue
                blob = (separated_mask == i).astype(int)
                nb_pixels = np.sum(blob)
                if nb_pixels < 32:
                    continue
                # Get the bounding box
                bbox = np.array((Image.fromarray(blob.astype(np.uint8) * 255)).getbbox())

                x1, y1, x2, y2 = bbox
                # Add some randomness to the coordinates
                x1 = max(0, x1 -random.randint(0, 20), 0)
                y1 = max(0, y1 - random.randint(0, 20), 0)
                x2 = min(image.shape[2], x2 + random.randint(0, 20))
                y2 = min(image.shape[1], y2 + random.randint(0, 20))
                cropped_image = image[0,y1:y2, x1:x2]
                cropped_mask = blob[y1:y2, x1:x2]

                result.append({
                    "image": cropped_image,
                    "mask": cropped_mask
                })

        return result




class CroppedImageDataset(Dataset):

    def __init__(self, class_name: str, train: bool, n: int = -1, device: Optional[None] = None, transform=None, mask_transform=None):
        self.transform = transform
        self.mask_transform = mask_transform

        self._train = train
        self._class_name = class_name
        self._n = n
        self._device = device if device is not None else get_device()

        self.imgs = load_images(class_name=self._class_name, full_images=False, train=self._train, n=self._n, device=self._device)
        self.gt_masks = load_gt_masks(class_name=self._class_name, full_images=False, train=self._train, n=self._n, device=self._device)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        gt_mask = self.gt_masks[idx]
        original_image_size = image.shape[:2]

        if self.transform:
            image = self.transform.apply_image(image)
        if self.mask_transform:
            gt_mask = self.mask_transform(gt_mask)

        return image, gt_mask, original_image_size