from torch.utils.data import Dataset
import torch
from typing import Optional
import cv2


from data_utils.utils import get_device
from data_utils.load_data import load_images, load_gt_masks, load_palette_gt_masks


class FullImageDataset(Dataset):
    def __init__(self, class_name: str, train: bool, n: int = -1, device: Optional[None] = None, transform=None, mask_transform=None, transforms=None):
        self.transform = transform
        self.mask_transform = mask_transform
        self.transforms = transforms

        self._train = train
        self._class_name = class_name
        self._n = n
        self._device = device if device is not None else get_device()

        # Load all image files, sorting them to ensure that they are aligned
        self.imgs = load_images(self._class_name, self._train, full_images=True, n=self._n, device=self._device)
        self.masks_data = load_palette_gt_masks(self._class_name, self._train, full_images=True, n=self._n, device=self._device)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        mask_data = self.masks_data[idx]
        masks = mask_data["mask"]
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
        for i in range(boxes.shape[0]):
            box = boxes[i]
            mask = masks[i]

            x1, y1, x2, y2 = box
            cropped_image = image[y1:y2, x1:x2]
            cropped_mask = mask[y1:y2, x1:x2]

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

        self.imgs = load_images(self._class_name, self._train, self._n, self._device)
        self.gt_masks = load_gt_masks(self._class_name, self._train, self._n, self._device)

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