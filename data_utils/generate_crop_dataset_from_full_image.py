from torch.data.utils import DataLoader
from tqdm import tqdm
import os
import cv2
import numpy as np

from data_utils.datasets import FullImageDataset, CropDataset
from data_utils.utils import set_description

class_name: str = "boulders"
train: bool = True

mode = "train" if train else "val"
full_image_dataset = FullImageDataset(class_name=class_name, train=train)
data_loader = DataLoader(full_image_dataset, batch_size=1)

image_idx = 0
with tqdm(enumerate(data_loader), total=len(data_loader)) as pbar:
    for i, data in pbar:
        set_description(pbar, f"Generating crop dataset", i, frequency=1)
        image, mask = data
        list_of_crops = full_image_dataset.crop(image, mask)
        for crop in list_of_crops:
            image = crop["image"] # cv2 image
            mask = crop["mask"] # torch tensor

            # Save both as png images
            directory_path = f"./datasets/{class_name}/cropped/{mode}"
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            image_path = os.path.join(directory_path, f"images/{image_idx}.png")
            mask_path = os.path.join(directory_path, f"masks/{image_idx}.png")
            cv2.imwrite(image_path, image)

            # Convert mask to grayscale
            mask = mask.squeeze().cpu().numpy()
            mask = mask.astype(np.uint8) * 255
            cv2.imwrite(mask_path, mask)
            image_idx += 1
