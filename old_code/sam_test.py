from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann["segmentation"]
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


# Load the image
print("Loading image...")
image = cv2.imread("./test.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(20,20))
# plt.imshow(image)
# plt.axis('off')
# plt.show()
print("Image loaded")

# Load the model
print("\nLoading SAM model...")
# sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")
# sam = sam_model_registry["vit_l"](checkpoint="./sam_vit_l_0b3195.pth")
sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam.to(device)
print("SAM model loaded")
print("Using device:", device)

# Generate masks
print("\nGenerating masks...")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
print(len(masks))
print(masks[0].keys())
plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis("off")
plt.savefig("./test_masks_lq.png")
print("Masks generated")

predictor = SamPredictor(sam)
predictor.set_image(image)
help(predictor.predict)
masks, _, _ = predictor.predict("boulder")

# Save the masks in an image
print("\nSaving masks...")
plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis("off")
plt.savefig("./test_masks_2.png")
print("Masks saved")
