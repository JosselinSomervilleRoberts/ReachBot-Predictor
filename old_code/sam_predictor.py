from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
from torch.nn.functional import threshold, normalize

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

import labelbox

# Enter your Labelbox API key here
LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGg0NHRyODEwNnFpMDcxMWU1dWdjM2Q3Iiwib3JnYW5pemF0aW9uSWQiOiJjbGg0NHRyN3AwNnFoMDcxMWNtbGM1Z29lIiwiYXBpS2V5SWQiOiJjbGg4MmhzYmIwY2N1MDcwamM0eHY4Z29iIiwic2VjcmV0IjoiNTM2MDI0NDFmMDY4OGM2ZTIyZmJhMjFjZWQxYzk2MWQiLCJpYXQiOjE2ODMxNDA2NjQsImV4cCI6MjMxNDI5MjY2NH0.AfCKYZ-Bc7xA5i3ybMJIaLIHB8NB7_NXt41vcZH7Z-8"
# Create Labelbox client
lb = labelbox.Client(api_key=LB_API_KEY)
# Get project by ID
project = lb.get_project("clh44xst90kru07zfdq3qdf72")
# Export image and text data as an annotation generator:
labels = project.label_generator()
# Export all labels as a json file:
labels = project.export_labels(download=True)
print(labels[0]["External ID"])

# Download all masks and displays thems
import requests
from PIL import Image
from io import BytesIO

img = None
objects = labels[0]["Label"]["objects"]
for i in range(len(objects)):
    x = objects[i]["instanceURI"]
    response = requests.get(x)

    if img is None:
        img = Image.open(BytesIO(response.content))
    else:
        # Draw on top of the original image
        img2 = Image.open(BytesIO(response.content))
        img.paste(img2, (0, 0), img2)
# Set img to gt_mask
gt_mask = img
gt_mask = cv2.cvtColor(np.array(gt_mask), cv2.COLOR_RGB2GRAY)
gt_mask = gt_mask / 255.0
gt_binary_mask = torch.from_numpy(gt_mask).to(device)
gt_mask = gt_binary_mask
print("Ground truth mask loaded")


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
original_image_size = image.shape[:2]
# plt.figure(figsize=(20,20))
# plt.imshow(image)
# plt.axis('off')
# plt.show()
print("Image loaded")

# Load the model
print("\nLoading SAM model...")
sam_model = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")
# sam = sam_model_registry["vit_l"](checkpoint="./sam_vit_l_0b3195.pth")
# sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
sam_model = sam_model.to(device)
print("SAM model loaded")
print("Using device:", device)

optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters())
loss_fn = torch.nn.MSELoss()

# Load the ground truth mask
print("\nLoading ground truth mask...")
gt_mask = cv2.imread("./test_mask.png")
gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
gt_mask = gt_mask / 255.0
gt_binary_mask = torch.from_numpy(gt_mask).to(device)
print("Ground truth mask loaded")

input_image = image
input_size = input_image.shape[:2]
box_torch = torch.tensor([[0, 0, input_size[1], input_size[0]]]).to(device)
with torch.no_grad():
    image_embedding = sam_model.image_encoder(input_image)
with torch.no_grad():
    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
low_res_masks, iou_predictions = sam_model.mask_decoder(
    image_embeddings=image_embedding,
    image_pe=sam_model.prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=False,
)
upscaled_masks = sam_model.postprocess_masks(
    low_res_masks, input_size, original_image_size
).to(device)
binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(device)

loss = loss_fn(binary_mask, gt_binary_mask)
print("Loss:", loss.item())

# Plot the masks
plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns([{"segmentation": gt_mask}])
plt.axis("off")
plt.savefig("./test_masks_gt.png")

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns([{"segmentation": binary_mask.cpu().numpy()}])
plt.axis("off")
plt.savefig("./test_masks_pred.png")


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
plt.savefig("./test_masks_hq.png")
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
