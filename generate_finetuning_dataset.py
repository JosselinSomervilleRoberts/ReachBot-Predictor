import os
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw

CLASSES = ["edge", "boulder", "crack"]
DATASET_FOLDER = "./dataset"
OUTPUT_FOLDER = "./finetuning_dataset"

def mask_to_bbox(mask):
    # Convert the mask to a binary mask
    mask = mask.convert("L")
    mask = mask.point(lambda x: 0 if x < 128 else 255, '1')
    mask = mask.convert("1")

    # Get the bounding box
    bbox = mask.getbbox()

    # Add some noise and padding to the bounding box
    bbox = (bbox[0] - 10, bbox[1] - 10, bbox[2] + 10, bbox[3] + 10)

    # Convert the bounding box to a binary mask
    bbox_mask = Image.new("1", mask.size)

    # Draw the bounding box
    bbox_mask_draw = ImageDraw.Draw(bbox_mask)
    bbox_mask_draw.rectangle(bbox, fill=1)

    return bbox_mask

# Check if the dataset folder exists
if not os.path.exists(DATASET_FOLDER):
    raise Exception("Dataset folder not found. Please run generate_dataset.py first.")

# Get the number of images in the dataset
num_images = len(os.listdir(DATASET_FOLDER))
print(f"Found {num_images} images")

# Create the output folder
if not os.path.exists("finetuning_dataset"):
    os.makedirs("finetuning_dataset")
if not os.path.exists("finetuning_dataset/images"):
    os.makedirs("finetuning_dataset/images")
if not os.path.exists("finetuning_dataset/masks"):
    os.makedirs("finetuning_dataset/masks")
if not os.path.exists("finetuning_dataset/bboxes"):
    os.makedirs("finetuning_dataset/bboxes")

# For each image folder, open classes.txt and get the classes
# For each line (i.e. mask) in classes.txt, load the mask if the class is in CLASSES
# Then generate a bounding box for the mask and save it in the output folder
# The bounding box is saved as a binary mask
# It if generated with some noise and padding

# The final architecture of the dataset folder is:
# finetuning_dataset
# ├── images
# │   ├── 0.png
# │   ├── 1.png
# │   ├── ...
# │   └── N.png
# │── masks
# │   ├── 0.png
# │   ├── 1.png
# │   ├── ...
# │   └── N.png
# │── bboxes
# │   ├── 0.png
# │   ├── 1.png
# │   ├── ...
# │   └── N.png

num_masks_added = 0
num_masks_failed = 0

for i in tqdm(range(num_images)):
    img_folder = os.path.join(DATASET_FOLDER, str(i))
    classes_file = os.path.join(img_folder, "classes.txt")

    # Get the classes
    with open(classes_file, "r") as f:
        classes = f.read().splitlines()

    # Get the masks
    for j, class_mask in enumerate(classes):
        if class_mask in CLASSES:
            try:
                # Load the mask
                mask_path = os.path.join(img_folder, "mask_" + str(j) + ".png")
                mask = Image.open(mask_path)

                # Generate the bounding box
                bbox = mask_to_bbox(mask)

                # Save the mask
                mask.save(os.path.join(OUTPUT_FOLDER, "masks", str(num_masks_added) + ".png"))

                # Copy the image
                img_path = os.path.join(img_folder, "image.png")
                img = Image.open(img_path)
                img.save(os.path.join(OUTPUT_FOLDER, "images", str(num_masks_added) + ".png"))

                # Save the bbox
                bbox.save(os.path.join(OUTPUT_FOLDER, "bboxes", str(num_masks_added) + ".png"))

                num_masks_added += 1
            except Exception as e:
                num_masks_failed += 1
                print(f"Error processing image {i} mask {j}: {e}")
                continue

print(f"Generated {num_masks_added} masks")
print(f"Failed to generate {num_masks_failed} masks")