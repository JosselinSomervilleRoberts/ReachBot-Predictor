# This is the code that I used to copy the files from the original dataset to the new dataset.
# It takes the files from the original dataset and copies them to the new dataset, but it
# increments the file names by 20. This is useful for combining datasets.

import os
import shutil

# Replace these paths with your actual folder paths
folder_a_path = (
    "/media/jsomerviller/SSD2/ReachBot-Predictor/datasets/cracks_seg_full/ann_dir/train"
)
folder_b_path = "/media/jsomerviller/SSD2/ReachBot-Predictor/datasets/original_data/cracks_real/masks"
increment = 20

if not os.path.exists(folder_b_path):
    raise Exception(f"Folder {folder_b_path} does not exist")

# Iterate over files in folder A
for filename in os.listdir(folder_a_path):
    if filename.endswith(".png"):
        try:
            i = int(filename.split(".")[0])
            new_filename = f"{i + increment}.png"
            src_file = os.path.join(folder_a_path, filename)
            dst_file = os.path.join(folder_b_path, new_filename)
            shutil.copy(src_file, dst_file)
        except ValueError:
            pass  # Ignore files that don't match the naming pattern

print("Files copied successfully.")
