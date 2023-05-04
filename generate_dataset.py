import labelbox
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Get the labelbox project
LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGg0NHRyODEwNnFpMDcxMWU1dWdjM2Q3Iiwib3JnYW5pemF0aW9uSWQiOiJjbGg0NHRyN3AwNnFoMDcxMWNtbGM1Z29lIiwiYXBpS2V5SWQiOiJjbGg4MmhzYmIwY2N1MDcwamM0eHY4Z29iIiwic2VjcmV0IjoiNTM2MDI0NDFmMDY4OGM2ZTIyZmJhMjFjZWQxYzk2MWQiLCJpYXQiOjE2ODMxNDA2NjQsImV4cCI6MjMxNDI5MjY2NH0.AfCKYZ-Bc7xA5i3ybMJIaLIHB8NB7_NXt41vcZH7Z-8"
lb = labelbox.Client(api_key=LB_API_KEY)
project = lb.get_project('clh44xst90kru07zfdq3qdf72')

print("Downloading labels...")
labels = project.label_generator()
labels = project.export_labels(download = True)
print("Done")

# Create folder name "dataset" if it doesn't exist
import os
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Download all images and save them in the "dataset" folder
print("Downloading images...")
for i in tqdm(range(len(labels))):
    x = labels[i]['Labeled Data']
    response = requests.get(x)
    img = Image.open(BytesIO(response.content))

    # Create folder named i if it doesn't exist
    if not os.path.exists("dataset/" + str(i)):
        os.makedirs("dataset/" + str(i))
    else:
        print(f"Image {i} already exists")
        continue

    # Save image
    img.save("dataset/" + str(i) + "/image.png")

    # Get all masks
    classes = []
    objects = labels[i]['Label']['objects']
    for j in range(len(objects)):
        x = objects[j]['instanceURI']
        response = requests.get(x)
        img = Image.open(BytesIO(response.content))
        img.save("dataset/" + str(i) + "/mask_" + str(j) + ".png")
        classes.append(objects[j]['title'])

    # Save classes
    with open("dataset/" + str(i) + "/classes.txt", "w") as f:
        f.write("\n".join(classes))

print("Done")