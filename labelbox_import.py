# install latest labelbox version (3.0 or above)
# !pip3 install labelbox[data]

import labelbox
# Enter your Labelbox API key here
LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGg0NHRyODEwNnFpMDcxMWU1dWdjM2Q3Iiwib3JnYW5pemF0aW9uSWQiOiJjbGg0NHRyN3AwNnFoMDcxMWNtbGM1Z29lIiwiYXBpS2V5SWQiOiJjbGg4MmhzYmIwY2N1MDcwamM0eHY4Z29iIiwic2VjcmV0IjoiNTM2MDI0NDFmMDY4OGM2ZTIyZmJhMjFjZWQxYzk2MWQiLCJpYXQiOjE2ODMxNDA2NjQsImV4cCI6MjMxNDI5MjY2NH0.AfCKYZ-Bc7xA5i3ybMJIaLIHB8NB7_NXt41vcZH7Z-8"
# Create Labelbox client
lb = labelbox.Client(api_key=LB_API_KEY)
# Get project by ID
project = lb.get_project('clh44xst90kru07zfdq3qdf72')
# Export image and text data as an annotation generator:
labels = project.label_generator()
# Export all labels as a json file:
labels = project.export_labels(download = True)
print(labels[0]["External ID"])

# Download all masks and displays thems
import requests
from PIL import Image
from io import BytesIO
img = None
objects = labels[0]['Label']['objects']
for i in range(len(objects)):
    x = objects[i]['instanceURI']
    response = requests.get(x)

    if img is None:
        img = Image.open(BytesIO(response.content))
    else:
        # Draw on top of the original image
        img2 = Image.open(BytesIO(response.content))
        img.paste(img2, (0, 0), img2)
img.show()