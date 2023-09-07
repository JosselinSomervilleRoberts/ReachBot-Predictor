# Scripts

This folder contains all the scripts used in the paper. For an example of their usage, check the repo's [README](../README.md) to see for which figure they have been used.

The folder `example_outputs` shows a few outputs obtained with them.

The rest of this README explains how to build the dataset fron scratch.


## Dataset builder

This folder contains all the files necessary to download and preprocess the data for various applications:
* Multi-class finetuning
* Data-augmentation by mask direction following
* Single class binary classification
* Self-supervised learnin

This code assumes that you use **Labelbox**. If you don't that is fine, as long as you provide an original dataset under the correct format:
* Create a folder containing you dataset and set in `config.ini` the path to your dataset under `LABELBOX_DATASET`.
* Your dataset should contain folders name `{{i}}` where `i` is the index of the image *(So if you have 5 images, your dataset folder should contain 5 folder: "0", "1", "2", "3", "4").
* Then each image folder should contain:
    * `image.png`: the image
    * `classes.txt`: a file where the $i+1$-th line defines the class of your $i$-th mask.
    * `mask_{{i}}.png`: an image representing your $i$-th mask. The background should be white (value of 255) and your class should be black (value of 0).

### Downloading LabelBox data
If you use LabelBox, simply fill your API key and your project ID in `config.ini` and run:
```bash
python generate_labelbox_dataset.py
```
which will generate a folder with the structure described above.

### Generating the finetuning dataset
Once you have you labelbox dataset ready you can run:
```bash
python generate_finetuning_dataset.py
```
This will create a finetuning dataset. For each class that you want to finetune, it will create a folder named after the class in which you will find 3 folders: bboxes, images and masks.

### Generating the crack dataset
Now simply run:
```bash
python generate_cracks_dataset_full_images.py
```