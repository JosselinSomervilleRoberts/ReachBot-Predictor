# SKIL - A Skeleton-based Approach For Rock Crack Detection Towards A Climbing Robot Application
[paper](TODO) | [dataset](https://drive.google.com/drive/folders/1-3A6pQJ-ASxK9UKKm5T2XSwLYLYB8qZV)

In proceedings - IEEE IRC 2023

**Authors:** Josselin Somerville Roberts, Yoni Gozlan, Paul-Emiale Giacomelli, Julia Di

## Abstract 
Conventional wheeled robots are unable to traverse precipitous cave environments, which are of scientific interest for their exposed bedrock stratigraphy. Multi-limbed climbing robot designs, such as ReachBot, are able to grasp irregular surface features and execute climbing motions to overcome obstacles, given that they may find suitable grasp locations. To support grasp site identification, we present a method for detecting thin rock cracks and edges, the SKeleton Intersection Loss (SKIL). SKIL is a loss designed for thin object segmentation that leverages the skeleton of the label. A dataset of RGB images from Pinnacles National Park was collected, manually annotated, and augmented. New metrics have been proposed for thin object segmentation such that the impact of the object width on the score is minimized. In addition, the metric is less sensitive to translation which can often lead to a score of zero when computing classical metrics such as dice on thin objects. Our fine-tuned models outperform previous methods on similar thin object segmentation tasks such as blood vessel segmentation and show promise for integration onto a robotic system.


## Table of contents
* [Installation](#installation)
* [Datasets](#datasets)
* [Recreate the results from the paper](#recreate)

## Installation
We provide a conda environment. Simply run:
```bash
conda env create -f environment.yml
conda activate reachbot
cd mmsegmentation
pip install -e .
```

## Datasets
To recreate the `cracks` dataset from you own image follow this [guide](./dataset_builder/README.md).

To download our `cracks` dataset (already formatted), please use this [link](TODO). Then make sure to place the content of the dataset in `./datasets/cracks` (this folder should contain `ann_dir` and `img_dir`).

To download the blood vessels datasets (`CHASE DB1`, `DRIVE`, `HRF` and `STARE`) please use `mmsegmentation`'s [guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#chase-db1). You can then use our script to combine the datasets; `dataset_combiner/dataset_combiner.py` *(Make sure to properly rename the images of each dataset for this, see the documentation of the script for more details)*. you can also generate the deformed datasets by modifying the `MODIFIERS` object.




## <a name="recreate"></a>Recreate the results from the paper

In this section we describe the exact commands to run the same experiments as us and recreate the exact same results. This include even figure like the first figure of the paper. All commands should be run inside the `reachbot` environment from `./mmsegmentation`.

### Figure 1.
To generate the mosaique of images you can use our script:
```bash
cd ../dataset_builder/generate_workflow_picture
python split_image.py --input_path <PATH-TO-YOUR-IMAGE>
```

### Figures 2, 3 and 4
Figure 2 was drawn and Figures 3 and 4 are images taken from the datasets.

### Figure 5.
To recreate Figure 5, we provide the script `dataset_builder/metrics_comparison.py`. Then run the script with different parameters to recreate the images:
```bash
cd ../dataset_builder
python metrics_comparison.py --desired_dice 0.2 --desired_crack_metric_diff 0.2 --output_path metric_images_2
python metrics_comparison.py --desired_dice 0.4 --desired_crack_metric_diff 0.2 --output_path metric_images_4
python metrics_comparison.py --desired_dice 0.6 --desired_crack_metric_diff 0.2 --output_path metric_images_6
```

### Table I. and Figure 6.
Run the following trainings:
```bash
python tools/train_repeat.py --num_repeats 20 --config \
paper_configs/vit/cracks/dice.py \
paper_configs/vit/cracks/cl_dice.py \
paper_configs/vit/cracks/skil_dice.py \
paper_configs/vit/cracks/skil_prod.py \
-- --amp
```
See our Wandb run:
* **Combined cracks** with ViT-B: [link](https://wandb.ai/single-shot-robot/cracks_combined_segmentation_prod?workspace=user-)

### Table II. and Figure 7.
Run the following trainings:
```bash
python tools/train_repeat.py --num_repeats 20 --config \
paper_configs/vit/vessels/dice.py \
paper_configs/vit/vessels/cl_dice.py \
paper_configs/vit/vessels/skil_dice.py \
paper_configs/vit/vessels/skil_prod.py \
-- --amp
```
See our Wandb run:
* **Combined vessels** with ViT-B: [link](https://wandb.ai/single-shot-robot/vessels_combined_segmentation_prod?workspace=user-josselin)

### Table III. and Figure 8.
Run the following trainings:
```bash
python tools/train_repeat.py --num_repeats 10 --config \
paper_configs/unet/cracks/ce.py \
paper_configs/unet/cracks/cl_dice.py \
paper_configs/unet/cracks/skil_dice.py \
paper_configs/unet/cracks/skil_prod.py \
-- --amp
```
See our Wandb run:
* **Combined cracks** dataset with U-Net: [link](https://wandb.ai/single-shot-robot/CRACKS_segmentation_prod?workspace=user-)

### Table IV.
Run the following trainings:
```bash
python tools/train_repeat.py --num_repeats 10 --config \
paper_configs/unet/stare/ce.py \
paper_configs/unet/stare/cl_dice.py \
paper_configs/unet/stare/skil_dice.py \
paper_configs/unet/stare/skil_prod.py \
paper_configs/unet/chase/ce.py \
paper_configs/unet/chase/cl_dice.py \
paper_configs/unet/chase/skil_dice.py \
paper_configs/unet/chase/skil_prod.py \
-- --amp
```
See our Wandb runs:
* **STARE** dataset with U-Net: [link](https://wandb.ai/single-shot-robot/STARE_segmentation_prod?workspace=user-)
* **CHASE DB1** dataset with U-Net: [link](https://wandb.ai/single-shot-robot/CHASE_segmentation_prod?workspace=user-)

### Figure 9.
Run the following script. It will prompt a menu to choose the augmentation to run, choose the one you want.
```bash
cd ../dataset_builder
python test_annotation_modifiers.py
```

### Tables V. and VI.
Run the following trainings:
```bash
python tools/train_repeat.py --num_repeats 10 --config \
paper_configs/vit/vessels_shifted/dice.py \
paper_configs/vit/vessels_shifted/cl_dice.py \
paper_configs/vit/vessels_shifted/skil_dice.py \
paper_configs/vit/vessels_width/dice.py \
paper_configs/vit/vessels_width/cl_dice.py \
paper_configs/vit/vessels_width/skil_dice.py \
paper_configs/vit/vessels_cropped/dice.py \
paper_configs/vit/vessels_cropped/cl_dice.py \
paper_configs/vit/vessels_cropped/skil_dice.py \
paper_configs/vit/vessels_degraded/dice.py \
paper_configs/vit/vessels_degraded/cl_dice.py \
paper_configs/vit/vessels_degraded/skil_dice.py \
-- --amp
```
Table VI. is then obtained by devising the entried of Table V. by the entried of Table II. (See paper for more details).

See our Wandb runs:
* **Shifted** combined vessels dataset with U-Net: [link](https://wandb.ai/single-shot-robot/vessels_combined_shifted_segmentation_prod?workspace=user-)
* **Random width** combined vessels dataset with U-Net: [link](https://wandb.ai/single-shot-robot/vessels_combined_width_segmentation_prod?workspace=user-)
* **Branches cropped** combined vessels dataset with U-Net: [link](https://wandb.ai/single-shot-robot/vessels_combined_cropped_segmentation_prod?workspace=user-)
* **Combined deformations** combined vessels dataset with U-Net: [link](https://wandb.ai/single-shot-robot/vessels_combined_degraded_segmentation_prod?workspace=user-)



<!-- # Crack

To get the data, check the `dataset_builder` `README` file.

To train a model, check the `mmsegmentation` `README` file. -->

<!-- ## Installation

Run the following commands

```bash
conda create -n reachbot python=3.8
conda activate reachbot
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install matplotlib
pip install scikit-image
pip install chardet
pip install "labelbox[data]"
pip install tensorboard
pip install git+https://github.com/JosselinSomervilleRoberts/JossPythonToolbox.git
pip install --upgrade --force-reinstall Pillow
```

Verify installation by trying in `python`:

```python
import torch
from toolbox.printing import warn, print_color
if torch.cuda.is_available():
    print_color("CUDA is available", color="green")
else:
    warn("CUDA is not available")
```

Then you can generate the datasets by doing:

1. Create a `config.ini` file: `cp config.example.ini config.ini` and fill `LABELBOX_API_KEY` and `LABELBOX_PROJECT_ID` with your own values.
2. Run `python generate_labelbox_dataset.py` to download the dataset from Labelbox and generate the `datasets/labelbox` folder.
3. Run `python generate_finetuning_dataset.py` to generate the `datasets/finetune` folder _(ready to be used for finetuning)_.

## MMSelfSup

### Installation

Necessitates `python=3.8` and pytorch

Run the following commands

```bash
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc1'
```

Install mmselfsup from source

```bash
cd mmselfsup
pip install -v -e .
```

Verify the installation

```python
import mmselfsup
print(mmselfsup.__version__)
```

Install MMDetection and MMSegmentation

```bash
pip install 'mmdet>=3.0.0rc0' 'mmsegmentation>=1.0.0rc0'
```

### Commands

Pretrain model from existing weights

```bash
python tools/train.py configs/selfsup/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_reachbot.py --cfg-options model.pretrained=../pretrained_ckpt/mocov3_resnet50_8xb512-amp-coslr-800e_in1k_20220927-e043f51a.pth
```

Pretrain model from scratch

```bash
python tools/train.py configs/selfsup/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_reachbot.py
``` -->
