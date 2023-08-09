# Crack

To get the data, check the `dataset_builder` `README` file.

To train a model, check the `mmsegmentation` `README` file.

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
