## Installation
Run the following commands
```bash
conda create -n reachbot python=3.8
conda activate reachbot
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install matplotlib
pip install chardet
pip install "labelbox[data]"
pip install --upgrade --force-reinstall Pillow
pip install git+https://github.com/JosselinSomervilleRoberts/JossPythonToolbox.git
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
3. Run `python generate_finetuning_dataset.py` to generate the `datasets/finetune` folder *(ready to be used for finetuning)*.