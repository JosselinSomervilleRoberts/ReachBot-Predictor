# Reachbot mmsegmentation

This is a fork of **mmsegmentation**.

## Installation

To install run the following:
```bash
conda create -n reachbot python=3.8
conda activate reachbot
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib
pip install scikit-image
pip install chardet
pip install tensorboard
pip install git+https://github.com/JosselinSomervilleRoberts/JossPythonToolbox.git
pip install --upgrade --force-reinstall Pillow
pip install nvidia-ml-py3
pip install 'mmdet>=3.0.0rc0' 'mmsegmentation>=1.0.0rc0'
pip install -e .
```

## Training

You can edit `config.py` but make sure not to commit your changes.
Then simply run:
```bash
python tools/train.py config.py
```


## Adding new models, datatasets, schedules

If you want to add new models, datasets or schedules, you can do so in `custom_configs` which are not *mmsegmentation*'s config. Then make sure to edit `custom_configs/main.py` to make your new config file available in `config.py`.