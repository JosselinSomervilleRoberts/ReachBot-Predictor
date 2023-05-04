```bash
conda create -n reachbot python=3.8
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install chardet
pip install "labelbox[data]"
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install --upgrade --force-reinstall Pillow  
```