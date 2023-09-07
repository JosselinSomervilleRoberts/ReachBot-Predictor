# Figure out CUDA memory available
import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


assert get_device() == "cuda", "CUDA not available"

import nvidia_smi

nvidia_smi.nvmlInit()


def get_cuda_free_Go() -> float:
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return info.free / 1024**3


memory_available = get_cuda_free_Go()
print(f"Detected {memory_available:.2f} GB of CUDA memory available")

if memory_available >= 16:
    crop_size = (512, 512)
    train_batch_size = 4
elif memory_available >= 8:
    crop_size = (128, 128)
    train_batch_size = 4
elif memory_available >= 4:
    crop_size = (64, 64)
    train_batch_size = 2
elif memory_available >= 2:
    crop_size = (32, 32)
    train_batch_size = 2
else:
    raise Exception("Not enough CUDA memory available")
