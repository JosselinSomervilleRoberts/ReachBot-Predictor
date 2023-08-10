import nvidia_smi
import os
import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# Source: https://adamoudad.github.io/posts/progress_bar_with_tqdm/
def get_ram_used() -> float:
    # Getting all memory using os.popen()
    total_memory, used_memory, free_memory = map(
        int, os.popen("free -t -m").readlines()[-1].split()[1:]
    )

    # Memory usage
    ram_used = (used_memory / total_memory) * 100
    return ram_used


def get_cuda_used() -> float:
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return 100 - 100 * info.free / info.total


def set_description(pbar, description: str, k: int, frequency: int = 50):
    if k % frequency == 0:
        pbar.set_description(
            f"{description} (RAM used: {get_ram_used():.2f}% / CUDA used {get_cuda_used():.2f}%)"
        )
