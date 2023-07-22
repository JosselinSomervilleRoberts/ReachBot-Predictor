import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Optional, Union


class Plotter:
    """Plotting static class."""

    img_index: int = 0
    subplot_index: int = 0
    stated: bool = False
    base_path: str = None

    @staticmethod
    def plot_mask(mask: Union[torch.Tensor, np.ndarray], title: Optional[str] = None):
        """Plot a tmask."""
        if not Plotter.started:
            raise Exception("You must call start() before calling plot()!")
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        while len(mask.shape) > 2 and mask.shape[0] == 1:
            mask = mask[0]
        plt.subplot(Plotter.n_rows, Plotter.n_cols, Plotter.subplot_index + 1)
        plt.imshow(mask)
        plt.colorbar(cmap="gray")
        if title is not None:
            plt.title(title)
        Plotter.subplot_index += 1


    @staticmethod
    def plot_image(image: Union[torch.Tensor, np.ndarray], title: Optional[str] = None):
        """Plot an image."""
        if not Plotter.started:
            raise Exception("You must call start() before calling plot()!")
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        while len(image.shape) > 3 and image.shape[0] == 1:
            image = image[0]
        if len(image.shape) > 3 or len(image.shape) <= 2:
            raise Exception("The image must have  3 dimensions! The current dimensions are: " + str(image.shape))
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if not image.shape[2] == 3:
            raise Exception("The image must have 3 channels! The current dimensions are: " + str(image.shape))
        plt.subplot(Plotter.n_rows, Plotter.n_cols, Plotter.subplot_index + 1)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        Plotter.subplot_index += 1


    @staticmethod
    def plot(tensor: Union[torch.Tensor, np.ndarray], title: Optional[str] = None):
        """Plot a tensor."""
        if len(tensor.shape) == 2:
            return Plotter.plot_mask(tensor, title)
        while len(tensor.shape) > 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if len(tensor.shape) > 3:
            raise Exception("The tensor must have 3 dimensions or less! The current dimensions are: " + str(tensor.shape))
        if tensor.shape[0] == 3 or tensor.shape[2] == 3:
            Plotter.plot_image(tensor, title)
        else:
            Plotter.plot_mask(tensor, title)

    @staticmethod
    def skip():
        """Skip a plot."""
        if not Plotter.started:
            raise Exception("You must call start() before calling skip()!")
        Plotter.subplot_index += 1

    @staticmethod
    def start(num_rows: int, num_cols: int, base_path: Optional[str] = None):
        """Start a new plot."""
        Plotter.base_path = base_path
        if not Plotter.base_path:
            raise Exception("You must provide a base_path!")
        Plotter.n_rows = num_rows
        Plotter.n_cols = num_cols
        plt.figure(figsize=(Plotter.n_cols * 5, Plotter.n_rows * 3))
        Plotter.subplot_index = 0
        Plotter.started = True

    @staticmethod
    def finish():
        """Finish a plot."""
        if not Plotter.started:
            raise Exception("You must call start() before calling finish()!")
        plt.savefig(f"{Plotter.base_path}/similarity_mask_{Plotter.img_index}.png")
        plt.close()
        Plotter.subplot_index = 0
        Plotter.img_index += 1
        Plotter.started = False
        # print(f"Finished plot {Plotter.img_index}!")