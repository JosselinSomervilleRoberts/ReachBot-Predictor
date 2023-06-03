import numpy as np
import PIL.Image
from matplotlib import pylab as P


def ShowImage(im, title="", ax=None):
    if ax is None:
        P.figure()
    P.axis("off")
    P.imshow(im)
    P.title(title)


def ShowGrayscaleImage(im, orig, title="", ax=None, alpha=0.3):
    if ax is None:
        P.figure()
    P.axis("off")
    P.imshow(orig)
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1, alpha=alpha)
    P.title(title)


def ShowHeatMap(im, title, orig, ax=None, alpha=0.3):
    if ax is None:
        P.figure()
    P.axis("off")
    P.imshow(orig)
    P.imshow(im, cmap="inferno", alpha=alpha)
    P.title(title)


def LoadImage(file_path):
    im = PIL.Image.open(file_path)
    im = im.resize((299, 299))
    im = np.asarray(im)
    return im
