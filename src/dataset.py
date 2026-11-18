import numpy as np
import torch

from pathlib import Path
from PIL import Image, ImageSequence
from torch.utils.data import Dataset


def load_tiff_stack(path):

    with Image.open(path) as tif:

        images = [
            np.array(frame.copy())
            for frame in ImageSequence.Iterator(tif)
        ]

    return np.stack(images, axis=0)