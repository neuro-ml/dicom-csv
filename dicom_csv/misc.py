import os
from operator import itemgetter
from os.path import join as jp

import numpy as np
from pydicom import dcmread

from .spatial import *
from .utils import *

__all__ = 'load_series',


# TODO: move to pathlib
def load_series(row: pd.Series, base_path: PathLike = None, orientation: bool = None) -> np.ndarray:
    """
    Loads an image based on its ``row`` in the metadata dataframe.

    If ``base_path`` is not None, PathToFolder is assumed to be relative to it.

    If ``orientation`` is True, the loaded image will be transposed and flipped
    to standard (Coronal, Sagittal, Axial) orientation.

    Required columns: PathToFolder, FileNames.
    """
    folder, files = row.PathToFolder, row.FileNames.split('/')
    if base_path is not None:
        folder = os.path.join(base_path, folder)
    if contains_info(row, 'InstanceNumbers'):
        files = map(itemgetter(1), sorted(zip_equal(split_floats(row.InstanceNumbers), files)))

    x = np.stack([dcmread(jp(folder, file)).pixel_array for file in files], axis=-1)
    if contains_info(row, 'RescaleSlope'):
        x = x * row.RescaleSlope
    if contains_info(row, 'RescaleIntercept'):
        x = x + row.RescaleIntercept

    if orientation is None:
        orientation = contains_info(row, *ORIENTATION)
    if not orientation:
        return x

    return normalize_orientation(x, row)
