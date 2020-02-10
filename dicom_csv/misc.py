import os
import warnings
from operator import itemgetter
from os.path import join as jp

import numpy as np
from pydicom import dcmread
from nibabel import Nifti1Header, Nifti1Image

from .spatial import *
from .utils import *

__all__ = 'load_series',


# TODO: move to pathlib
def load_series(row: pd.Series, base_path: PathLike = None, orientation: bool = None) -> np.ndarray:
    """
    Loads an image based on its ``row`` in the metadata dataframe.

    If ``base_path`` is not None, PathToFolder is assumed to be relative to it.

    If ``orientation`` is True, the loaded image will be transposed and flipped
    to standard (Coronal, Sagittal, Axial) orientation. If None, the orientation
    will be standardized only if possible (i.e. all the necessary metadata is present).

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

    warnings.warn('Image shape is changed, and possibly not consistent with the metadata')
    return normalize_orientation(x, row)


def construct_nifti(reference_row: pd.Series, array=None) -> Nifti1Image:
    """Construct a nifti image from dicoms.

    Notes:
    ImagePositionPatient_x,y,z;
    PixelSpacing_x,y;
    SpacingBetweenSlices;
    ImageShape are stored

    TODO: check ImagePositionPatient to be the very first one
    TODO: update requirements
    """
    if array is None:
        array = load_series(reference_row, orientation=False)

    M = get_orientation_matrix(reference_row)
    offset = list(reference_row[['ImagePositionPatient0',
                                 'ImagePositionPatient1']].values[0])
    offset.append(sorted([float(loc) for loc in reference_row['SliceLocations'].split(',')])[0])
    OM = np.concatenate((M, np.array(offset).reshape(-1,1)), axis=1)

    header = Nifti1Header()
    data_shape = [int(s) for s in reference_row['PixelArrayShape'].split(',')]
    data_shape.append(reference_row['SlicesCount'])
    header.set_data_shape(data_shape)
    header.set_zooms(reference_row[['PixelSpacing0',
                                    'PixelSpacing1',
                                    'SpacingBetweenSlices']].values[0])
    header.set_sform(OM)
    header.set_dim_info(slice=2)
    return Nifti1Image(array, OM, header=header)

