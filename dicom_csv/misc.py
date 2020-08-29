import os
import warnings
from operator import itemgetter
from os.path import join as jp

import numpy as np
from pydicom import dcmread

from dicom_csv.rtstruct.contour import read_rtstruct, contours_to_image, contour_to_mask
from .spatial import *
from .spatial import get_image_size
from .utils import *

__all__ = 'load_series',


# TODO: legacy support
class Default:
    pass


# TODO: move to pathlib
def load_series(row: pd.Series, base_path: PathLike = None, orientation: Union[bool, None] = Default,
                scaling: bool = None) -> np.ndarray:
    """
    Loads an image based on its ``row`` in the metadata dataframe.

    If ``base_path`` is not None, PathToFolder is assumed to be relative to it.

    If ``orientation`` is True, the loaded image will be transposed and flipped
    to standard (Coronal, Sagittal, Axial) orientation. If None, the orientation
    will be standardized only if possible (i.e. all the necessary metadata is present).

    Required columns: PathToFolder, FileNames.
    """
    if orientation is Default:
        orientation = None
        warnings.warn('The default value for `orientation` will be changed to `False` in next releases. '
                      'Pass orientation=None, if you wish to keep the old behaviour.', UserWarning)

    folder, files = row.PathToFolder, row.FileNames.split('/')
    if base_path is not None:
        folder = os.path.join(base_path, folder)
    if contains_info(row, 'InstanceNumbers'):
        files = map(itemgetter(1), sorted(zip_equal(split_floats(row.InstanceNumbers), files)))

    x = np.stack([dcmread(jp(folder, file)).pixel_array for file in files], axis=-1)

    if scaling and not contains_info(row, 'RescaleSlope', 'RescaleIntercept'):
        raise ValueError('Not enough information for scaling.')
    if scaling is not False and contains_info(row, 'RescaleSlope'):
        x = x * row.RescaleSlope
    if scaling is not False and contains_info(row, 'RescaleIntercept'):
        x = x + row.RescaleIntercept

    if orientation is None:
        orientation = contains_info(row, *ORIENTATION)
    if orientation:
        x = normalize_orientation(x, row)

    return x


def load_rtstruct(rtstruct_row: pd.Series, contour_name: str = None) -> dict:
    """Loads all masks from RTStruct.

    Parameters
    ---
    rtstruct_row - pandas.Series,
        single row from a DataFrame after dicom_csv.rtstruct.collect_rtstruct

    contour_name - str,
        name of the contour to return, if None, returns dictionary with all masks, default is None.
    """

    contours_world = read_rtstruct(rtstruct_row)
    contours_image = contours_to_image(rtstruct_row, contours_world)
    size = get_image_size(rtstruct_row)
    if contour_name is not None:
        try:
            return contour_to_mask(contours_image[contour_name], size=size)
        except KeyError:
            print(f'Contour {contour_name} is not presented in RTStruct.')

    masks = dict()
    for contour_name, contour in contours_image.items():
        masks[contour_name] = contour_to_mask(contours_image[contour_name], size=size)

    return masks


def construct_nifti(reference_row: pd.Series, array=None, base_path: PathLike = None):
    """Construct a nifti image from dicoms.

    If ``array`` is not None, image metadata is taken from reference_row,
    and image tensor is an array. This is mainly for saving contour masks.

    Notes:
    Metadata stored in NIFTI:
    ImagePositionPatient_x,y,z; PixelSpacing_x,y; SpacingBetweenSlices; ArrayShape
    """
    from nibabel import Nifti1Header, Nifti1Image

    if array is None:
        array = load_series(reference_row, orientation=False, base_path=base_path)

    M = get_orientation_matrix(reference_row)
    offset = get_patient_position(reference_row)[0, 1:]
    slice_spacing = get_slice_spacing(reference_row)
    pixel_spacings = reference_row[['PixelSpacing0', 'PixelSpacing1']].values
    OM = np.eye(4)
    OM[:3, :3] = M
    OM[:3, 3] = offset
    OM = OM * np.diag(np.hstack((pixel_spacings, slice_spacing)))
    data_shape = get_image_size(reference_row)

    # Looks like Nifti1Image overwrites OM if it is provided in header
    # see https://github.com/nipy/nibabel/blob/master/nibabel/nifti1.py Nifti1Header.set_qform()
    header = Nifti1Header()
    header.set_data_shape(data_shape)

    return Nifti1Image(array, OM, header=header)
