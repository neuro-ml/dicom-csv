import os
import warnings
from operator import itemgetter
from os.path import join as jp

import numpy as np

from .rtstruct.contour import read_rtstruct, contours_to_image, contour_to_mask
from .spatial import *
from .spatial import get_image_size
from .utils import *

__all__ = 'load_series', 'get_image', 'stack_images'


def get_image(instance: Dataset):
    def _to_int(x):
        if x == int(x):
            x = int(x)
        return x

    array = instance.pixel_array
    slope, intercept = instance.get('RescaleSlope'), instance.get('RescaleIntercept')
    if slope is not None and slope != 1:
        array = array * _to_int(slope)
    if intercept is not None and intercept != 0:
        array = array + _to_int(intercept)

    return array


def stack_images(series: Series, axis: int = -1):
    return np.moveaxis(np.array(list(map(get_image, series))), 0, axis)


# TODO: legacy support
class Default:
    pass


# TODO: move to pathlib
@np.deprecate
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


def construct_nifti(series: Series):
    """Construct a nifti image from DICOMs.

    Notes:
    ---
    Metadata stored in NIFTI:
    ImagePositionPatient_x,y,z; PixelSpacing_x,y; SpacingBetweenSlices; ArrayShape
    """
    from nibabel import Nifti1Header, Nifti1Image

    m = get_orientation_matrix(series)
    offset = get_image_position_patient(series)[0]
    voxel_spacing = get_voxel_spacing(series)
    om = np.eye(4)
    om[:3, :3] = m
    om[:3, 3] = offset
    om = om * np.diag(np.hstack([voxel_spacing, 1.]))
    data_shape = get_image_size(series)

    # Looks like Nifti1Image overwrites OM if it is provided in header
    # see https://github.com/nipy/nibabel/blob/master/nibabel/nifti1.py Nifti1Header.set_qform()

    # About qform, sform
    # https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/qsform_brief_usage
    header = Nifti1Header()
    header.set_data_shape(data_shape)
    array = stack_images(series)

    return Nifti1Image(array, om, header=header)
