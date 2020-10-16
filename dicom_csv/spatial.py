import numpy as np
import pandas as pd
from typing import Sequence
from pydicom import Dataset
from dicom_csv.interface import csv_series
from enum import Enum
from .utils import *

__all__ = [
    'get_orientation_matrix', 'restore_orientation_matrix',
    'should_flip', 'normalize_orientation', 'get_slice_spacing',
    'get_xyz_spacing', 'get_flipped_axes', 'get_axes_permutation',
    'get_image_position_patient', 'get_slice_locations', 'get_image_plane', 'Plane',
    'get_pixel_spacing'
]


class Plane(Enum):
    Sagittal, Coronal, Axial = 0, 1, 2


# TODO: Returns list if dicom_csv.interface.RowIndex instances, only tested for Sequence[Dataset]
@csv_series
def order_series(series: Sequence[Dataset], decreasing=True):
    """Returns sequence of instances in decreasing/increasing order of their slice locations."""
    slices_location = get_slice_locations(series)
    slices_order = np.argsort(slices_location)
    if decreasing:
        slices_order = slices_order[::-1]
    return [series[i] for i in slices_order]


@csv_series
def _get_slices_spacing(series: Sequence[Dataset]) -> Sequence[float]:
    """Returns distances between slices."""
    slice_locations = get_slice_locations(series)
    deltas = np.abs(np.diff(sorted(slice_locations)))
    return deltas


@csv_series
def get_slice_spacing(series: Sequence[Dataset]):
    """Returns constant distance between slices of a series."""
    deltas = _get_slices_spacing(series)
    if np.any(np.isclose(deltas, 0)):
        raise ValueError('Duplicated slices.')
    if np.max(deltas) - np.min(deltas) > 0.1:
        raise ValueError('Non-constant slice spacing.')
    return deltas[0]


@csv_series
def is_axial(series: Sequence[Dataset]):
    """Checks if series has an Axial main plain."""
    return get_image_plane(series) == Plane.Axial


@csv_series
def get_image_plane(series: Sequence[Dataset]) -> Plane:
    """
    Returns main plane of the image if it exists. Might not work with large rotation, since there is no
    `main plane` in that case.
    """
    pos = get_image_position_patient(series)
    slices = get_slice_locations(series)
    slices_order = np.argsort(slices)
    index = np.argmax(np.abs(np.diff(pos[slices_order], axis=0)), axis=1)[0]
    return Plane(index)


@csv_series
def get_slice_locations(series: Sequence[Dataset]):
    """
    Computes slices location from ImagePositionPatient.
    """
    image_position = get_image_position_patient(series)
    om = get_orientation_matrix(series)
    return image_position @ om.T[:, -1]


def _get_image_orientation_patient(instance: Dataset):
    try:
        return np.array(list(map(float, instance.ImageOrientationPatient)))
    except AttributeError as e:
        raise AttributeError('The tag "ImageOrientationPatient" is missing.') from e


@csv_series
def get_orientation_matrix(series: Union[Sequence[Dataset], pd.DataFrame]):
    """Returns 3 x 3 orientation matrix from single series."""
    # TODO: check if it always stored in a column-wise fashion
    om = _get_image_orientation_patient(series[0])
    if not np.all([np.allclose(om, _get_image_orientation_patient(x)) for x in series]):
        raise AttributeError('Orientation matrix varies across slices.')

    x, y = om.reshape(2, 3)
    return np.stack([x, y, np.cross(x, y)])


@csv_series
def get_image_position_patient(series: Sequence[Dataset]):
    """Returns ImagePositionPatient stacked into array."""
    try:
        return np.stack([s.ImagePositionPatient for s in series])
    except AttributeError as e:
        raise AttributeError('The tag "ImagePositionPatient" is missing.') from e


@csv_series
def get_pixel_spacing(series: Sequence[Dataset]):
    """Returns pixel spacing (two numbers) in mm."""
    pixel_spacings = np.stack([s.PixelSpacing for s in series])
    if (pixel_spacings.max(axis=0) - pixel_spacings.min(axis=0)).max() > 0.01:
        raise ValueError('The series has inconsistent pixel spacing.')
    return pixel_spacings[0]


@csv_series
def get_xyz_spacing(series: Sequence[Dataset]):
    """Returns voxel spacing: pixel spacing and distance between slices' centers."""
    dx, dy = get_pixel_spacing(series)
    dz = get_slice_spacing(series)
    return dx, dy, dz


@csv_series
def get_image_size(series: Sequence[Dataset]):
    # TODO: check uniqueness across instances
    rows, columns = series[0].Rows, series[0].Columns
    slices = len(series)
    return rows, columns, slices


def restore_orientation_matrix(metadata: Union[pd.Series, pd.DataFrame]):
    """
    Fills nan values (if possible) in ``metadata``'s ImageOrientationPatient* rows.

    Required columns: ImageOrientationPatient[0-5]

    Notes
    -----
    The input dataframe will be mutated.
    """

    def restore(vector):
        null = pd.isnull(vector)
        if null.any() and not null.all():
            length = 1 - (vector[~null] ** 2).sum()
            vector = vector.copy()
            vector[null] = length / np.sqrt(null.sum())

        return vector

    coords = np.moveaxis(metadata[ORIENTATION].astype(float).values.reshape(-1, 2, 3), 1, 0)
    result = np.concatenate([restore(x) for x in coords], axis=1)

    if metadata.ndim == 1:
        result = extract_dims(result)

    metadata[ORIENTATION] = result
    return metadata


def order_slice_locations(dicom_metadata: pd.Series):
    locations = split_floats(dicom_metadata.SliceLocations)
    if np.any([np.isnan(loc) for loc in locations]):
        raise ValueError("Some SliceLocations are missing")
    # Do not put `restore_slice_location` here,
    # since `should_flip` has unexpected behaviour in that case.
    return np.array(sorted(zip_equal(
        split_floats(dicom_metadata.InstanceNumbers),
        locations
    ))).T


def should_flip(dicom_metadata: pd.Series):
    """
    Returns True if the whole series' should be flipped
    in order to account for 'HF?' patient position.
    """
    _, locations = order_slice_locations(dicom_metadata)
    direction = dicom_metadata.PatientPosition[:2] == 'HF'
    flip = locations[0] > locations[-1]
    return flip != direction


# TODO: deprecate
def get_axes_permutation(row: pd.Series):
    return np.abs(get_orientation_matrix(row)).argmax(axis=0)


# TODO: deprecate
def get_flipped_axes(row: pd.Series):
    flips = []
    m = get_orientation_matrix(row)
    for i, j in enumerate(np.abs(m).argmax(axis=1)):
        flips.append(m[i, j] < 0)

    if contains_info(row, 'InstanceNumbers', 'SliceLocations') and should_flip(row):
        flips[-1] = not flips[-1]

    return [axis for axis, flip in enumerate(flips) if flip]


# TODO: something must return transpose order, so we can apply it to all important metadata
# TODO: take PatientPosition into account
def transpose_series(series: Sequence[Dataset], plane: Union[Plane, int] = Plane.Axial):
    pass


# TODO: rewrite based on deployment code, specifically use transpose based on Plane
def normalize_orientation(image: np.ndarray, row: pd.Series):
    """
    Transposes and flips the ``image`` to standard (Coronal, Sagittal, Axial) orientation.

    Warnings
    --------
    Changing image orientation. New image orientation will not coincide with metadata!
    """

    if not contains_info(row, *ORIENTATION):
        raise ValueError('There is no enough metadata to standardize the image orientation.')

    if np.isnan(get_orientation_matrix(row)).any():
        raise ValueError('There is no enough metadata to standardize the image orientation.')

    image = np.flip(image, axis=get_flipped_axes(row))
    return image.transpose(*get_axes_permutation(row))
