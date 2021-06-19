from itertools import groupby
from typing import Sequence, Tuple, Union, NamedTuple
from enum import Enum

import numpy as np
import pandas as pd

from .interface import csv_series
from .tags import get_common_tag, get_tag, drop_duplicated_instances
from .utils import Series, Instance, ORIENTATION, extract_dims, split_floats, zip_equal, contains_info, collect
from .exceptions import *
from .misc import get_image

__all__ = [
    'get_orientation_matrix', 'get_slice_plane', 'get_slices_plane', 'Plane', 'order_series',
    'get_slice_orientation', 'get_slices_orientation', 'SlicesOrientation',
    'get_slice_locations', 'locations_to_spacing', 'get_slice_spacing', 'get_pixel_spacing',
    'get_voxel_spacing', 'get_image_position_patient', 'drop_duplicated_slices',
    # deprecated
    'get_axes_permutation', 'get_flipped_axes', 'get_image_plane', 
    'restore_orientation_matrix'
]


class Plane(Enum):
    Sagittal, Coronal, Axial = 0, 1, 2


class SlicesOrientation(NamedTuple):
    transpose: bool
    flip_axes: tuple


def _get_image_position_patient(instance: Instance):
    return np.array(list(map(float, get_tag(instance, 'ImagePositionPatient'))))


@csv_series
def get_image_position_patient(series: Series):
    """Returns ImagePositionPatient stacked into array."""
    return np.stack(list(map(_get_image_position_patient, series)))


def _get_image_orientation_patient(instance: Instance):
    return np.array(list(map(float, get_tag(instance, 'ImageOrientationPatient'))))


def _get_orientation_matrix(instance: Instance):
    row, col = _get_image_orientation_patient(instance).reshape(2, 3)
    return np.stack([row, col, np.cross(row, col)])


@csv_series
def get_orientation_matrix(series: Series) -> np.ndarray:
    """
    Returns a 3 x 3 orthogonal transition matrix from the image-based basis to the patient-based basis.
    Rows are coordinates of image-based basis vectors in the patient-based basis, while columns are
    coordinates of patient-based basis vectors in the image-based basis vectors.
    """
    om = _get_orientation_matrix(series[0])
    if not np.all([np.allclose(om, _get_orientation_matrix(i)) for i in series]):
        raise ConsistencyError('Orientation matrix varies across slices.')

    return om


def _get_image_planes(orientation_matrix):
    return tuple(Plane(i) for i in np.argmax(np.abs(orientation_matrix), axis=1))


def get_slice_plane(instance: Instance) -> Plane:
    return _get_image_planes(_get_orientation_matrix(instance))[2]


@csv_series
def get_slices_plane(series: Series) -> Plane:
    unique_planes = set(map(get_slice_plane, series))
    if len(unique_planes) > 1:
        raise ConsistencyError('Slice plane varies across slices.')

    plane, = unique_planes
    return plane


def get_slice_orientation(instance: Instance) -> SlicesOrientation:
    om = _get_orientation_matrix(instance)
    planes = _get_image_planes(om)

    if set(planes) != {Plane.Sagittal, Plane.Coronal, Plane.Axial}:
        raise ValueError('Main image planes cannot be treated as saggital, coronal and axial.')

    if planes[2] != Plane.Axial:
        raise NotImplementedError('We do not know what is normal orientation for non-axial slice.')

    transpose = planes[0] == Plane.Coronal
    if transpose:
        om = om[[1, 0, 2]]

    flip_axes = []
    if om[1, 1] < 0:
        flip_axes.append(0)
    if om[0, 0] < 0:
        flip_axes.append(1)

    return SlicesOrientation(transpose=transpose, flip_axes=tuple(flip_axes))


@csv_series
def get_slices_orientation(series: Series) -> SlicesOrientation:
    orientations = set(map(get_slice_orientation, series))
    if len(orientations) > 1:
        raise ConsistencyError('Slice orientation varies across slices.')

    orientation, = orientations
    return orientation


@csv_series
def order_series(series: Series, decreasing=True) -> Series:
    index = get_slices_plane(series).value
    return sorted(series, key=lambda s: _get_image_position_patient(s)[index], reverse=decreasing)


@csv_series
def get_slice_locations(series: Series) -> np.ndarray:
    """
    Computes slices location from ImagePositionPatient. 
    WARNING: the order of slice locations can be both increasing or decreasing for ordered series 
    (see order_series).
    """
    om = get_orientation_matrix(series)
    return np.array([_get_image_position_patient(i) @ om[-1] for i in series])


@csv_series
def _get_slices_deltas(series: Series) -> np.ndarray:
    """Returns distances between slices."""
    slice_locations = get_slice_locations(series)
    deltas = np.abs(np.diff(sorted(slice_locations)))
    return deltas


@csv_series
def get_slice_spacing(series: Series, max_delta: float = 0.1, errors: bool = True) -> float:
    """
    Returns constant distance between slices of a series.
    If the series doesn't have constant spacing - raises ValueError if ``errors`` is True,
    returns ``np.nan`` otherwise.
    """
    try:
        locations = get_slice_locations(series)
    except ConsistencyError:
        if errors:
            raise
        return np.nan

    return locations_to_spacing(sorted(locations), max_delta, errors)


def locations_to_spacing(locations: Sequence[float], max_delta: float = 0.1, errors: bool = True):
    def throw(err):
        if errors:
            raise err
        return np.nan

    if len(locations) <= 1:
        return throw(ValueError('Need at least 2 locations to calculate spacing.'))

    deltas = np.diff(locations)
    if len(np.unique(np.sign(deltas))) != 1:
        return throw(ConsistencyError('The locations are not strictly monotonic.'))

    deltas = np.abs(deltas)
    min_, max_ = deltas.min(), deltas.max()
    diff = max_ - min_
    if diff > max_delta:
        return throw(ConsistencyError(f'Non-constant spacing, ranging from {min_} to {max_} (delta: {diff}).'))

    return deltas.mean()


@csv_series
def get_pixel_spacing(series: Series) -> Tuple[float, float]:
    """Returns pixel spacing (two numbers) in mm."""
    pixel_spacings = np.stack([s.PixelSpacing for s in series])
    if (pixel_spacings.max(axis=0) - pixel_spacings.min(axis=0)).max() > 0.01:
        raise ConsistencyError('Pixel spacing varies across slices.')
    return pixel_spacings[0]


@csv_series
def get_voxel_spacing(series: Series):
    """Returns voxel spacing: pixel spacing and distance between slices' centers."""
    dx, dy = get_pixel_spacing(series)
    dz = get_slice_spacing(series)
    return dx, dy, dz


@csv_series
def get_image_size(series: Series):
    rows = get_common_tag(series, 'Rows')
    columns = get_common_tag(series, 'Columns')
    slices = len(series)
    return rows, columns, slices


def drop_duplicated_slices(series: Series, tolerance_hu=1) -> Series:
    series = drop_duplicated_instances(series)
    
    indices = list(range(len(series)))
    slice_locations = get_slice_locations(series)
    try:
        instance_numbers = [get_tag(i, 'InstanceNumber') for i in series]
        indices = sorted(indices, key=lambda i: (slice_locations[i], instance_numbers[i]))
    except TagMissingError:
        indices = sorted(indices, key=lambda i: slice_locations[i])
    
    new_indices = []
    for _, group in groupby(indices, key=lambda i: slice_locations[i]):
        group = list(group)
        image = get_image(series[group[0]])
        if not all(np.allclose(get_image(series[i]), image, atol=tolerance_hu) for i in group):
            raise ValueError(
                'Slices have same locations, but different pixel arrays.')
        new_indices.append(group[0])

    return [series[i] for i in sorted(new_indices)]


# ------------------ DEPRECATED ------------------------


get_image_plane = np.deprecate(get_slices_plane, old_name='get_image_plane')
get_xyz_spacing = np.deprecate(get_voxel_spacing, old_name='get_xyz_spacing')


@np.deprecate
def get_axes_permutation(row: pd.Series):
    return np.abs(get_orientation_matrix(row)).argmax(axis=0)


@np.deprecate
@csv_series
@collect
def get_flipped_axes(series: Series):
    m = get_orientation_matrix(series)
    for axis, j in enumerate(np.abs(m).argmax(axis=1)[:2]):
        if m[axis, j] < 0:
            yield axis


@np.deprecate
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


# TODO: something must return transpose order, so we can apply it to all important metadata
# TODO: take PatientPosition into account
# def transpose_series(series: Series, plane: Union[Plane, int] = Plane.Axial):
#     pass


# TODO: rewrite based on deployment code, specifically use transpose based on Plane
@np.deprecate
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


# TODO: legacy?
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
