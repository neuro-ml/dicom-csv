import numpy as np

from .utils import *

__all__ = [
    'get_orientation_matrix', 'get_orientation_axis', 'restore_orientation_matrix',
    'should_flip', 'normalize_orientation', 'get_slice_spacing', 'get_patient_position',
    'get_fixed_orientation_matrix', 'get_xyz_spacing'
]


def get_orientation_matrix(metadata: Union[pd.Series, pd.DataFrame]):
    """Required columns: ImageOrientationPatient[0-5]"""
    orientation = metadata[ORIENTATION].astype(float).values.reshape(-1, 2, 3)
    cross = np.cross(orientation[:, 0], orientation[:, 1], axis=1)
    result = np.concatenate([orientation, cross[:, None]], axis=1)

    if metadata.ndim == 1:
        result = extract_dims(result)
    return result


def get_fixed_orientation_matrix(row, return_main_plain_axis=False):
    """Sometimes Orientation Matrix is stored in row-wise fashion instead of column-wise.
    Here we check this and return column-wise OM"""

    # TODO: compare return_main_plain with `get_orientation_axis`

    def check(d):
        """Two out of three coordinates should be equal across slices."""
        return (d < 0.05).sum() == 2

    coords = get_patient_position(row)[:, 1:]
    OM = get_orientation_matrix(row)

    for om in [OM, OM.T]:
        new_coords = coords.dot(om)
        delta = np.max(new_coords, axis=0) - np.min(new_coords, axis=0)
        if check(delta):
            if return_main_plain_axis:
                return om, np.where(delta < 0.05)[0]
            return om
    raise ValueError('ImagePositionPatient coordinates are inconsistent.')


def get_orientation_axis(metadata: Union[pd.Series, pd.DataFrame]):
    """Required columns: ImageOrientationPatient[0-5]"""
    m = get_orientation_matrix(metadata)
    matrix = np.atleast_3d(m)
    result = np.array([np.nan if np.isnan(row).any() else np.abs(row).argmax(axis=0)[2] for row in matrix])

    if m.ndim == 2:
        result = extract_dims(result)
    return result


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


def restore_slice_locations(dicom_metadata: pd.Series):
    """Restore SliceLocation from ImagePositionPatient,
    as if orientation matrix was Identity"""

    pos = get_patient_position(dicom_metadata)
    instances, coords = pos[:, 0], pos[:, 1:]
    OM, main_plain_axis = get_fixed_orientation_matrix(dicom_metadata, return_main_plain_axis=True)
    new_coords = coords.dot(OM)
    j = list({0, 1, 2}.difference(set(main_plain_axis)))[0]
    return np.vstack((instances, new_coords[:, j]))


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
    in order to account for 'HFS' patient position.
    """
    _, locations = order_slice_locations(dicom_metadata)
    direction = dicom_metadata.PatientPosition == 'HFS'
    flip = locations[0] > locations[-1]
    return flip != direction


def get_slice_spacing(dicom_metadata: pd.Series, check: bool = True) -> float:
    """
    Computes the spacing between slices for a dicom series.
    Add slice restoration in case of non diagonal rotation matrix
    """
    instances, locations = restore_slice_locations(dicom_metadata)
    dx, dy = np.diff([instances, locations], axis=1)
    spacing = dy / dx

    if len(spacing) == 0:
        if check:
            raise ValueError('The provided metadata must contain al least 2 images.')

        return np.nan

    if spacing.max() - spacing.min() > 0.01:
        if check:
            raise ValueError(f'Seems like this series has an inconsistent slice spacing: {spacing}.')

        return np.nan

    return np.abs(spacing.mean())


def normalize_orientation(image: np.ndarray, row: pd.Series):
    """
    Transposes and flips the ``image`` to standard (Coronal, Sagittal, Axial) orientation.

    Warnings
    --------
    Changing image orientation. New image orientation will not coincide with metadata!
    """

    if not contains_info(row, *ORIENTATION):
        raise ValueError('There is no enough metadata to standardize the image orientation.')

    m = get_orientation_matrix(row)
    if np.isnan(get_orientation_matrix(row)).any():
        raise ValueError('There is no enough metadata to standardize the image orientation.')

    if contains_info(row, 'InstanceNumbers', 'SliceLocations') and should_flip(row):
        image = image[..., ::-1]

    for i, j in enumerate(np.abs(m).argmax(axis=1)):
        if m[i, j] < 0:
            image = np.flip(image, axis=i)
    return image.transpose(*np.abs(m).argmax(axis=0))


def get_patient_position(row: pd.Series):
    """Returns ImagePatientPosition_x,y,z"""
    # TODO: Consider rewriting this to take into account non identity OMs
    pos = np.array(sorted(zip(
        split_floats(row['InstanceNumbers']),
        split_floats(row['ImagePositionPatient0s']),
        split_floats(row['ImagePositionPatient1s']),
        split_floats(row['ImagePositionPatient2s']),
    )))
    return pos


def get_xyz_spacing(row: pd.Series):
    """Returns pixel spacing + distance between slices (between their centers),
    in an order consistent with ImagePositionPatient's columns order."""
    _, indices = get_fixed_orientation_matrix(row, return_main_plain_axis=True)
    xyz = np.zeros(3)
    xy = list(row[['PixelSpacing0', 'PixelSpacing1']].values)
    index_z = list({0, 1, 2}.difference(set(indices)))[0]

    xyz[indices[0]] = xy[0]
    xyz[indices[1]] = xy[1]
    xyz[index_z] = get_slice_spacing(row)
    return xyz
