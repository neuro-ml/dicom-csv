from pydicom import dcmread
from pydicom.dataset import Dataset

import numpy as np

from ..utils import split_ints, split_floats
from .meta import _get_contour_seq_name
from ..spatial import get_fixed_orientation_matrix, get_xyz_spacing, get_patient_position

__all__ = 'read_rtstruct', 'contours_to_image'


def _read_contour_sequence(dataset: Dataset) -> dict:

    contour_sequence_dict = dict()

    if not hasattr(dataset, 'ContourSequence'):
        raise AttributeError('Dataset does not have Contour Sequence.')

    for _slice in dataset.ContourSequence:
        key = _slice.ContourImageSequence[0].ReferencedSOPInstanceUID
        coords = np.array(_slice.ContourData)
        n = len(coords) // 3
        coords = coords.reshape((n, 3))
        if key not in contour_sequence_dict:
            contour_sequence_dict[key] = [coords]
        else:
            contour_sequence_dict[key].append(coords)

    return contour_sequence_dict


def read_rtstruct(row) -> dict:
    if row.InstanceNumbers is None:
        raise AttributeError('Contour does not have associated image.')

    p = '/'.join(row[['PathToFolder', 'FileName']].values)
    numbers = np.array(split_ints(row.InstanceNumbers)) - 1
    keys = row['SOPInstanceUIDs'].split(',')
    d = dict(zip(keys, numbers))
    rtstruct = dcmread(p)
    contours = list(rtstruct.ROIContourSequence)
    roi_names = _get_contour_seq_name(rtstruct=rtstruct)
    contours_result = dict()

    for ind, (name, roi) in enumerate(zip(roi_names, contours)):
        try:
            _coords = _read_contour_sequence(roi)
            _coords = dict([(d[key], value) for key, value in _coords.items()])
            # It is possible to have multiple ROIs with the same ROIName,
            # therefore we add an index.
            contours_result[f'{ind}_{name}'] = _coords
        except AttributeError:
            print(f'No {name} contour.')

    return contours_result


def contours_to_image(row, contours_dict):
    """Moves contours coordinates to image space."""
    #  TODO: write get_position function
    OM = get_fixed_orientation_matrix(row)
    xyz = get_xyz_spacing(row)
    pos = get_patient_position(row)[:, 1:]

    contours_image_dict = dict()

    for roi_name, roi_coordinates in contours_dict.items():
        contours_image_dict[roi_name] = dict()
        for slice_number, coordinates_list in roi_coordinates.items():
            contours_image_dict[roi_name][slice_number] = []
            for coords in coordinates_list:
                new_coords = (coords - pos[slice_number]).dot(OM) / xyz
                contours_image_dict[roi_name][slice_number].append(new_coords)

    return contours_image_dict
