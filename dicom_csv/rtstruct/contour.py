from pydicom import dcmread
from pydicom.dataset import Dataset

import numpy as np
import pandas as pd

from ..utils import split_ints
from .meta import _get_contour_seq_name
from ..spatial import get_fixed_orientation_matrix, get_xyz_spacing, get_patient_position

__all__ = 'read_rtstruct', 'contours_to_image'


def _read_contour_sequence(dataset: Dataset) -> dict:
    """Extract a single contour from contour's Dataset."""
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


def read_rtstruct(row: pd.Series) -> dict:
    """Read dicom file with RTStruture."""
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


def contours_to_image(row: pd.Series, contours_dict: dict) -> dict:
    """Moves contours coordinates to image space.

    Returns
    ---

    contours_image_dict - dict,
        key - slice number
        value - list of numpy.arrays containing disjoint contours (as x,y coordinates)
    """
    OM = get_fixed_orientation_matrix(row)
    xyz = get_xyz_spacing(row, restore_slice_location=True)
    pos = get_patient_position(row)[:, 1:]

    contours_image_dict = dict()

    for roi_name, roi_coordinates in contours_dict.items():
        contours_image_dict[roi_name] = dict()
        for slice_number, coordinates_list in roi_coordinates.items():
            contours_image_dict[roi_name][slice_number] = []
            for coords in coordinates_list:
                new_coords = (coords - pos[slice_number]).dot(OM) / xyz
                # After rotation 1 out of 3 coordinates will be the same for all points of the curve,
                # append only remaining coordinates.
                _s = np.zeros(new_coords.shape[0]-1)
                indx = np.where([not np.allclose(_s, ci) for ci in np.diff(new_coords, axis=0).T])[0]
                contours_image_dict[roi_name][slice_number].append(new_coords.T[indx].T)

    return contours_image_dict


def image_to_contours(row: pd.Series, name2roi: dict):
    """
    Moves image contour coordinates to initial image space. Inverse of contours_to_image.
    row: corresponds to the first slice in series (defined by InstanceNumber)
    name2roi: dict structure: {'roi_name', ([list of slice numbers], [list of coordinate pairs (x,y), in plane coords]])}
    returns: dict structure: {'roi_name', ([list of slice numbers], [[list of coordinate triplets (x,y,z)],])}
    """
    OM = get_fixed_orientation_matrix(row)
    xyz = get_xyz_spacing(row)
    pos = get_patient_position(row)[:, 1:]

    image_contours_dict = dict()

    for roi_name, roi_coordinates in name2roi.items():
        image_contours_dict[roi_name] = ([], [])
        for slice_number, coordinates_list in zip(*name2roi[roi_name]):
            coords_xyz = np.c_[coordinates_list, np.zeros(coordinates_list.shape[0])]
            image_contours_dict[roi_name][0].append(slice_number)
            image_contours_dict[roi_name][1].append(((coords_xyz * xyz).dot(np.linalg.inv(OM)) + pos[slice_number]))

    return image_contours_dict


def reformat_to_dicts(contours):
    """
    Transforms image_to_contours-like structure to contours_to_image-like structure
    """
    res = {}
    for name, data in contours.items():
        res[name] = {}
        for sl_number, c in zip(*data):
            if sl_number not in res[name]:
                res[name][sl_number] = [c,]
            else:
                res[name][sl_number].append(c)

    return res


def reformat_to_tuples(contours):
    """
    Transforms contours_to_image-like structure to image_to_contours-like structure
    """
    res = {}
    for name, data in contours.items():
        res[name] = ([],[])
        for sl, arr in data.items():
            for inner in arr:
                res[name][0].append(sl)
                res[name][1].append(inner)

    return res