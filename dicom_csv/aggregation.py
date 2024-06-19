"""Tools for grouping DICOM metadata into images."""
from typing import Callable, Sequence, Union

import numpy as np
import pandas as pd

from .misc import stack_images
from .spatial import get_image_position_patient, get_orientation_matrix, get_voxel_spacing, order_series
from .utils import Series


__all__ = 'aggregate_images', 'normalize_identifiers', 'select'


def _remove_dots(x):
    try:
        return str(int(float(x)))
    except ValueError:
        return x


def aggregate_images(metadata: pd.DataFrame, by: Union[str, Sequence[str]],
                     process_series: Callable = None) -> pd.DataFrame:
    """
    Groups DICOM ``metadata`` into images (series).

    Parameters
    ----------
    metadata
        a dataframe with metadata returned by `join_tree`.
    by
        a list of column names by which the grouping will be performed.
        Default columns are: PatientID, SeriesInstanceUID, StudyInstanceUID,
        PathToFolder, PixelArrayShape, SequenceName.
    process_series
        a function that processes an aggregated series before it will be joined into a single entry

    References
    ----------
    See the :doc:`tutorials/dicom` tutorial for more details.

    Notes
    -----
    The following columns are added:
        | SlicesCount: the number of files/slices in the image.
        | FileNames: a list of slash ("/") separated file names.
        | InstanceNumbers: (if InstanceNumber is in columns) a list of comma separated InstanceNumber values.

    The following columns are removed:
        FileName (replaced by FileNames), InstanceNumber (replaced by InstanceNumbers),
        any other columns that differ from file to file.
    """

    def get_unique_cols(df):
        # TODO: deal with float precision errors
        return [col for col in df.columns if len(df[col].dropna().unique()) == 1]

    def process_group(entry):
        if process_series is not None:
            entry = process_series(entry)

        res = entry.iloc[[0]][get_unique_cols(entry)]
        res['FileNames'] = '/'.join(entry.FileName)
        res['SlicesCount'] = len(entry)
        # TODO: move the saved fields to arguments
        try:
            res['InstanceNumbers'] = ','.join(map(_remove_dots, entry.InstanceNumber))
        except (ValueError, TypeError):
            res['InstanceNumbers'] = None
        if 'SliceLocation' in entry:
            res['SliceLocations'] = ','.join(entry.SliceLocation.astype(str))
        for position in ['ImagePositionPatient0', 'ImagePositionPatient1', 'ImagePositionPatient2']:
            if position in entry:
                res[f'{position}s'] = ','.join(entry[position].astype(str))
        if 'SOPInstanceUID' in entry:
            res['SOPInstanceUIDs'] = ','.join(entry.SOPInstanceUID.astype(str))

        return res.drop(['FileName'], axis=1, errors='ignore')

    if isinstance(by, str):
        by = [by]
    else:
        by = list(by)

    not_string = metadata[by].applymap(lambda x: not isinstance(x, str)).any()
    if not_string.any():
        not_strings = ', '.join(not_string.index[not_string])
        raise ValueError(f'The following columns do not contain only strings: {not_strings}. '
                         'You should probably check for NaN values.')

    return metadata.groupby(by).apply(process_group).reset_index(drop=True)


def normalize_identifiers(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Converts PatientID to str and fills nan values in SequenceName.

    Notes
    -----
    The input dataframe will be mutated.
    """
    metadata['PatientID'] = metadata.PatientID.apply(_remove_dots)
    if 'SequenceName' in metadata:
        metadata.SequenceName.fillna('', inplace=True)
    return metadata


def select(dataframe: pd.DataFrame, query: str, **where: str) -> pd.DataFrame:
    query = ' '.join(query.format(**where).splitlines())
    return dataframe.query(query).dropna(axis=1, how='all').dropna(axis=0, how='all')


def _get_nifti_header(shape: tuple):
    from nibabel import Nifti1Header

    header = Nifti1Header()
    header.set_data_shape(shape)
    header.set_dim_info(slice=2)
    header.set_xyzt_units('mm')
    return header


def _get_affine(om: np.ndarray, pos: list, voxel: list):
    voxel = np.diag(voxel)
    OM = np.eye(4)
    om = om @ voxel
    OM[:3, :3] = om
    OM[:3, 3] = pos
    return OM


def get_nifti(series: Series, mask: np.ndarray = None):
    """
    Construct NIFTI image from list of DICOMs.
    """
    from nibabel import Nifti1Image

    series = order_series(series)
    image = stack_images(series)
    om = get_orientation_matrix(series)
    pos = list(get_image_position_patient(series)[0])
    voxel = list(get_voxel_spacing(series))
    affine = _get_affine(om, pos, voxel)
    header = _get_nifti_header(image.shape)
    if mask is None:
        return Nifti1Image(image, affine, header=header)
    return Nifti1Image(image, affine, header=header), Nifti1Image(mask, affine, header=header)
