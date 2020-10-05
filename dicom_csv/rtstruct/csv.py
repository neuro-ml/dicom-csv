import pandas as pd
from os.path import join as jp
from dicom_csv.rtstruct.meta import _get_series_instance_uid
from dicom_csv.aggregation import *

__all__ = 'collect_rtstruct'

by = ('PatientID', 'SeriesInstanceUID', 'StudyInstanceUID', 'PathToFolder')
IMAGE_COLUMNS = ['ImageOrientationPatient0', 'ImageOrientationPatient1',
                 'ImageOrientationPatient2', 'ImageOrientationPatient3',
                 'ImageOrientationPatient4', 'ImageOrientationPatient5',
                 'PixelSpacing0', 'PixelSpacing1',
                 'InstanceNumbers', 'SOPInstanceUIDs',
                 'ImagePositionPatient0s', 'ImagePositionPatient1s',
                 'ImagePositionPatient2s', 'SeriesInstanceUID',
                 'PixelArrayShape', 'SlicesCount', 'SliceLocations']


def collect_rtstruct(df, by=by):
    # TODO: Add relative path option
    # TODO: better RTSTRUCT detection
    # TODO: rewrite this
    """Extract rows related to RTSTRUCTs from  and add some columns"""
    contours = df.query('Modality == "RTSTRUCT"').drop(['NoError', 'HasPixelArray'], axis=1).dropna(how='all', axis=1)

    contours['ReferenceSeriesInstanceUID'] = contours[['PathToFolder', 'FileName']] \
        .apply(lambda x: _get_series_instance_uid(jp(x[0], x[1])), axis=1)

    images = aggregate_images(df.fillna('-'), by=by)[IMAGE_COLUMNS]

    contours = pd.merge(contours, images, how='left', left_on='ReferenceSeriesInstanceUID', right_on='SeriesInstanceUID')\
                 .drop('SeriesInstanceUID_y', axis=1)
    return contours.rename(columns={'SeriesInstanceUID_x':'SeriesInstanceUID'})
