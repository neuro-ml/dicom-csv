import pandas as pd

from os.path import join as jp

from .meta import _get_series_instance_uid
from dicom_csv.aggregation import *

__all__ = 'collect_rtstruct'


def collect_rtstruct(df, ):
    # TODO: Add relative path option, RTSTRUCT detection
    """Extract rows related to RTSTRUCTs from  and add some columns"""
    contours = df.query('Modality == "RTSTRUCT"') \
           .drop(['NoError', 'HasPixelArray'], axis=1) \
           .dropna(how='all', axis=1)

    contours['ReferenceSeriesInstanceUID'] = contours[['PathToFolder', 'FileName']] \
        .apply(lambda x: _get_series_instance_uid(jp(x[0], x[1])), axis=1)

    temp = aggregate_images(normalize_identifiers(df))
    temp = temp[['ImageOrientationPatient0', 'ImageOrientationPatient1',
                 'ImageOrientationPatient2', 'ImageOrientationPatient3',
                 'ImageOrientationPatient4', 'ImageOrientationPatient5',
                 'PixelSpacing0', 'PixelSpacing1',
                 'InstanceNumbers', 'SOPInstanceUIDs',
                 'ImagePositionPatient0s', 'ImagePositionPatient1s',
                 'ImagePositionPatient2s', 'SeriesInstanceUID',
                 'PixelArrayShape', 'SlicesCount']]

    contours = pd.merge(contours, temp,
                        how='left',
                        left_on='ReferenceSeriesInstanceUID',
                        right_on='SeriesInstanceUID')\
                 .drop('SeriesInstanceUID_y', axis=1)
    return contours.rename(columns={'SeriesInstanceUID_x':'SeriesInstanceUID'})
