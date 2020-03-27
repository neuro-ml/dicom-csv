import os.path.join as jp

from .meta import _get_series_instance_uid

__all__ = 'collect_rtstruct'


def collect_rtstruct(df, ):
    # TODO: Add relative path option, RTSTRUCT detection
    """Extract rows related to RTSTRUCTs from  and add some columns"""
    df = df.query('Modality == "RTSTRUCT"') \
           .drop(['NoError', 'HasPixelArray'], axis=1) \
           .dropna(how='all', axis=1)

    df['ReferenceSeriesInstanceUID'] = df[['PathToFolder', 'FileName']] \
        .apply(lambda x: _get_series_instance_uid(jp(x[0], x[1])), axis=1)
    return df
