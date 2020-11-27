from functools import wraps
import pandas as pd
from dicom_csv.utils import split_floats

from dicom_csv.crawler import SERIAL


def csv_instance(func):
    @wraps(func)
    def wrapper(instance, *args, **kwargs):
        if isinstance(instance, pd.Series):
            instance = SeriesWrapper(instance)
        return func(instance, *args, **kwargs)

    return wrapper


def csv_series(func):
    @wraps(func)
    def wrapper(series, *args, **kwargs):
        if isinstance(series, pd.Series):
            # make sure that this is an aggregated row
            assert 'FileNames' in series
            assert 'FileName' not in series
            # unpack it
            # TODO: need a better conversion between aggregated and plain meta
            files = series.FileNames.split('/')
            fields = {
                'FileName': files,
            }
            for key in ['ImagePositionPatient0', 'ImagePositionPatient1', 'ImagePositionPatient2', 'InstanceNumber']:
                col = f'{key}s'
                if col in series:
                    fields[key] = split_floats(series[col])

            series = pd.DataFrame([series] * len(files)).copy().drop([f'{col}s' for col in fields], axis=1)
            for col, values in fields.items():
                series[col] = values

        if isinstance(series, pd.DataFrame):
            series = DataframeWrapper(series)

        return func(series, *args, **kwargs)
    return wrapper


def out_csv(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, CSVWrapper):
            # single result
            return result.unwrap()

        if isinstance(result, (tuple, list)) and any(isinstance(x, CSVWrapper) for x in result):
            rows = []
            for entry in result:
                assert isinstance(entry, CSVWrapper), entry
                assert not isinstance(entry, DataframeWrapper)
                rows.append(entry.unwrap())

            return pd.concat(rows, 1).T

        return result

    return wrapper


class CSVWrapper:
    def unwrap(self):
        raise NotImplementedError


class DataframeWrapper(CSVWrapper):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def unwrap(self):
        return self.dataframe

    def __len__(self):
        return len(self.dataframe)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index):
        assert index < len(self)
        return RowIndex(self.dataframe, index)


class RowIndex(CSVWrapper):
    def __init__(self, dataframe, index):
        self.index = index
        self.dataframe = dataframe

    def _row(self):
        return self.dataframe.iloc[self.index]

    def unwrap(self):
        return self._row()

    def __getattr__(self, item):
        return _get_field(self._row(), item)


class SeriesWrapper(CSVWrapper):
    def __init__(self, row: pd.Series):
        self.row = row

    def unwrap(self):
        return self.row

    def __getattr__(self, item):
        return _get_field(self.row, item)


def _get_field(row, name):
    if name in SERIAL:
        result = []
        idx = 0
        indexed = f'{name}{idx}'
        while indexed in row:
            result.append(getattr(row, indexed))
            idx += 1
            indexed = f'{name}{idx}'

        return result

    return getattr(row, name)
