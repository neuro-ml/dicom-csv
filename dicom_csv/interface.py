from functools import wraps
import pandas as pd
from dicom_csv.crawler import SERIAL


def csv_instance(func):
    @wraps(func)
    def wrapper(instance, *args, **kwargs):
        if isinstance(instance, pd.Series):
            instance = RowWrapper(instance)

        return func(instance, *args, **kwargs)

    return wrapper


def csv_series(func):
    @wraps(func)
    def wrapper(series, *args, **kwargs):
        if isinstance(series, pd.DataFrame):
            series = DataframeWrapper(series)

        return func(series, *args, **kwargs)

    return wrapper


class DataframeWrapper:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index):
        assert index < len(self)
        return RowIndex(self.dataframe, index)


class RowIndex:
    def __init__(self, dataframe, index):
        self.index = index
        self.dataframe = dataframe

    def __getattr__(self, item):
        return _get_field(self.dataframe.iloc[self.index], item)


class RowWrapper:
    def __init__(self, row: pd.Series):
        self.row = row

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
