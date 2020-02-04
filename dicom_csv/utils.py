from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[Path, str]
ORIENTATION = [f'ImageOrientationPatient{i}' for i in range(6)]


def split_floats(string):
    return list(map(float, string.split(',')))


def contains_info(row, *cols):
    return all(col in row and pd.notnull(row[col]) for col in cols)


def extract_dims(x):
    assert len(x) == 1, len(x)
    return x[0]


def zip_equal(*args):
    assert not args or all(len(args[0]) == len(x) for x in args)
    return zip(*args)
