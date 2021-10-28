from io import BytesIO
from pathlib import Path
from typing import Union, Sequence

import pandas as pd
from pydicom import Dataset, dcmread, dcmwrite
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian


PathLike = Union[Path, str]
ORIENTATION = [f'ImageOrientationPatient{i}' for i in range(6)]

Instance = Dataset
Series = Instances = Sequence[Dataset]


def split_floats(string, sep=','):
    return list(map(float, string.split(sep)))


def split_ints(string, sep=','):
    return list(map(int, string.split(sep)))


def contains_info(row, *cols):
    return all(col in row and pd.notnull(row[col]) for col in cols)


def extract_dims(x):
    assert len(x) == 1, len(x)
    return x[0]


def zip_equal(*args):
    assert not args or all(len(args[0]) == len(x) for x in args)
    return zip(*args)


def bufferize_instance(instance: Dataset, force=True, write_like_original=True):
    """Makes a copy of the ``instance``. Faster than deepcopy."""
    with BytesIO() as buffer:
        dcmwrite(buffer, instance, write_like_original=write_like_original)
        buffer.seek(0)
        return dcmread(buffer, force=force)


def set_file_meta(instance: Dataset):
    meta = FileMetaDataset()
    # meta.FileMetaInformationGroupLength = ??
    # meta.FileMetaInformationVersion = b'\x00\x01'
    meta.MediaStorageSOPClassUID = instance.SOPClassUID
    meta.MediaStorageSOPInstanceUID = instance.SOPInstanceUID
    meta.TransferSyntaxUID = ImplicitVRLittleEndian
    instance.is_implicit_VR = instance.is_little_endian = True
    # meta.ImplementationClassUID = ??
    # meta.ImplementationVersionName = ??
    instance.file_meta = meta
    instance.preamble = b'\x00' * 128


def collect(func):
    return lambda *args, **kwargs: list(func(*args, **kwargs))
