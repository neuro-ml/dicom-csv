from io import BytesIO
from pathlib import Path
from typing import Union, Sequence

import pandas as pd
from pydicom import Dataset, dcmread, dcmwrite
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian
from dicom_csv.utils import Series
from dicom_csv.spatial import get_voxel_spacing, get_orientation_matrix, get_image_position_patient
from nibabel import Nifti1Header, Nifti1Image

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


def _get_nifti_header(shape: tuple):
    header = Nifti1Header()
    header.set_data_shape(shape)
    header.set_dim_info(slice=2)
    header.set_xyzt_units('mm')
    return header


def _get_affine(om: np.ndarray, pos: list, voxel: list):
    voxel = np.diag(voxel)
    OM = np.eye(4)
    om = om @ voxel
    OM[:3, :3]= om
    OM[:3, 3] = pos
    return OM


def get_nifti(series: Series, mask: np.ndarray=None):
    """
    Construct NIFTI image from list of DICOMs.
    """
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
