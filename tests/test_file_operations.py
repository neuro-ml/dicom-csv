import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from dicom_csv import join_tree, aggregate_images, order_series, stack_images
from dicom_csv.misc import get_image


def load_series(row, base_path):
    dicoms = [pydicom.dcmread(Path(base_path) / row.PathToFolder / file) for file in row.FileNames.split('/')]
    dicoms = order_series(dicoms)
    return stack_images(dicoms, -1)


def _make_synthetic_slice(filepath, rows, cols, slice_index, study_uid, series_uid):
    """Write one minimal CT-like DICOM slice (16-bit, uncompressed) for testing."""
    pixel_array = np.zeros((rows, cols), dtype=np.int16)
    pixel_array[:] = -1000 + slice_index * 2
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(filepath), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "CT"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 1.0
    ds.ImagePositionPatient = [0.0, 0.0, float(slice_index)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelData = pixel_array.tobytes()
    ds.save_as(filepath, enforce_file_format=True)


@pytest.fixture
@lru_cache()
def meta(tests_folder):
    return join_tree(tests_folder / 'crawler', relative=True, verbose=False, force=False)


def test_crawler(meta):
    size = len(meta)
    assert size == 3426
    assert meta.NoError.sum() == size - 1
    # assert meta.HasPixelArray.sum() == size - 1

    bad_row = meta.loc[~meta.NoError].iloc[0]
    assert bad_row.PathToFolder == '.'
    assert bad_row.FileName == 'file-with-error.txt'

    meta = meta.loc[meta.NoError]
    assert len(meta.StudyInstanceUID.unique()) == 10
    assert len(meta.SeriesInstanceUID.unique()) == 14


def test_aggregation(meta, tests_folder):
    base_path = tests_folder / 'crawler'
    meta = meta[meta.NoError]
    by = ['SeriesInstanceUID']

    images = aggregate_images(meta, by)
    sorted_images = aggregate_images(meta, by, process_series=lambda series: series.sort_values('FileName'))
    assert len(images) == len(sorted_images) == 14

    x = load_series(images.loc[0], base_path)
    y = load_series(sorted_images.loc[0], base_path)
    np.testing.assert_array_almost_equal(x, y)


def test_stack_images_matches_np_stack_for_all_axes():
    """stack_images(series, axis=k) must match np.stack([get_image(ds) for ds in series], axis=k) for k in (-1, 0, 1, 2)."""
    n_slices, rows, cols = 5, 32, 32
    study_uid = generate_uid()
    series_uid = generate_uid()
    with tempfile.TemporaryDirectory(prefix="dicom_csv_test_") as tmp:
        tmp_path = Path(tmp)
        for i in range(n_slices):
            _make_synthetic_slice(
                tmp_path / f"slice_{i:04d}.dcm",
                rows, cols, i, study_uid, series_uid,
            )
        paths = sorted(tmp_path.glob("*.dcm"))
        series = [pydicom.dcmread(str(p)) for p in paths]
        for axis in (-1, 0, 1, 2):
            expected = np.stack([get_image(ds) for ds in series], axis=axis)
            actual = stack_images(series, axis=axis)
            np.testing.assert_array_equal(
                actual, expected,
                err_msg=f"stack_images(series, axis={axis}) should match np.stack([get_image(ds) for ds in series], axis={axis})",
            )
