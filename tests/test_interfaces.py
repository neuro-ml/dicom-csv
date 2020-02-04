from pathlib import Path
import numpy as np

from dicom_csv import join_dicom_tree, aggregate_images, load_series

base_path = Path('~/dicom_data').expanduser()


def get_meta():
    return join_dicom_tree(base_path, relative=True, verbose=False)


def test_crawler():
    meta = get_meta()
    size = len(meta)
    assert size == 2588
    assert meta.NoError.sum() == size - 1
    assert meta.HasPixelArray.sum() == size - 1

    bad_row = meta.loc[~meta.NoError].iloc[0]
    assert bad_row.PathToFolder == '.'
    assert bad_row.FileName == 'readme.txt'


def test_aggregation():
    meta = get_meta()
    meta = meta[meta.NoError]
    by = ('PatientID', 'SeriesInstanceUID', 'StudyInstanceUID', 'PathToFolder', 'PixelArrayShape')

    images = aggregate_images(meta, by)
    sorted_images = aggregate_images(meta, by, process_series=lambda series: series.sort_values('FileName'))
    assert len(images) == len(sorted_images) == 4

    x = load_series(images.loc[0], base_path)
    y = load_series(sorted_images.loc[0], base_path)
    np.testing.assert_array_almost_equal(x, y)
