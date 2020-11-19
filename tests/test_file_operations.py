from functools import lru_cache

import numpy as np
import pytest

from dicom_csv import join_tree, aggregate_images, load_series


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

    x = load_series(images.loc[0], base_path, orientation=True)
    y = load_series(sorted_images.loc[0], base_path, orientation=True)
    np.testing.assert_array_almost_equal(x, y)
