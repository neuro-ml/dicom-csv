import numpy as np
import pandas as pd
import pytest
from pydicom import dcmread

from dicom_csv.spatial import (
    get_orientation_matrix,
    get_image_position_patient,
    get_slice_locations,
    Plane,
    get_slices_plane,
    _get_slices_deltas,
    get_pixel_spacing,
    get_image_size,
    order_series
)


@pytest.fixture
def image(tests_folder):
    df = pd.read_csv(tests_folder / 'spatial/mri_data.csv')
    # TODO: add more series for diversity
    SERIES = '1.2.840.113619.2.374.2807.4233243.16142.1527731842.74'
    return df.query('SeriesInstanceUID == @SERIES')


@pytest.fixture
def series(tests_folder):
    return [dcmread(tests_folder / 'spatial' / file.PathToFolder / file.FileName) for _, file in image.iterrows()]


def test_get_orientation_matrix(image):
    om = get_orientation_matrix(image)
    target = np.array([0.9882921127294, 0.03687270420588, 0.14805101688742,
                       -0.0437989943104, 0.99807987034582, 0.04379749431055]).reshape(2, 3)

    assert om.shape == (3, 3)
    assert np.allclose(om[:2, :], target, atol=1e-5)
    assert np.allclose(om[0, :] @ om[1, :], 0, atol=1e-5)


def test_get_image_position_patient(image):
    pos = get_image_position_patient(image)
    assert pos.shape == (216, 3)
    # TODO: add values, e.g. pos[0] check


def test_get_slice_locations(image):
    test_slice_loc = image.SliceLocation.values
    loc = get_slice_locations(image)
    order_loc = np.argsort(loc)
    order_test = np.argsort(test_slice_loc)

    assert len(loc) == 216
    assert np.allclose(order_loc, order_test)


def test_get_image_plane(image):
    plane = get_slices_plane(image)
    assert plane == Plane.Axial


def test_get_slice_spacing(image):
    spacings = _get_slices_deltas(image)
    assert spacings.shape == (215,)
    assert np.allclose(spacings.mean(), 0.8)


def test_get_pixel_spacing(image):
    xy_spacings = get_pixel_spacing(image)
    assert xy_spacings.shape == (2,)
    assert np.allclose(xy_spacings, [0.4688, 0.4688])


def test_get_image_size(image):
    rows, columns, slices = get_image_size(image)
    assert (rows, columns, slices) == (512, 512, 216)


@pytest.mark.skip
def test_order_series(series):
    series = order_series(series)
