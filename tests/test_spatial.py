from pathlib import Path
import numpy as np
import pandas as pd
from pydicom import dcmread
from dicom_csv.spatial import (
    get_orientation_matrix,
    get_image_position_patient,
    get_slice_locations,
    get_image_plane,
    Plane,
    _get_slices_spacing,
    get_pixel_spacing,
    get_image_size,
    order_series
)

# TODO: add more series for diversity
SERIES = '1.2.840.113619.2.374.2807.4233243.16142.1527731842.74'


def _get_image():
    folder = Path('/home/anvar/mri_data')
    df = pd.read_csv(folder / 'mri_data.csv')
    return df.query('SeriesInstanceUID == @SERIES')


def _get_series():
    image = _get_image()
    folder = Path('/home/anvar/mri_data')
    return [dcmread(folder / file.PathToFolder / file.FileName) for _, file in image.iterrows()]


def test_get_orientation_matrix():
    image = _get_image()
    om = get_orientation_matrix(image)
    OM = np.array([0.9882921127294, 0.03687270420588, 0.14805101688742,
                  -0.0437989943104, 0.99807987034582, 0.04379749431055]).reshape(2, 3)

    assert om.shape == (3, 3)
    assert np.allclose(om[:2, :], OM, atol=1e-5)
    assert np.allclose(om[0, :] @ om[1, :], 0, atol=1e-5)


def test_get_image_position_patient():
    image = _get_image()
    pos = get_image_position_patient(image)
    assert pos.shape == (216, 3)
    # TODO: add values, e.g. pos[0] check


def test_get_slice_locations():
    image = _get_image()
    test_slice_loc = image.SliceLocation.values
    loc = get_slice_locations(image)
    order_loc = np.argsort(loc)
    order_test = np.argsort(test_slice_loc)

    assert len(loc) == 216
    assert np.allclose(order_loc, order_test)


def test_get_image_plane():
    image = _get_image()
    plane = get_image_plane(image)
    assert plane == Plane.Axial


def test_get_slice_spacing():
    image = _get_image()
    spacings = _get_slices_spacing(image)
    assert spacings.shape == (215,)
    assert np.allclose(spacings.mean(), 0.8)


def test_get_pixel_spacing():
    image = _get_image()
    xy_spacings = get_pixel_spacing(image)
    assert xy_spacings.shape == (2,)
    assert np.allclose(xy_spacings, [0.4688, 0.4688])


def test_get_image_size():
    image = _get_image()
    rows, columns, slices = get_image_size(image)
    assert (rows, columns, slices) == (512, 512, 216)


def test_order_series():
    # TODO
    image = _get_series()
    image = order_series(image)
    pass


if __name__ == '__main__':
    test_get_orientation_matrix()
    test_get_image_position_patient()
    test_get_slice_locations()
    test_get_image_plane()
    test_get_slice_spacing()
    test_get_pixel_spacing()
    test_get_image_size()
    test_order_series()