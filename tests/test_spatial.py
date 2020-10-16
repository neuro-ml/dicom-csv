from dicom_csv import join_tree
from pathlib import Path
import numpy as np
from dicom_csv.spatial import (
    get_orientation_matrix,
    get_image_position_patient,
    get_slice_locations,
    get_image_plane,
    Plane,
    _get_slices_spacing,
    get_pixel_spacing,
    get_image_size
)

# TODO: add more series for diversity
SERIES = '1.2.840.113619.2.374.2807.4233243.16142.1527731842.74'


def test_get_orientation_matrix():
    om = get_orientation_matrix(image)
    OM = np.array([0.9882921127294, 0.03687270420588, 0.14805101688742,
                  -0.0437989943104, 0.99807987034582, 0.04379749431055]).reshape(2, 3)

    assert om.shape == (3, 3)
    assert np.allclose(om[:2, :], OM, atol=1e-5)
    assert np.allclose(om[0, :] @ om[1, :], 0, atol=1e-5)


def test_get_image_position_patient():
    pos = get_image_position_patient(image)
    assert pos.shape == (216, 3)
    # TODO: add values, e.g. pos[0] check


def test_get_slice_locations():
    loc = get_slice_locations(image)
    order_loc = np.argsort(loc)
    order_test = np.argsort(test_slice_loc)

    assert len(loc) == 216
    assert np.allclose(order_loc, order_test)


def test_get_image_plane():
    plane = get_image_plane(image)
    assert plane == Plane.Axial


def test_get_slice_spacing():
    spacings = _get_slices_spacing(image)
    assert spacings.shape == (215,)
    assert np.allclose(spacings.mean(), 0.8)


def test_get_pixel_spacing():
    xy_spacings = get_pixel_spacing(image)
    assert xy_spacings.shape == (2,)
    assert np.allclose(xy_spacings, [0.4688, 0.4688])


def test_get_image_size():
    rows, columns, slices = get_image_size(image)
    assert (rows, columns, slices) == (512, 512, 216)


if __name__ == '__main__':
    folder = Path('/home/anvar/mri_data')
    df = join_tree(folder, ignore_extensions=['.ipynb'])
    image = df.query('SeriesInstanceUID == @SERIES')

    test_get_orientation_matrix()
    test_get_image_position_patient()

    test_slice_loc = image.SliceLocation.values
    test_get_slice_locations()
    test_get_image_plane()
    test_get_slice_spacing()
    test_get_pixel_spacing()
    test_get_image_size()