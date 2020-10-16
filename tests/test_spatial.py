from dicom_csv import join_tree
from pathlib import Path
import numpy as np
from dicom_csv.spatial import get_orientation_matrix, get_image_position_patient

SERIES = '1.2.840.113619.2.374.2807.4233243.16142.1527731842.74'


def test_get_orientation_matrix():
    om = get_orientation_matrix(image)
    OM = np.array([0.9882921127294, 0.03687270420588, 0.14805101688742,
                  -0.0437989943104, 0.99807987034582, 0.04379749431055]).reshape(2, 3)
    assert om.shape == (3, 3)
    assert np.allclose(om[:2, :], OM, atol=1e-5)


def test_get_image_position_patient():
    pos = get_image_position_patient(image)
    assert pos.shape == (216, 3)
    # TODO: add values, e.g. pos[0] check


if __name__ == '__main__':
    folder = Path('/home/anvar/mri_data')
    df = join_tree(folder, ignore_extensions=['.ipynb'])
    image = df.query('SeriesInstanceUID == @SERIES')

    test_get_orientation_matrix()
    test_get_image_position_patient()