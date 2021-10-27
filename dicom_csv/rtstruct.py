import pydicom
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from skimage.draw import polygon
from dicom_csv import order_series
from dicom_csv.utils import Series
from dicom_csv.spatial import (get_orientation_matrix, get_voxel_spacing,
                               get_image_position_patient, get_image_size,
                              get_slices_plane, Plane)

# TODO: fix logic -> read -> move to voxel space -> move to mask
# TODO: Consider only work with 2D coords, completely dropping the third column
# See https://dicom.innolitics.com/ciods/rt-structure-set/roi-contour/30060039/30060040/30060050 for details

def _update_dict_list_key(d, key, value):
    """Modifies input dictionary inplace."""
    if key not in d:
        d[key] = [value]
    else:
        d[key].append(value)
        
        
@dataclass
class Contour:
    contour_name: tuple
    contour_data: dict
    reference_series_instance_uid: str
    coordinate_space_patient: bool=True
    image_shape: tuple=None
    image_position_patient: dict=None
    image_plane: Plane=None
        
    
    def _get_coordinate_indices(self):
        if self.image_plane == Plane.Axial:
            a, b = 0, 1
        elif self.image_plane == Plane.Sagittal:
            a, b = 0, 1
        elif self.image_plane == Plane.Coronal:
            a, b = 1, 2 # TODO
        else:
            raise ValueError
        return a, b
    
    
    def get_mask(self, ) -> np.ndarray:
        """Converts image conours to 3D mask."""
        if self.coordinate_space_patient:
            raise AttributeError(f'Coordinates must be in image coordinate space.')
        
        a, b = self._get_coordinate_indices()
        mask = np.zeros(self.image_shape)
        for i, uid in enumerate(self.image_position_patient):
            if uid in self.contour_data.keys():
                contours = self.contour_data[uid]
                for contour in contours:
                    y, x = polygon(np.abs(contour[:, a]), np.abs(contour[:, b]), self.image_shape[:2])
                    mask[x, y, i] = 1
        return mask
    
    
def get_contour_seq_name(rtstruct: pydicom.dataset.Dataset,
                         encode: str='cp1252', decode: str='cp1251') -> tuple:
    """Recall, that some of the contours are not actually contained in the RTStruct."""
    return (roi_seq.ROIName.encode(encode).decode(decode) for roi_seq in rtstruct.StructureSetROISequence)


def get_reference_series_instance_uid(rtstruct: pydicom.dataset.Dataset) -> str:
    return rtstruct.ReferencedFrameOfReferenceSequence[0]\
        .RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID


def read_contour_sequence(dataset: pydicom.dataset.Dataset) -> dict:
    """Reads coordinate data for a single contour object:
    pydicom.dataset.Dataset.ROIContourSequence[i].ContourSequence[j]"""
    contours = dict()
    
    if not hasattr(dataset, 'ContourSequence'):
        raise AttributeError('Dataset does not have Contour Sequence.')
    
    for image_slice in dataset.ContourSequence:
        sop_instance_uid = image_slice.ContourImageSequence[0].ReferencedSOPInstanceUID
        coords = np.array(image_slice.ContourData)
        n = len(coords) // 3
        coords = coords.reshape((n, 3))
        _update_dict_list_key(contours, sop_instance_uid, coords)
    return contours


def read_rtstruct(rtstruct: pydicom.dataset.Dataset) -> dict:
    """Reads content of RTStructure."""
    reference_series_uid = get_reference_series_instance_uid(rtstruct)
    contours_sequence = list(rtstruct.ROIContourSequence)
    roi_names = get_contour_seq_name(rtstruct)
    contours = dict()
        
    for i, (roi_name, roi_contour) in enumerate(zip(roi_names, contours_sequence)):
        try:
            coords = read_contour_sequence(roi_contour)
            contour = Contour((roi_name, i), coords, reference_series_uid)
            contours[(roi_name, i)] = contour
        except AttributeError:
            pass
    return contours
        
        
def _contour_to_image(contours_patient: Contour, orientation_matrix: np.ndarray,
                     voxel_spacing: np.ndarray, image_position_patient: dict, image_size:tuple,
                     image_plane: Plane): 
    """Moves contours coordinates to image space."""    
    contours_image = dict()
    # Order SOPInstanceUID in contour keys according to input series order
    for uid in image_position_patient.keys():
        if uid in contours_patient.contour_data.keys():
            slice_contour = contours_patient.contour_data[uid]
            for coords in slice_contour:
                coords_image = (coords - image_position_patient[uid]) @ orientation_matrix.T / voxel_spacing
                _update_dict_list_key(contours_image, uid, coords_image)
    return Contour(contour_data=contours_image, contour_name=contours_patient.contour_name,
        reference_series_instance_uid=contours_patient.reference_series_instance_uid,
        coordinate_space_patient=False, image_shape=image_size, image_position_patient=image_position_patient,
        image_plane=image_plane)


def contours_to_image(series: Series, rtstruct: pydicom.dataset.Dataset):
    """
    Read contours from rtstruct and move them into patient coordinate space.
    """
    series = order_series(series)
    sop_uids = [image.SOPInstanceUID for image in series]
    
    om = get_orientation_matrix(series)
    dx_dy_dz = np.array(get_voxel_spacing(series))
    image_size = get_image_size(series)
    pos = get_image_position_patient(series)
    image_plane = get_slices_plane(series)
    
    # TODO: Consider only work with 2D coords, completely dropping the third column
    pos = dict(zip(sop_uids, pos))
    
    contours_patient = read_rtstruct(rtstruct) # potential AttributeError if no ContourSequence
    contours_image = dict()
    for key, val in contours_patient.items():
        try:
            contours_image[key] = _contour_to_image(val, om, dx_dy_dz, pos, image_size, image_plane)
        except Exception as e:
            print(f'{e}, SeriesInstanceUID: {val.reference_series_instance_uid}, ContourName: {key}')
    return contours_image