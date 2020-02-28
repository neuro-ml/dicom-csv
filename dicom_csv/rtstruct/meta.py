from pydicom import dcmread


def _get_contour_seq_name(rtstruct_path: str = None,
                          rtstruct: str = None,
                          encode: str = 'cp1252',
                          decode: str = 'cp1251'):
    if rtstruct is None:
        rtstruct = dcmread(rtstruct_path)
    return [roi_seq.ROIName.encode(encode).decode(decode) for roi_seq in rtstruct.StructureSetROISequence]


def _get_series_instance_uid(rtstruct_path):
    dicom = dcmread(rtstruct_path)
    return dicom.ReferencedFrameOfReferenceSequence[0]\
        .RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
