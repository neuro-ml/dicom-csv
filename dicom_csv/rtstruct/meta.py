from pydicom import dcmread


def _get_contour_seq_name(rtstruct_path: str = None, rtstruct: str = None, encode: str = 'cp1252',
                          decode: str = 'cp1251', return_original=False):
    """Returns list of contour's names."""
    if rtstruct is None:
        rtstruct = dcmread(rtstruct_path)
    if return_original:
        return [roi_seq.ROIName for roi_seq in rtstruct.StructureSetROISequence]
    return [roi_seq.ROIName.encode(encode).decode(decode) for roi_seq in rtstruct.StructureSetROISequence]


def _get_series_instance_uid(rtstruct_path: str = None, rtstruct: str = None,):
    """Returns SeriesInstanceUID of an image associated with the given RTStructure."""
    if rtstruct is None:
        rtstruct = dcmread(rtstruct_path, force=True)
    return rtstruct.ReferencedFrameOfReferenceSequence[0]\
        .RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
