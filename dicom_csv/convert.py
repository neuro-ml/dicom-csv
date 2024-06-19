import logging
import warnings

from pydicom.uid import generate_uid

from .exceptions import TagMissingError
from .utils import Instance, Instances, Series, bufferize_instance, collect, set_file_meta


logger = logging.getLogger(__name__)


@collect
def expand_volumetric(instances: Instances) -> Instances:
    """Returns the incoming sequence of instances but with each volumetric instance expanded into a series."""
    for instance in instances:
        if is_volumetric_ct(instance):
            logger.info(f'Expanding volumetric series: {instance.SeriesInstanceUID}')
            yield from split_volume(instance)
        else:
            yield instance


def is_volumetric_ct(instance: Instance, errors: bool = True) -> bool:
    """Checks if the input Dataset is an Enhanced CT Image Storage (volumetric image)."""
    try:
        return instance.SOPClassUID == '1.2.840.10008.5.1.4.1.1.2.1'
    except AttributeError:
        if errors:
            raise
        return False


@collect
def split_volume(instance: Instance) -> Series:
    """Splits volumetric (EnchancedCTImageStorage) instance into separate frames."""
    if not is_volumetric_ct(instance):
        raise ValueError('The instance is not volumetric.')

    instance = bufferize_instance(instance)
    if not hasattr(instance, 'pixel_array'):
        raise TagMissingError('PixelData')
    if not hasattr(instance, 'PerFrameFunctionalGroupsSequence'):
        raise TagMissingError('PerFrameFunctionalGroupsSequence')
    if not hasattr(instance, 'SharedFunctionalGroupsSequence'):
        raise TagMissingError('SharedFunctionalGroupsSequence')

    pixel_array = instance.pixel_array
    frames_sequence = instance.PerFrameFunctionalGroupsSequence
    shared_tags = _get_shared_tags(instance.SharedFunctionalGroupsSequence[0])
    default_frame = _get_default_frame(instance, shared_tags)
    for i, (image, frame) in enumerate(zip(pixel_array, frames_sequence), 1):
        yield _set_frame_specific_tags(frame, image, i, default_frame)


def _exclude_callback(dataset, data_element):
    warnings.warn('depricate, do not use')
    if data_element.tag in EXCLUDE_TAGS:
        del dataset[data_element.tag]


def depricate_get_default_frame(instance, shared_tags):
    warnings.warn('depricate, do not use')
    """Extracts a metadata from volumetric enhanced instance, drops heavy tags."""
    # TODO: is it ok?
    default_frame = instance  # _bufferize_instance(instance3d)
    default_frame.walk(_exclude_callback)
    for tag in shared_tags:
        default_frame.add(tag)
    return default_frame


def delete_tags(instance, tags):
    for tag in tags:
        if tag in instance:
            del instance[tag]


def _get_default_frame(instance, shared_tags):
    """Extracts a metadata from volumetric enhanced instance, drops heavy tags."""
    delete_tags(instance, EXCLUDE_TAGS.keys())
    for tag in shared_tags:
        instance.add(tag)
    return instance


def _get_shared_tags(shared_tags):
    """Returns dict of tags shared amongst all frames."""
    # TODO Add others required tags
    # TODO Check Sequences existence
    return [
        *shared_tags.PixelValueTransformationSequence[0].iterall(),
        *shared_tags.PixelMeasuresSequence[0].iterall(),
        *shared_tags.CTReconstructionSequence[0].iterall(),
        *shared_tags.FrameVOILUTSequence[0].iterall(),
    ]


def _set_frame_specific_tags(frame, pixel_array, frame_number, default_frame):
    """Initialize all frame specific tags."""
    instance = bufferize_instance(default_frame)
    instance.ImagePositionPatient = frame.PlanePositionSequence[0].ImagePositionPatient
    instance.ImageOrientationPatient = frame.PlaneOrientationSequence[0].ImageOrientationPatient
    instance.PixelData = pixel_array.tobytes()
    instance.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    # TODO: should use an external generator?
    instance.SOPInstanceUID = generate_uid()
    instance.InstanceNumber = str(frame_number)
    set_file_meta(instance)
    instance.filename = None  # some problems with deepcopy https://github.com/pydicom/pydicom/issues/1147
    return instance


# excluded tags derived from
# https://github.com/dcm4che/dcm4che/blob/master/dcm4che-emf/src/main/java/org/dcm4che3/emf/MultiframeExtractor.java
EXCLUDE_TAGS = {
    ('0008', '9092'): 'ReferencedImageEvidenceSequence',
    ('0008', '9154'): 'SourceImageEvidenceSequence',
    ('0020', '9222'): 'DimensionIndexSequence',
    ('0028', '0008'): 'NumberOfFrames',
    ('5200', '9229'): 'SharedFunctionalGroupsSequence',
    ('5200', '9230'): 'PerFrameFunctionalGroupsSequence',
    ('7fe0', '0010'): 'PixelData',
}
