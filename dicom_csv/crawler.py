"""Contains functions for gathering metadata from individual DICOM files or entire directories."""
import logging
import os
import struct
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from pydicom import DataElement, Dataset, dcmread, errors, sequence, valuerep
from pydicom.uid import ImplicitVRLittleEndian
from tqdm import tqdm

from .convert import is_volumetric_ct, split_volume
from .utils import PathLike


__all__ = 'get_file_meta', 'join_tree'

SERIAL = {'ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing'}
PERSON_CLASS = valuerep.PersonName

logger = logging.getLogger(__name__)


def _throw(e):
    raise e


def read_dicom(path: PathLike, force: bool = False):
    try:
        return True, dcmread(str(path))
    except errors.InvalidDicomError:
        if force:
            dc = dcmread(str(path), force=True)
            dc.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
            return True, dc

        raise


def iter_private_tags(ds: Dataset) -> Iterable[DataElement]:
    ds.__repr__() # https://github.com/pydicom/pydicom/issues/1805
    for tag in ds.values():
        if tag.is_private:
            yield tag


def get_file_meta(path: PathLike, force: bool = True, read_pixel_array: bool = False,
                  unpack_volumetric: bool = False, extract_private: bool = False) -> Iterable[dict]:
    """
    Get a dict containing the metadata from the DICOM file located at ``path``.

    Parameters
    ---

    path - PathLike,
        full path to file

    force - bool,
        pydicom.filereader.dcmread force parameter, default is False

    read_pixel_array - bool,
        if True, crawler will add information about DICOM pixel_array, False significantly increases crawling time,
        default is True.

    Notes
    ---
    The following keys are added:
        | NoError: whether an exception was raised during reading the file.
        | HasPixelArray: (if NoError is True) whether the file contains a pixel array.
        | PixelArrayShape: (if HasPixelArray is True) the shape of the pixel array.


    For some formats the following packages might be required:
        >>> conda install -c glueviz gdcm # Python 3.5 and 3.6
        >>> conda install -c conda-forge gdcm # Python 3.7
    """
    try:
        no_error, instance = read_dicom(path, force)
    except (errors.InvalidDicomError, struct.error, OSError, NotImplementedError, AttributeError, KeyError):
        yield {'NoError': False}
        return

    if unpack_volumetric and is_volumetric_ct(instance, errors=False):
        instances = split_volume(instance)
    else:
        instances = [instance]

    for instance in instances:
        result = extract_meta(instance, read_pixel_array, extract_private)
        result.setdefault('NoError', True)
        yield result


def extract_meta(instance: Dataset, read_pixel_array: bool = False, extract_private: bool = False) -> dict:
    result = {}
    if read_pixel_array:
        try:
            has_px = hasattr(instance, 'pixel_array')
        except (TypeError, NotImplementedError):
            has_px = False
        except (ValueError, RuntimeError):
            has_px = True
            result['NoError'] = False

        # TODO: 7FE0?
        result['HasPixelArray'] = has_px

    for attr in instance.dir():
        try:
            value = instance.get(attr)
        except BaseException as e:
            logger.debug(f'Exception while accessing key "{attr}": {e.__class__.__name__} {e}')
            continue
        if value is None:
            continue

        if isinstance(value, PERSON_CLASS):
            result[attr] = str(value)

        elif isinstance(value, (int, float, str)):
            result[attr] = value

        elif attr in SERIAL:
            for pos, num in enumerate(value):
                result[f'{attr}{pos}'] = num

    if extract_private:
        for private_tag in iter_private_tags(instance):
            if isinstance(private_tag, sequence.Sequence):
                pass
            if private_tag.VR not in valuerep.LONG_VALUE_VR:
                value = instance.get(private_tag.tag).value
                if isinstance(value, (int, float)):
                    result[private_tag.name] = value
                if isinstance(value, str):
                    result[private_tag.name] = value[:100] # just in case

    return result


def join_tree(top: PathLike, ignore_extensions: Sequence[str] = (), relative: bool = True, verbose: int = 0,
              read_pixel_array: bool = False, force: bool = True, unpack_volumetric: bool = True, extract_private: bool = False,
              total: bool = False) -> pd.DataFrame:
    """
    Returns a dataframe containing metadata for each file in all the subfolders of ``top``.

    Parameters
    ----------
    top: PathLike
        path to crawled folder
    ignore_extensions: Sequence
        list of extensions to skip during crawling
    relative: bool
        whether the ``PathToFolder`` attribute should be relative to ``top`` default is True.
    verbose: int
        the verbosity level:
            | 0 - no progressbar
            | 1 - progressbar with iterations count
            | 2 - progressbar with filenames
    total: bool
        whether to show the total number of files in the progressbar.
        This is adds a bit of overhead, because each file will be visited a second time (without being opened).

    References
    ----------
    See the :doc:`tutorials/dicom` tutorial for more details.

    Notes
    -----
    The following columns are added:
        | NoError: whether an exception was raised during reading the file.
        | HasPixelArray:(if NoError is True) whether the file contains a pixel array(added if read_pixel_array is True).
        | PixelArrayShape: (if HasPixelArray is True) the shape of the pixel array (added if read_pixel_array is True).
        | PathToFolder
        | FileName

    For some formats the following packages might be required:
        >>> conda install -c glueviz gdcm # Python 3.5 and 3.6
        >>> conda install -c conda-forge gdcm # Python 3.7
    """
    for extension in ignore_extensions:
        if not extension.startswith('.'):
            raise ValueError(f'Each extension must start with a dot: "{extension}".')

    n_files = None
    if total and verbose:
        n_files = 0
        bar = tqdm(desc='Counting files')
        for root, _, files in os.walk(top, onerror=_throw, followlinks=True):
            for filename in files:
                if not any(filename.endswith(ext) for ext in ignore_extensions):
                    n_files += 1
                    bar.update()
        bar.close()

    result = []
    bar = tqdm(disable=not verbose, total=n_files)
    for root, _, files in os.walk(top, onerror=_throw, followlinks=True):
        root = Path(root)
        rel_path = root.relative_to(top)

        for filename in files:
            if any(filename.endswith(ext) for ext in ignore_extensions):
                continue

            bar.update()
            if verbose > 1:
                bar.set_description(str(rel_path / filename))

            for entry in get_file_meta(root / filename, force=force, read_pixel_array=read_pixel_array,
                                       unpack_volumetric=unpack_volumetric, extract_private=extract_private):
                entry['PathToFolder'] = str(rel_path if relative else root)
                entry['FileName'] = filename
                result.append(entry)

    return pd.DataFrame(result)
