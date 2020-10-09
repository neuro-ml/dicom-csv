"""Contains functions for gathering metadata from individual DICOM files or entire directories."""
import os
import struct
from os.path import join as jp
from typing import Sequence

import pandas as pd
import numpy as np
from tqdm import tqdm
from pydicom import valuerep, errors, dcmread
from pydicom.uid import ImplicitVRLittleEndian

from .utils import PathLike

__all__ = 'get_file_meta', 'join_tree'

SERIAL = {'ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing'}
PERSON_CLASS = valuerep.PersonName


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


def get_file_meta(path: PathLike, force: bool = True, read_pixel_array: bool = True) -> dict:
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
    result = {}

    try:
        result['NoError'], dc = read_dicom(path, force)
    except (errors.InvalidDicomError, struct.error, OSError, NotImplementedError, AttributeError, KeyError):
        result['NoError'] = False
        return result

    if read_pixel_array:
        try:
            has_px = hasattr(dc, 'pixel_array')
        except (TypeError, NotImplementedError):
            has_px = False
        except ValueError:
            has_px = True
            result['NoError'] = False
        except RuntimeError:
            has_px = True
            result['NoError'] = False

        result['HasPixelArray'] = has_px

        # TODO: stop_before_pixels, shape could be assessed from Row, Columns attributes
        if has_px and result['NoError']:
            result['PixelArrayShape'] = ','.join(map(str, dc.pixel_array.shape))

    for attr in dc.dir():
        try:
            value = dc.get(attr)
        except:
            continue
        if value is None:
            continue

        if isinstance(value, PERSON_CLASS):
            result[attr] = str(value)

        elif attr in SERIAL:
            for pos, num in enumerate(value):
                result[f'{attr}{pos}'] = np.round(num, 5)  # float precision errors

        elif isinstance(value, (int, float, str)):
            result[attr] = value

    return result


def join_tree(top: PathLike, ignore_extensions: Sequence[str] = (), relative: bool = True, verbose: int = 0,
              read_pixel_array: bool = True, get_attrs=None) -> pd.DataFrame:
    """
    Returns a dataframe containing metadata for each file in all the subfolders of ``top``.

    Parameters
    ---
    top - PathLike,
        path to crawled folder

    ignore_extensions - Sequence,
        list of extensions to skip during crawling

    relative - bool,
        whether the ``PathToFolder`` attribute should be relative to ``top`` default is True.

    verbose - int,
        the verbosity level:
            | 0 - no progressbar
            | 1 - progressbar with iterations count
            | 2 - progressbar with filenames

    get_attrs - function,
        function to use to collect metadata from the file, if None dicom_csv.crawler.get_file_meta is used.

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

    if get_attrs is None:
        get_attrs = get_file_meta

    result = []
    bar = tqdm(disable=not verbose)
    for root, _, files in os.walk(top, onerror=_throw, followlinks=True):
        rel_path = os.path.relpath(root, top)

        for file in files:
            if any(file.endswith(ext) for ext in ignore_extensions):
                continue

            bar.update()
            if verbose > 1:
                bar.set_description(jp(rel_path, file))

            entry = get_attrs(jp(root, file), read_pixel_array=read_pixel_array)
            entry['PathToFolder'] = rel_path if relative else root
            entry['FileName'] = file
            result.append(entry)

    return pd.DataFrame(result)
