"""Contains functions for gathering metadata from individual DICOM files or entire directories."""
import os
from os.path import join as jp
from typing import Sequence

import pandas as pd
from tqdm import tqdm
from pydicom import valuerep, errors, dcmread

from .utils import PathLike

__all__ = 'get_file_meta', 'join_tree'

SERIAL = {'ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing'}
PERSON_CLASS = (valuerep.PersonName3, valuerep.PersonNameBase)


def _throw(e):
    raise e


def get_file_meta(path: PathLike) -> dict:
    """
    Get a dict containing the metadata from the DICOM file located at ``path``.

    Notes
    -----
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
        dc = dcmread(str(path))
        result['NoError'] = True
    except (errors.InvalidDicomError, OSError, NotImplementedError, AttributeError):
        result['NoError'] = False
        return result

    try:
        has_px = hasattr(dc, 'pixel_array')
    except (TypeError, NotImplementedError):
        has_px = False
    except ValueError:
        has_px = True
        result['NoError'] = False
    else:
        if has_px:
            result['PixelArrayShape'] = ','.join(map(str, dc.pixel_array.shape))
    result['HasPixelArray'] = has_px

    for attr in dc.dir():
        try:
            value = dc.get(attr)
        except NotImplementedError:
            continue
        if value is None:
            continue

        if isinstance(value, PERSON_CLASS):
            result[attr] = str(value)

        elif attr in SERIAL:
            for pos, num in enumerate(value):
                result[f'{attr}{pos}'] = num

        elif isinstance(value, (int, float, str)):
            result[attr] = value

    return result


def join_tree(top: PathLike, ignore_extensions: Sequence[str] = (), relative: bool = True,
              verbose: int = 0) -> pd.DataFrame:
    """
    Returns a dataframe containing metadata for each file in all the subfolders of ``top``.

    Parameters
    ----------
    top
    ignore_extensions
    relative
        whether the ``PathToFolder`` attribute should be relative to ``top``.
    verbose
        the verbosity level:
            | 0 - no progressbar
            | 1 - progressbar with iterations count
            | 2 - progressbar with filenames

    References
    ----------
    See the :doc:`tutorials/dicom` tutorial for more details.

    Notes
    -----
    The following columns are added:
        | NoError: whether an exception was raised during reading the file.
        | HasPixelArray: (if NoError is True) whether the file contains a pixel array.
        | PixelArrayShape: (if HasPixelArray is True) the shape of the pixel array.
        | PathToFolder
        | FileName

    For some formats the following packages might be required:
        >>> conda install -c glueviz gdcm # Python 3.5 and 3.6
        >>> conda install -c conda-forge gdcm # Python 3.7
    """
    for extension in ignore_extensions:
        if not extension.startswith('.'):
            raise ValueError(f'Each extension must start with a dot: "{extension}".')

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

            entry = get_file_meta(jp(root, file))
            entry['PathToFolder'] = rel_path if relative else root
            entry['FileName'] = file
            result.append(entry)

    return pd.DataFrame(result)
