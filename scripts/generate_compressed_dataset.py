#!/usr/bin/env python3
"""
Generate a synthetic DICOM series dataset and save with optional compression.

Use for benchmarking stack_images on compressed data (RLE, JPEG-LS).
Output: one folder per series, e.g. out_dir/<series_uid>/slice_0000.dcm ...

Usage:
  # Uncompressed (Explicit VR Little Endian)
  python scripts/generate_compressed_dataset.py --out-dir ./data/ct_series --slices 100

  # RLE Lossless (built-in pydicom, no extra deps)
  python scripts/generate_compressed_dataset.py --out-dir ./data/ct_rle --slices 100 --compression rle

  # JPEG-LS Lossless (requires: pip install pyjpegls)
  python scripts/generate_compressed_dataset.py --out-dir ./data/ct_jpegls --slices 100 --compression jpegls

  # Full 500×512×512 series, RLE
  python scripts/generate_compressed_dataset.py --out-dir ./data/ct_500 --slices 500 --rows 512 --cols 512 --compression rle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

try:
    from pydicom.uid import RLELossless
except ImportError:
    RLELossless = None

try:
    from pydicom.uid import JPEGLSLossless
except ImportError:
    JPEGLSLossless = None


def _make_slice_dataset(
    rows: int,
    cols: int,
    slice_index: int,
    study_uid: str,
    series_uid: str,
    rescale_slope: float = 1.0,
    rescale_intercept: float = -1024.0,
) -> FileDataset:
    """Build one minimal CT-like DICOM dataset in memory (uncompressed PixelData)."""
    pixel_array = np.zeros((rows, cols), dtype=np.int16)
    pixel_array[:] = -1000 + slice_index * 2

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(
        "", {},
        file_meta=file_meta,
        preamble=b"\x00" * 128,
    )
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "CT"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = rescale_slope
    ds.RescaleIntercept = rescale_intercept
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 1.0
    ds.ImagePositionPatient = [0.0, 0.0, float(slice_index)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelData = pixel_array.tobytes()
    return ds


def _compress_and_save(
    ds: FileDataset,
    filepath: Path,
    compression: str,
    encoding_plugin: str | None = None,
) -> None:
    """Compress dataset if requested and save to filepath."""
    if compression == "rle":
        if RLELossless is None:
            raise RuntimeError("RLELossless not found in pydicom.uid; upgrade pydicom.")
        ds.compress(RLELossless, encoding_plugin=encoding_plugin or "pydicom")
    elif compression == "jpegls":
        if JPEGLSLossless is None:
            raise RuntimeError("JPEGLSLossless not found in pydicom.uid; upgrade pydicom.")
        try:
            ds.compress(JPEGLSLossless, encoding_plugin="pyjpegls")
        except Exception as e:
            raise RuntimeError(
                "JPEG-LS compression requires pyjpegls: pip install pyjpegls"
            ) from e
    # else: keep uncompressed (Explicit VR LE already set)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(filepath), enforce_file_format=True)


def generate_dataset(
    out_dir: Path,
    n_slices: int,
    rows: int = 512,
    cols: int = 512,
    compression: str = "none",
    study_uid: str | None = None,
    series_uid: str | None = None,
    encoding_plugin: str | None = None,
) -> tuple[Path, str, str]:
    """
    Generate synthetic CT series and save to out_dir/<series_uid>/slice_XXXX.dcm.

    Returns:
        (series_dir, study_uid, series_uid)
    """
    study_uid = study_uid or generate_uid()
    series_uid = series_uid or generate_uid()
    series_dir = out_dir / series_uid
    series_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_slices):
        ds = _make_slice_dataset(rows, cols, i, study_uid, series_uid)
        filepath = series_dir / f"slice_{i:04d}.dcm"
        _compress_and_save(ds, filepath, compression, encoding_plugin)

    return series_dir, study_uid, series_uid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic DICOM series dataset (optionally compressed).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory. Series will be written to <out-dir>/<series_uid>/.",
    )
    parser.add_argument("--slices", type=int, default=100, help="Number of slices.")
    parser.add_argument("--rows", type=int, default=512, help="Rows per slice.")
    parser.add_argument("--cols", type=int, default=512, help="Columns per slice.")
    parser.add_argument(
        "--compression",
        choices=["none", "rle", "jpegls"],
        default="none",
        help="Compression: none (Explicit VR LE), rle (RLE Lossless), jpegls (JPEG-LS Lossless).",
    )
    parser.add_argument(
        "--study-uid",
        type=str,
        default=None,
        help="StudyInstanceUID (default: generate new).",
    )
    parser.add_argument(
        "--series-uid",
        type=str,
        default=None,
        help="SeriesInstanceUID (default: generate new).",
    )
    parser.add_argument(
        "--encoding-plugin",
        type=str,
        default=None,
        help="For RLE: 'pylibjpeg' (faster) or 'pydicom'. Ignored for jpegls.",
    )
    args = parser.parse_args()

    try:
        series_dir, study_uid, series_uid = generate_dataset(
            args.out_dir.resolve(),
            args.slices,
            args.rows,
            args.cols,
            args.compression,
            args.study_uid,
            args.series_uid,
            args.encoding_plugin,
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    total_mb = args.slices * args.rows * args.cols * 2 / (1024 ** 2)
    print(f"Generated {args.slices} slices ({args.rows}×{args.cols}) in {series_dir}")
    print(f"  StudyInstanceUID:  {study_uid}")
    print(f"  SeriesInstanceUID: {series_uid}")
    print(f"  Compression:       {args.compression}")
    print(f"  Raw size ≈ {total_mb:.1f} MB")
    print()
    print("Benchmark with:")
    print(f"  python scripts/benchmark_stack_images.py --path {series_dir}")


if __name__ == "__main__":
    main()
