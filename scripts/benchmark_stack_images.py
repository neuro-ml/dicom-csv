#!/usr/bin/env python3
"""
Benchmark for DICOM series loading: dcmread, pixel_array, get_image, stack_images.

Use this to measure current pydicom/dicom_csv performance before optimising (e.g. with Rust).
Typical target: series of 500 slices 512×512 (≈250 MB).

Usage:
  # Real data (folder with .dcm files)
  python scripts/benchmark_stack_images.py --path /path/to/series_folder [--max-slices 500]

  # Synthetic data (generates temp DICOM files, 512×512×N by default)
  python scripts/benchmark_stack_images.py --synthetic [--slices 500] [--rows 512] [--cols 512]

  # Fewer runs for quick check
  python scripts/benchmark_stack_images.py --synthetic --slices 500 --runs 2

Interpretation:
  - With --synthetic (uncompressed): dcmread is I/O-bound; pixel_array is cheap (data in memory).
  - With real compressed data (JPEG/JPEG-LS): pixel_array and dcmread will be much slower;
    use --path to measure your actual workload.
  - stack_images total includes get_image for all slices plus np.stack (memory allocation).
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

# Assume we're run from repo root or dicom-csv; allow import of dicom_csv
import sys
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dicom_csv import stack_images
from dicom_csv.misc import get_image


def _make_synthetic_dicom(
    filepath: Path,
    rows: int,
    cols: int,
    slice_index: int,
    n_slices: int,
    rescale_slope: float = 1.0,
    rescale_intercept: float = -1024.0,
) -> None:
    """Write one minimal CT-like DICOM slice (16-bit, uncompressed)."""
    pixel_array = np.zeros((rows, cols), dtype=np.int16)
    pixel_array[:] = -1000 + slice_index * 2  # simple gradient for testing

    # File meta
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(filepath), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.PixelRepresentation = 1  # signed
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = rescale_slope
    ds.RescaleIntercept = rescale_intercept
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 1.0
    ds.ImagePositionPatient = [0.0, 0.0, float(slice_index)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelData = pixel_array.tobytes()

    ds.save_as(filepath, enforce_file_format=True)


def generate_synthetic_series(
    dirpath: Path,
    n_slices: int,
    rows: int = 512,
    cols: int = 512,
) -> list[Path]:
    """Create synthetic DICOM series in dirpath; return list of paths."""
    dirpath.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_slices):
        p = dirpath / f"slice_{i:04d}.dcm"
        _make_synthetic_dicom(p, rows, cols, i, n_slices)
        paths.append(p)
    return paths


def load_series_from_folder(folder: Path, max_slices: int | None, force: bool = True) -> list[pydicom.Dataset]:
    """Load DICOM files from folder, optionally limit number of slices."""
    paths = sorted(folder.glob("*.dcm")) + sorted(folder.glob("*.DCM"))
    if not paths:
        raise FileNotFoundError(f"No .dcm files in {folder}")
    if max_slices is not None:
        paths = paths[:max_slices]
    return [pydicom.dcmread(str(p), force=force) for p in paths]


def run_benchmark(
    series: list[pydicom.Dataset],
    warmup: int = 1,
    runs: int = 3,
) -> dict[str, list[float]]:
    """Run timing for pixel_array, get_image, stack_images (multiple runs each)."""
    results: dict[str, list[float]] = {
        "pixel_array": [],
        "get_image": [],
        "get_image_loop": [],   # list(map(get_image, series)) — arrays only
        "np_stack": [],        # np.stack(arrays, axis=-1) only
        "stack_images": [],
    }
    n = len(series)

    # Warmup
    for _ in range(warmup):
        _ = stack_images(series, axis=-1)

    # pixel_array: total time to extract pixel_array for all slices
    for _ in range(runs):
        t0 = time.perf_counter()
        for ds in series:
            _ = ds.pixel_array
        t1 = time.perf_counter()
        results["pixel_array"].append(t1 - t0)

    # get_image: total time for get_image on all slices
    for _ in range(runs):
        t0 = time.perf_counter()
        for ds in series:
            _ = get_image(ds)
        t1 = time.perf_counter()
        results["get_image"].append(t1 - t0)

    # Breakdown: get_image_loop (build list) vs np_stack (single np.stack)
    for _ in range(runs):
        t0 = time.perf_counter()
        arrays = [get_image(ds) for ds in series]
        t1 = time.perf_counter()
        results["get_image_loop"].append(t1 - t0)
        t2 = time.perf_counter()
        img = np.stack(arrays, axis=-1)
        t3 = time.perf_counter()
        results["np_stack"].append(t3 - t2)
        assert img.shape[-1] == n, (img.shape, n)

    # stack_images: full pipeline (get_image for all + np.stack)
    for _ in range(runs):
        t0 = time.perf_counter()
        img = stack_images(series, axis=-1)
        t1 = time.perf_counter()
        results["stack_images"].append(t1 - t0)
        assert img.shape[-1] == n, (img.shape, n)

    return results


def run_benchmark_with_dcmread(
    paths: list[Path],
    warmup: int = 1,
    runs: int = 3,
    force: bool = True,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Run benchmark including dcmread from paths."""
    dcmread_times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        series = [pydicom.dcmread(str(p), force=force) for p in paths]
        t1 = time.perf_counter()
        dcmread_times.append(t1 - t0)
    series = [pydicom.dcmread(str(p), force=force) for p in paths]

    rest = run_benchmark(series, warmup=warmup, runs=runs)
    return {"dcmread": dcmread_times}, rest


def print_report(
    n_slices: int,
    rows: int,
    cols: int,
    dcmread_results: dict[str, list[float]] | None,
    other_results: dict[str, list[float]],
) -> None:
    """Print a simple table of timings."""
    def stats(times: list[float]) -> str:
        if not times:
            return "N/A"
        mean = sum(times) / len(times)
        return f"{mean:.3f}s"

    def per_slice_ms(times: list[float], n: int) -> str:
        if not times or n <= 0:
            return "N/A"
        mean = sum(times) / len(times)
        return f"{(mean / n) * 1000:.2f} ms"

    print("\n" + "=" * 60)
    print("DICOM stack_images benchmark")
    print("=" * 60)
    print(f"  Series shape: {rows} x {cols} x {n_slices}  (≈ {n_slices * rows * cols * 2 / (1024**2):.1f} MB raw)")
    print()

    if dcmread_results:
        for name, times in dcmread_results.items():
            print(f"  {name:20} total: {stats(times):>10}   per slice: {per_slice_ms(times, n_slices):>12}")
    for name, times in other_results.items():
        if name in ("stack_images", "get_image_loop", "np_stack"):
            print(f"  {name:20} total: {stats(times):>10}")
        else:
            # pixel_array / get_image: times are total for all slices
            print(f"  {name:20} total: {stats(times):>10}   per slice: {per_slice_ms(times, n_slices):>12}")
    print("=" * 60)
    if "np_stack" in other_results and other_results["np_stack"] and other_results["stack_images"]:
        np_mean = sum(other_results["np_stack"]) / len(other_results["np_stack"])
        stk_mean = sum(other_results["stack_images"]) / len(other_results["stack_images"])
        if np_mean / stk_mean > 0.5:
            print("  → stack_images uses pre-allocate + fill. Further gains: Rust/direct decode into buffer.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DICOM series loading (dcmread, pixel_array, get_image, stack_images).",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to folder containing .dcm files (real data).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic DICOM series in a temp dir instead of using --path.",
    )
    parser.add_argument("--slices", type=int, default=500, help="Number of slices (synthetic or limit).")
    parser.add_argument("--rows", type=int, default=512, help="Rows per slice (synthetic only).")
    parser.add_argument("--cols", type=int, default=512, help="Columns per slice (synthetic only).")
    parser.add_argument(
        "--max-slices",
        type=int,
        default=None,
        help="Max slices to load from --path (default: all).",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs before timing.")
    parser.add_argument("--runs", type=int, default=3, help="Number of timed runs.")
    args = parser.parse_args()

    if args.synthetic:
        with tempfile.TemporaryDirectory(prefix="dicom_bench_") as tmp:
            tmp_path = Path(tmp)
            generate_synthetic_series(tmp_path, args.slices, args.rows, args.cols)
            paths = sorted(tmp_path.glob("*.dcm"))
            dcmread_results, other_results = run_benchmark_with_dcmread(
                paths, warmup=args.warmup, runs=args.runs
            )
            # Get shape from first file
            ds0 = pydicom.dcmread(str(paths[0]))
            rows, cols = int(ds0.Rows), int(ds0.Columns)
            print_report(args.slices, rows, cols, dcmread_results, other_results)
        return

    if args.path is None:
        parser.error("Either --path or --synthetic is required.")
    folder = args.path.resolve()
    if not folder.is_dir():
        parser.error(f"Not a directory: {folder}")

    paths = sorted(folder.glob("*.dcm")) + sorted(folder.glob("*.DCM"))
    if not paths:
        raise SystemExit(f"No .dcm files in {folder}")
    if args.max_slices is not None:
        paths = paths[: args.max_slices]
    n_slices = len(paths)
    dcmread_results, other_results = run_benchmark_with_dcmread(
        paths, warmup=args.warmup, runs=args.runs
    )
    ds0 = pydicom.dcmread(str(paths[0]))
    rows, cols = int(ds0.Rows), int(ds0.Columns)
    print_report(n_slices, rows, cols, dcmread_results, other_results)


if __name__ == "__main__":
    main()
