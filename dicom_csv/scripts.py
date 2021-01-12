import argparse
import numpy as np

from dicom_csv import join_tree
from dicom_csv.rtstruct.contour import read_rtstruct
from dicom_csv.rtstruct.csv import collect_rtstruct


def join_to_csv():
    parser = argparse.ArgumentParser(
        description='Saves a csv dataframe containing metadata for each file in all the subfolders of `top`.')
    parser.add_argument('top', help='the top folder where the search will begin.')
    parser.add_argument('output', help='path to the output file.')
    parser.add_argument('-a', '--absolute', default=False, action='store_true',
                        help='whether the paths in the dataframe should be `absolute` or relative to `top`.')
    parser.add_argument('-v', '--verbose', default=0, action='count',
                        help='whether to show a progressbar.')
    parser.add_argument('-t', '--total', default=False, action='store_true',
                        help='whether to show the total number of files in the progressbar.')
    # parser.add_argument('-f', '--force', default=False, action='store_true',
    #                     help='whether to fix the endianness tag.')
    args = parser.parse_args()

    df = join_tree(
        args.top, relative=not args.absolute, verbose=args.verbose, read_pixel_array=False,
        total=args.total,
    )
    df.to_csv(args.output, index=False)
    if args.verbose:
        size = len(df)
        errors = size - df.NoError.sum()
        if errors:
            percentage = 100 * errors / (size or 1)
            print(f'\n{errors} files ({percentage:.2f}%) were opened with errors.')


def collect_contours():
    """Help function for high-level debugging."""

    example = """usage:
  collect_contours /path/to/subject_folder /path/to/file.npy

load results:
  np.load(/path/to/file.npy).items()
    """
    # TODO: remove this function
    parser = argparse.ArgumentParser(
        description='Saves json with contours from subject folder.',
        epilog=example,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', help='subject folder')
    parser.add_argument('output', help='path to the output file.')

    args = parser.parse_args()

    data_csv = join_tree(args.folder, relative=False)
    rtstruct_csv = collect_rtstruct(data_csv)
    result = dict()
    for rtstruct in rtstruct_csv.iterrows():
        patient_id = rtstruct[1].PatientID
        mask_suid = rtstruct[1].SeriesInstanceUID
        reference_suid = rtstruct[1].ReferenceSeriesInstanceUID
        contours_dict = read_rtstruct(rtstruct[1])

        result[str(patient_id)] = {
            'ReferenceSeriesInstanceUID': reference_suid,
            'SeriesInstanceUID': mask_suid,
            'Contours': contours_dict
        }

    np.save(args.output, result)
