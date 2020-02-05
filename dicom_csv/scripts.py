import argparse

from dicom_csv import join_tree


def join_to_csv():
    parser = argparse.ArgumentParser(
        description='Saves a csv dataframe containing metadata for each file in all the subfolders of `top`.')
    parser.add_argument('top', help='The top folder where the search will begin.')
    parser.add_argument('output', help='Path to the output file.')
    parser.add_argument('-a', '--absolute', default=False, action='store_true',
                        help='Whether the paths in the dataframe should be `absolute` or relative to `top`.')
    parser.add_argument('-v', '--verbose', default=0, action='count',
                        help='Whether to show a progressbar.')
    args = parser.parse_args()

    join_tree(args.top, relative=not args.absolute, verbose=args.verbose).to_csv(args.output)
