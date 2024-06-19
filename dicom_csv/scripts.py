import argparse

from .crawler import join_tree


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
