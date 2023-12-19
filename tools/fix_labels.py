import os
import sys
import glob
import argparse
import numpy as np
import xarray as xr
import rasterio as rio


def arr_to_tif(raster_f, segments, out_tif='s.tif', ndval=-10001):
    """
    Save array into GeoTIF file.
    Args:
        raster_f (str): input data filename
        segments (numpy.array): array with values
        out_tif (str): output filename
        ndval (int): no data value
    Return:
        save GeoTif to local disk
    ----------
    Example
    ----------
        arr_to_tif('inp.tif', segments, 'out.tif', ndval=-9999)
    """
    # get geospatial profile, will apply for output file
    with rio.open(raster_f) as src:
        meta = src.profile
        nodatavals = src.read_masks(1).astype('int16')

    # load numpy array if file is given
    if type(segments) == 'str':
        segments = np.load(segments)
    segments = segments.astype('int16')

    nodatavals[nodatavals == 0] = ndval
    segments[nodatavals == ndval] = nodatavals[nodatavals == ndval]

    out_meta = meta  # modify profile based on numpy array
    out_meta['count'] = 1  # output is single band
    out_meta['dtype'] = 'int16'  # data type is float64

    # write to a raster
    with rio.open(out_tif, 'w', **out_meta) as dst:
        dst.write(segments, 1)


# -----------------------------------------------------------------------------
# main
#
# python fix_labels.py options here
# -----------------------------------------------------------------------------
def main():

    # -------------------------------------------------------------------------
    # Process command-line args.
    # -------------------------------------------------------------------------
    desc = 'Use this application to preprocess label files.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--image-regex", type=str, required=True, dest='images_regex',
        help="Regex to load image files")

    parser.add_argument(
        "--label-regex", type=str, required=True, dest='labels_regex',
        help="Regex to load label files")

    parser.add_argument(
        "--output-dir", type=str, required=True, dest='output_dir',
        help="Output directory to store labels")

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Preprocessing of label files
    # -------------------------------------------------------------------------
    # Get list of data and label files
    images_list = sorted(glob.glob(args.images_regex))
    labels_list = sorted(glob.glob(args.labels_regex))

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    for image_filename, label_filename in zip(images_list, labels_list):

        label = xr.open_rasterio(label_filename).values
        label = np.squeeze(label)
        print(f"Unique classes: {np.unique(label)}")

        # preprocessing specific for this project
        label[label != 1] = 0
        print(f"Unique classes after preprocessing: {np.unique(label)}")

        # output filename setup
        output_filename = label_filename.split('/')[-1]
        output_filename = os.path.join(args.output_dir, output_filename)

        # to raster
        arr_to_tif(
            raster_f=image_filename, segments=label, out_tif=output_filename)
    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
