import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import rioxarray as rxr
from pathlib import Path
from sklearn.metrics import accuracy_score, jaccard_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score

# python validation.py --image-regex \
# '/Users/jacaraba/Desktop/holidays_work/senegal/validation/*.gpkg' \
# --label-regex '/Users/jacaraba/Desktop/results_mosaic_v2' \
# --output-dir /Users/jacaraba/Desktop/holidays_work/senegal/validation_scores


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

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    all_sets = []
    for image_filename in images_list:

        # open geopackage
        points_df = gpd.read_file(image_filename)

        # open expected filename
        label_filename = os.path.join(
            args.labels_regex, f'{Path(image_filename).stem}_clouds.tif')

        if not Path(label_filename).is_file():
            label_filename = os.path.join(
                args.labels_regex, f'{Path(image_filename).stem}.tif')

        rds = rxr.open_rasterio(label_filename)
        rds = rds[0, :, :]

        val_values = []
        for index in points_df.index:

            val = rds.sel(
                x=points_df['geometry'][index].x,
                y=points_df['geometry'][index].y,
                method="nearest"
            )
            val_values.append(int(val.values))

        points_df['val'] = val_values

        all_sets.append(points_df)

    all_sets_concat = pd.concat(all_sets)

    accuracy = accuracy_score(
        all_sets_concat['mask'], all_sets_concat['val'])
    precision = precision_score(
        all_sets_concat['mask'], all_sets_concat['val'])
    recall = recall_score(all_sets_concat['mask'], all_sets_concat['val'])
    jaccard = jaccard_score(all_sets_concat['mask'], all_sets_concat['val'])
    confs = confusion_matrix(all_sets_concat['mask'], all_sets_concat['val'])

    print(all_sets_concat.index)
    print('cloud points:', all_sets_concat['mask'].value_counts())
    print(f'acc: {accuracy}')
    print(f'prec: {precision}')
    print(f'rec: {recall}')
    print(f'jacc: {jaccard}')
    print(confs)

    print(
        "Total producer left: ", confs[0, 0],
        confs[1, 0], confs[0, 0] + confs[1, 0])
    print(
        "Total producer right: ", confs[0, 1],
        confs[1, 1], confs[0, 1] + confs[1, 1])
    print(
        "Total user up: ", confs[0, 0],
        confs[0, 1], confs[0, 0] + confs[0, 1])
    print(
        "Total user down: ", confs[1, 0],
        confs[1, 1], confs[1, 0] + confs[1, 1]
    )
    print(
        "Producer accuracy not cloud",
        confs[0, 0] / (confs[0, 0] + confs[1, 0]))
    print(
        "Producer accuracy cloud",
        confs[1, 1] / (confs[0, 1] + confs[1, 1]))

    print(
        "User accuracy not cloud",
        confs[0, 0] / (confs[0, 0] + confs[0, 1]))
    print(
        "User accuracy cloud",
        confs[1, 1] / (confs[1, 0] + confs[1, 1]))

    print(
        "overall accuracy",
        (confs[0, 0] + confs[1, 1]) /
        (confs[0, 0] + confs[0, 1] + confs[1, 0] + confs[1, 1])
    )

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
