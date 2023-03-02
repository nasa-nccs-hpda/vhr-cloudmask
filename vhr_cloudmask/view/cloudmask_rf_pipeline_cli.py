# -*- coding: utf-8 -*-
"""
CPU and GPU Random Forest Pipeline - preprocess, train, predict and visualize.
Author: Jordan A. Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
"""

import os
import gc
import sys
import glob
import random
import logging
import argparse
from typing import List
from pathlib import Path
from tqdm import tqdm

import cv2
import joblib
import numpy as np
import xarray as xr
import pandas as pd
import rioxarray as rxr
from scipy.ndimage import median_filter, binary_fill_holes

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as sklRFC

from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# from model.indices import addindices, fdi, si, ndwi, modify_bands

try:
    import cupy as cp
    import cudf as cf
    from cuml.ensemble import RandomForestClassifier as cumlRFC
    from cuml.dask.ensemble import RandomForestClassifier as cumlRFC_mg
    from cupyx.scipy.ndimage import median_filter_gpu
    cp.random.seed(seed=None)
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

__all__ = ["cumlRFC_mg", "median_filter_gpu", "median_filter"]


# -----------------------------------------------------------------------------
# rf_driver.py methods
# -----------------------------------------------------------------------------
def modify_bands(
        xraster: xr.core.dataarray.DataArray, input_bands: List[str],
        output_bands: List[str], drop_bands: List[str] = []):
    """
    Drop multiple bands to existing rasterio object
    """
    # Do not modify if image has the same number of output bands
    if xraster['band'].shape[0] == len(output_bands):
        return xraster

    # Drop any bands from input that should not be on output
    for ind_id in list(set(input_bands) - set(output_bands)):
        drop_bands.append(input_bands.index(ind_id)+1)
    return xraster.drop(dim="band", labels=drop_bands, drop=True)


def predict(data, model, ws=[5120, 5120], probabilities=False):
    """
    Predict from model.
    :param data: raster xarray object
    :param model: loaded model object
    :param ws: window size to predict on
    :return: prediction output in numpy format
    ----------
    Example
        raster.toraster(filename, raster_obj.prediction, outname)
    ----------
    """
    # open rasters and get both data and coordinates
    rast_shape = data[0, :, :].shape  # shape of the wider scene
    wsy, wsx = ws[0], ws[1]  # in memory sliding window predictions

    # if the window size is bigger than the image, predict full image
    if wsy > rast_shape[0]:
        wsy = rast_shape[0]
    if wsx > rast_shape[1]:
        wsx = rast_shape[1]

    prediction = np.zeros(rast_shape)  # crop out the window
    logging.info(f'wsize: {wsy}x{wsx}. Prediction shape: {prediction.shape}')

    for sy in tqdm(range(0, rast_shape[0], wsy)):  # iterate over x-axis
        for sx in range(0, rast_shape[1], wsx):  # iterate over y-axis
            y0, y1, x0, x1 = sy, sy + wsy, sx, sx + wsx  # assign window
            if y1 > rast_shape[0]:  # if selected x exceeds boundary
                y1 = rast_shape[0]  # assign boundary to x-window
            if x1 > rast_shape[1]:  # if selected y exceeds boundary
                x1 = rast_shape[1]  # assign boundary to y-window

            window = data[:, y0:y1, x0:x1]  # get window
            window = window.stack(z=('y', 'x'))  # stack y and x axis
            window = window.transpose("z", "band").values  # reshape

            # perform sliding window prediction
            if probabilities:
                prediction[y0:y1, x0:x1] = \
                    model.predict_proba(window)[:, 1].reshape(
                        (y1 - y0, x1 - x0))
            else:
                prediction[y0:y1, x0:x1] = \
                    model.predict(window).reshape((y1 - y0, x1 - x0))
    # save raster
    return prediction


def grow(merged_mask, eps=120):
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, struct)


def denoise(merged_mask, eps=30):
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(merged_mask, cv2.MORPH_OPEN, struct)


def binary_fill(merged_mask):
    return binary_fill_holes(merged_mask)


# -----------------------------------------------------------------------------
# main
#
# python rf_driver.py options here
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Random Forest Segmentation pipeline for tabular and spatial data.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--gpu', dest='has_gpu', action='store_true', default=False)

    parser.add_argument(
        '--output-model', type=str, default='output.pkl', required=False,
        dest='output_pkl', help='Path to the output PKL file (.pkl)')

    parser.add_argument(
        '--data-csv', type=str, required=False,
        dest='data_csv', help='Path to the data CSV configuration file')

    parser.add_argument(
        '--train-csv', type=str, required=False,
        dest='train_csv', help='Path to the output CSV file')

    parser.add_argument(
        '--bands', type=str, nargs='*', required=False,
        dest='bands', help='Bands to store in CSV file',
        default=['CoastalBlue', 'Blue', 'Green', 'Yellow',
                 'Red', 'RedEdge', 'NIR1', 'NIR2'])

    parser.add_argument(
        '--input-bands', type=str, nargs='*', required=False,
        dest='input_bands', help='Input bands from predicted data',
        default=['CoastalBlue', 'Blue', 'Green', 'Yellow',
                 'Red', 'RedEdge', 'NIR1', 'NIR2'])

    parser.add_argument(
        '--step', type=str, nargs='*', required=True,
        dest='pipeline_step', help='Pipeline step to perform',
        default=['preprocess', 'train', 'predict', 'vis'],
        choices=['preprocess', 'train', 'predict', 'vis'])

    parser.add_argument(
        '--seed', type=int, required=False, dest='seed',
        default=42, help='Random SEED value')

    parser.add_argument(
        '--test-size', type=float, required=False,
        dest='test_size', default=0.20, help='Test size rate (e.g .30)')

    parser.add_argument(
        '--n-trees', type=int, required=False,
        dest='n_trees', default=20, help='Number of trees (e.g 20)')

    parser.add_argument(
        '--max-features', type=str, required=False,
        dest='max_feat', default='log2', help='Max features (e.g log2)')

    parser.add_argument(
        '--rasters', type=str, required=False, dest='rasters',
        default='*.tif', help='rasters to search for')

    parser.add_argument(
        '--window-size', type=int, required=False,
        dest='ws', default=5120, help='Prediction window size (e.g 5120)')

    parser.add_argument(
        '--output-dir', type=str, required=False,
        dest='output_dir', default='', help='output directory')

    parser.add_argument(
        '--output-probabilities', dest='output_probabilities',
        action='store_true', default=False)

    parser.add_argument(
        '--postprocess', type=bool, required=False,
        dest='postprocess', default=True,
        help='bool for postprocessing (denoise, fill)')

    args = parser.parse_args()

    # --------------------------------------------------------------------------------
    # set logging
    # --------------------------------------------------------------------------------
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)  # set stdout handler
    ch.setLevel(logging.INFO)

    # set formatter and handlers
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # --------------------------------------------------------------------------------
    # preprocess step
    # --------------------------------------------------------------------------------
    if "preprocess" in args.pipeline_step:

        # ----------------------------------------------------------------------------
        # 1. Read data csv file
        # ----------------------------------------------------------------------------
        assert os.path.exists(args.data_csv), f'{args.data_csv} not found.'
        data_df = pd.read_csv(args.data_csv)  # initialize data dataframe
        assert not data_df.isnull().values.any(), f'NaN found: {args.data_csv}'
        logging.info(f'Open {args.data_csv} for preprocessing.')

        # ----------------------------------------------------------------------------
        # 2. Extract points out of spatial imagery - rasters
        # ----------------------------------------------------------------------------
        points_df = pd.DataFrame(columns=args.bands + ['CLASS'])
        points_list = []
        logging.info(f"Generating {data_df['ntiles'].sum()} points dataset.")

        # start iterating over each file
        for df_index in data_df.index:

            # get filename for output purposes, in the future, add to column
            filename = Path(data_df['data'][df_index]).stem
            logging.info(f'Processing {filename}')

            # read imagery from disk and process both image and mask
            img = rxr.open_rasterio(data_df['data'][df_index]).values
            mask = rxr.open_rasterio(data_df['label'][df_index]).values
            logging.info(f'Imagery shape: {img.shape}')

            # squeeze mask if needed, all non-1 values to 0
            mask = np.squeeze(mask) if len(mask.shape) != 2 else mask

            # get N points from imagery
            points_per_class = data_df['ntiles'][df_index] // 2

            # extract values from imagery, two classes
            for cv in range(2):

                bbox = img.shape  # size of the image
                counter = 0  # counter for class balancing

                while counter < points_per_class:

                    # get indices and extract spectral and class value
                    y, x = random.randrange(bbox[1]), random.randrange(bbox[2])
                    sv, lv = img[:, y, x], int(mask[y, x])

                    if lv == cv:
                        # trying speed up here - looks like from list is faster
                        points_list.append(
                            pd.DataFrame(
                                [np.append(sv, [lv])],
                                columns=list(points_df.columns))
                        )
                        counter += 1

        df_points = pd.concat(points_list)

        # ----------------------------------------------------------------------------
        # 3. Save file to disk
        # ----------------------------------------------------------------------------
        os.makedirs(Path(args.train_csv).parent.absolute(), exist_ok=True)
        df_points.to_csv(args.train_csv, index=False)
        logging.info(f'Saved dataset file at: {args.train_csv}')

    # --------------------------------------------------------------------------------
    # train step
    # --------------------------------------------------------------------------------
    if "train" in args.pipeline_step:

        # ----------------------------------------------------------------------------
        # 1. Read data csv file
        # ----------------------------------------------------------------------------
        assert os.path.exists(args.train_csv), f'{args.train_csv} not found.'
        data_df = pd.read_csv(
            args.train_csv, sep=',', names=args.bands + ['CLASS'])
        assert not data_df.isnull().values.any(), f'Na found: {args.train_csv}'
        logging.info(f'Open {args.train_csv} dataset for training.')

        # ----------------------------------------------------------------------------
        # 2. Shuffle and Split Dataset
        # ----------------------------------------------------------------------------
        data_df = data_df.sample(frac=1).reset_index(drop=True)  # shuffle data

        # split dataset, fix type
        x = data_df.iloc[:, :-1].astype(np.float32)
        y = data_df.iloc[:, -1].astype(np.int8)

        # split data into training and test
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=args.test_size, random_state=args.seed)
        del data_df, x, y

        # logging some of the model information
        logging.info(f'X size: {x_train.shape}, Y size:  {y_train.shape}')
        logging.info(f'ntrees={str(args.n_trees)}, maxfeat={args.max_feat}')

        # ------------------------------------------------------------------
        # 3. Instantiate RandomForest object - FIX this area
        # ------------------------------------------------------------------
        if args.has_gpu:  # run using RAPIDS library

            # initialize cudf data and log into GPU memory
            logging.info('Training model via RAPIDS.')

            # single gpu setup
            x_train = cf.DataFrame.from_pandas(x_train)
            x_test = cf.DataFrame.from_pandas(x_test)
            y_train = cf.Series(y_train.values)
            rf_funct = cumlRFC  # RF Classifier

        else:
            logging.info('Training model via SKLearn.')
            rf_funct = sklRFC

        # Initialize model
        rf_model = rf_funct(
            n_estimators=args.n_trees, max_features=args.max_feat)

        # ------------------------------------------------------------------
        # 4. Fit Model
        # ------------------------------------------------------------------
        # fit model to training data and predict for accuracy score
        rf_model.fit(x_train, y_train)

        if args.has_gpu:
            acc_score = accuracy_score(
                y_test, rf_model.predict(x_test).to_array())
            p_score = precision_score(
                y_test, rf_model.predict(x_test).to_array())
            r_score = recall_score(
                y_test, rf_model.predict(x_test).to_array())
            f_score = f1_score(
                y_test, rf_model.predict(x_test).to_array())
        else:
            acc_score = accuracy_score(y_test, rf_model.predict(x_test))
            p_score = precision_score(y_test, rf_model.predict(x_test))
            r_score = recall_score(y_test, rf_model.predict(x_test))
            f_score = f1_score(y_test, rf_model.predict(x_test))

        logging.info(f'Test Accuracy:  {acc_score}')
        logging.info(f'Test Precision: {p_score}')
        logging.info(f'Test Recall:    {r_score}')
        logging.info(f'Test F-Score:   {f_score}')

        # make output directory
        os.makedirs(
            os.path.dirname(os.path.realpath(args.output_pkl)), exist_ok=True)

        # export model to file
        try:
            joblib.dump(rf_model, args.output_pkl)
            logging.info(f'Model has been saved as {args.output_pkl}')
        except Exception as e:
            logging.error(f'ERROR: {e}')

    # --------------------------------------------------------------------------------
    # predict step
    # --------------------------------------------------------------------------------
    if "predict" in args.pipeline_step:

        assert os.path.exists(args.output_pkl), f'{args.output_pkl} not found.'
        model = joblib.load(args.output_pkl)  # loading pkl in parallel
        logging.info(f'Loaded model {args.output_pkl}.')

        # create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # 3b3. apply model and get predictions
        rasters = glob.glob(args.rasters)
        assert len(rasters) > 0, "No raster found"
        logging.info(f'Predicting {len(rasters)} files.')

        for raster_filename in rasters:  # iterate over each raster

            filename = Path(raster_filename).stem
            output_filename = os.path.join(
                args.output_dir, f'{filename}.cloudmask.tif')

            if not os.path.isfile(output_filename):

                gc.collect()  # clean garbage
                logging.info(f'Starting new prediction...{raster_filename}')
                img = rxr.open_rasterio(raster_filename)

                img = modify_bands(
                    img, input_bands=args.input_bands,
                    output_bands=args.bands)
                logging.info(f'Imagery shape after band mod {img.shape}')

                prediction = predict(
                    img, model, ws=[args.ws, args.ws],
                    probabilities=args.output_probabilities)
                print(prediction.min(), prediction.max())

                if args.postprocess and not args.output_probabilities:
                    prediction = denoise(np.uint8(prediction))
                    prediction = binary_fill(prediction)
                    prediction = grow(np.uint8(prediction))

                img = img.transpose("y", "x", "band")
                img = img.drop(
                    dim="band",
                    labels=img.coords["band"].values[1:],
                    drop=True
                )

                prediction = xr.DataArray(
                    np.expand_dims(prediction, axis=-1),
                    name='mask',
                    coords=img.coords,
                    dims=img.dims,
                    attrs=img.attrs)

                prediction.attrs['long_name'] = ('mask')
                prediction = prediction.transpose("band", "y", "x")

                nodata = prediction.rio.nodata
                prediction = prediction.where(img != nodata)
                prediction.rio.write_nodata(nodata, encoded=True, inplace=True)
                prediction.rio.to_raster(
                    output_filename, BIGTIFF="IF_SAFER", compress='LZW')

                del prediction

            else:
                logging.info(f'{output_filename} already predicted.')

    # --------------------------------------------------------------------------------
    # predict step
    # --------------------------------------------------------------------------------
    if "vis" in args.pipeline_step:

        assert os.path.exists(args.output_pkl), f'{args.output_pkl} not found.'
        model = joblib.load(args.output_pkl)  # loading pkl in parallel
        logging.info(f'Loaded model {args.output_pkl}.')

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
