import os
import re
import cv2
import time
import logging
import rasterio
import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio.features as riofeat

from pathlib import Path
from scipy import ndimage
from skimage import color
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from tensorflow_caney.utils.data import modify_bands, \
    get_mean_std_metadata, read_metadata
from tensorflow_caney.utils import indices
from tensorflow_caney.inference import inference
from tensorflow_caney.utils.model import load_model
from tensorflow_caney.utils.system import seed_everything

from vhr_cloudmask.model.config.cloudmask_config \
    import CloudMaskConfig as Config
from tensorflow_caney.model.pipelines.cnn_segmentation import CNNSegmentation

CHUNKS = {'band': 'auto', 'x': 'auto', 'y': 'auto'}
__status__ = "Production"


# -----------------------------------------------------------------------------
# class CloudMaskPipeline
# -----------------------------------------------------------------------------
class CloudMaskPipeline(CNNSegmentation):
    """This is a conceptual class representation of a CNN Segmentation
    TensorFlow pipeline. It is essentially an extended combination of the
    :class:`tensorflow_caney.model.pipelines.cnn_segmentation.CNNSegmentation`.

    :param logger: A logger device
    :type logger: str
    :param conf: Configuration device
    :type conf: omegaconf.OmegeConf object
    :param data_csv: CSV filename with data files for training
    :type data_csv: str
    :param experiment_name: Experiment name description
    :type experiment_name: str
    :param images_dir: Directory to store training images
    :type images_dir: str
    :param labels_dir: Directory to store training labels
    :type labels_dir: str
    :param model_dir: Directory to store trained models
    :type model_dir: str
    """
    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(
                self,
                config_filename: str = None,
                data_csv: str = None,
                model_filename: str = None,
                output_dir: str = None,
                inference_regex_list: str = None,
                input_bands: str = None,
                output_bands: str = None,
                postprocessing_steps: list = None,
                default_config: str = 'templates/cloudmask_default.yaml',
                logger=None
            ):
        """Constructor method
        """
        logging.info('Initializing CloudMaskPipeline')

        # Configuration file intialization
        if config_filename is None:
            config_filename = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                default_config)
            logging.info(f'Loading default config: {config_filename}')

        # load configuration into object
        self.conf = self._read_config(config_filename, Config)

        # Set logger
        self.logger = logger if logger is not None else self._set_logger()

        # rewrite model filename option if given from CLI
        if model_filename is not None:
            assert os.path.exists(model_filename), \
                f'{model_filename} does not exist.'
            self.conf.model_filename = model_filename

        # rewrite output directory if given from CLI
        if output_dir is not None:
            self.conf.inference_save_dir = output_dir
            os.makedirs(self.conf.inference_save_dir, exist_ok=True)

        # rewrite inference regex list
        if inference_regex_list is not None:
            self.conf.inference_regex_list = inference_regex_list

        # rewrite input bands if given from CLI
        if input_bands is not None:
            self.conf.input_bands = input_bands

        # rewrite output bands if given from CLI
        if output_bands is not None:
            self.conf.output_bands = output_bands

        # rewrite postprocessing steps if given from CLI
        if postprocessing_steps is not None:
            self.conf.postprocessing_steps = postprocessing_steps

        # Set Data CSV
        self.data_csv = data_csv

        # Set experiment name
        try:
            self.experiment_name = self.conf.experiment_name.name
        except AttributeError:
            self.experiment_name = self.conf.experiment_name

        # Output directories for metadata
        self.images_dir = os.path.join(self.conf.data_dir, 'images')
        logging.info(f'Images dir: {self.images_dir}')

        self.labels_dir = os.path.join(self.conf.data_dir, 'labels')
        logging.info(f'Labels dir: {self.labels_dir}')

        self.model_dir = self.conf.model_dir
        logging.info(f'Model dir:  {self.labels_dir}')

        logging.info(f'Output dir: {self.conf.inference_save_dir}')

        # Create output directories
        for out_dir in [
                self.images_dir, self.labels_dir,
                self.model_dir]:
            os.makedirs(out_dir, exist_ok=True)

        # save configuration into the model directory
        try:
            OmegaConf.save(
                self.conf, os.path.join(self.model_dir, 'config.yaml'))
            logging.info(f'Saved config under: {self.model_dir}')
        except PermissionError:
            logging.info('No permissions to save config, skipping step.')

        logging.info(
            f'Output GeoTIFF driver: {self.conf.prediction_driver}')

        # Seed everything
        seed_everything(self.conf.seed)

    # -------------------------------------------------------------------------
    # predict
    # -------------------------------------------------------------------------
    def predict(self) -> None:
        """This will perform inference on a list of GeoTIFF files provided
        as a list of regexes from the CLI.

        :return: None, outputs GeoTIFF cloudmask files to disk.
        :rtype: None
        """

        logging.info('Starting prediction stage')

        # if model filename does not exist, load the default model from HF
        if not os.path.exists(self.conf.model_filename):
            logging.info(
                f'{self.conf.model_filename} does not exist. ' +
                'Dowloading default model from HuggingFace.'
            )
            model_filename = hf_hub_download(
                repo_id=self.conf.hf_repo_id,
                filename=self.conf.hf_model_filename)
        else:
            model_filename = self.conf.model_filename
        logging.info(f'Model filename: {model_filename}')

        # Load model for inference
        model = load_model(
            model_filename=model_filename,
            model_dir=self.model_dir,
            conf=self.conf
        )

        # Retrieve mean and std, there should be a more ideal place
        if self.conf.standardization in ["global", "mixed"]:
            mean, std = get_mean_std_metadata(
                os.path.join(
                    self.model_dir,
                    f'mean-std-{self.conf.experiment_name}.csv'
                )
            )
            logging.info(f'Mean: {mean}, Std: {std}')
        else:
            mean = None
            std = None

        # gather metadata
        if self.conf.metadata_regex is not None:
            metadata = read_metadata(
                self.conf.metadata_regex,
                self.conf.input_bands,
                self.conf.output_bands
            )

        # Gather filenames to predict
        if len(self.conf.inference_regex_list) > 0:
            data_filenames = self.get_filenames(self.conf.inference_regex_list)
        else:
            data_filenames = self.get_filenames(self.conf.inference_regex)
        logging.info(f'{len(data_filenames)} files to predict')

        # iterate files, create lock file to avoid predicting the same file
        for filename in sorted(data_filenames):

            # start timer
            start_time = time.time()

            # set output directory
            basename = os.path.basename(os.path.dirname(filename))
            if basename == 'M1BS' or basename == 'P1BS':
                basename = os.path.basename(
                    os.path.dirname(os.path.dirname(filename)))

            output_directory = os.path.join(
                self.conf.inference_save_dir, basename)
            os.makedirs(output_directory, exist_ok=True)

            # set prediction output filename
            output_filename = os.path.join(
                output_directory,
                f'{Path(filename).stem}.{self.conf.experiment_type}.tif')

            # lock file for multi-node, multi-processing
            lock_filename = f'{output_filename}.lock'

            # predict only if file does not exist and no lock file
            if not os.path.isfile(output_filename) and \
                    not os.path.isfile(lock_filename):

                try:

                    logging.info(f'Starting to predict {filename}')

                    # if metadata is available
                    if self.conf.metadata_regex is not None:

                        # get timestamp from filename
                        year_match = re.search(
                            r'(\d{4})(\d{2})(\d{2})', filename)
                        timestamp = str(int(year_match.group(2)))

                        # get monthly values
                        mean = metadata[timestamp]['median'].to_numpy()
                        std = metadata[timestamp]['std'].to_numpy()
                        self.conf.standardization = 'global'

                    # create lock file
                    open(lock_filename, 'w').close()

                    # open filename
                    image = rxr.open_rasterio(filename)

                    # handle NTF files here, that differ from GeoTIFFs
                    if Path(filename).suffix == '.ntf':
                        image = image[0].to_array().squeeze()

                    logging.info(f'Prediction shape: {image.shape}')

                    # check bands in imagery, do not proceed if one band
                    if image.shape[0] == 1:
                        logging.info(
                            'Skipping file because of non sufficient bands')
                        continue

                except rasterio.errors.RasterioIOError:
                    logging.info(f'Skipped {filename}, probably corrupted.')
                    continue

                # Calculate indices and append to the original raster
                image = indices.add_indices(
                    xraster=image, input_bands=self.conf.input_bands,
                    output_bands=self.conf.output_bands)

                # Modify the bands to match inference details
                image = modify_bands(
                    xraster=image, input_bands=self.conf.input_bands,
                    output_bands=self.conf.output_bands)
                logging.info(f'Prediction shape after modf: {image.shape}')

                logging.info(
                    f'Prediction min={image.min().values}, ' +
                    f'max={image.max().values}')

                # Transpose the image for channel last format
                image = image.transpose("y", "x", "band")

                # Remove no-data values to account for edge effects
                temporary_tif = xr.where(image > -100, image, 600)

                # Sliding window prediction
                prediction, probability = \
                    inference.sliding_window_tiler_multiclass(
                        xraster=temporary_tif,
                        model=model,
                        n_classes=self.conf.n_classes,
                        overlap=self.conf.inference_overlap,
                        batch_size=self.conf.pred_batch_size,
                        threshold=self.conf.inference_treshold,
                        standardization=self.conf.standardization,
                        mean=mean,
                        std=std,
                        normalize=self.conf.normalize,
                        rescale=self.conf.rescale,
                        window=self.conf.window_algorithm,
                        probability_map=self.conf.probability_map
                    )

                # apply default postprocessing
                prediction = self.postprocessing(
                    prediction, self.conf.postprocessing_steps)

                # get cloud metrics for metadata
                cloud_metadata = self.calc_metadata(prediction)

                # Drop image band to allow for a merge of mask
                image = image.drop(
                    dim="band",
                    labels=image.coords["band"].values[1:],
                )

                # Get metadata to save raster prediction
                prediction = xr.DataArray(
                    np.expand_dims(prediction, axis=-1),
                    name=self.conf.experiment_type,
                    coords=image.coords,
                    dims=image.dims,
                    attrs=image.attrs
                )

                # Add metadata to raster attributes
                prediction.attrs['long_name'] = (self.conf.experiment_type)
                prediction.attrs['model_name'] = (model_filename)

                # add cloud metadata to raster attributes
                for mkey, mvalue in cloud_metadata.items():
                    prediction.attrs[mkey] = (mvalue)

                # transpose prediction for saving output
                prediction = prediction.transpose("band", "y", "x")

                # Set nodata values on mask
                nodata = prediction.rio.nodata
                prediction = prediction.where(image != nodata)
                prediction.rio.write_nodata(
                    self.conf.prediction_nodata, encoded=True, inplace=True)

                # Save output raster file to disk
                prediction.rio.to_raster(
                    output_filename,
                    BIGTIFF="IF_SAFER",
                    compress=self.conf.prediction_compress,
                    driver=self.conf.prediction_driver,
                    dtype=self.conf.prediction_dtype
                )
                del prediction

                # save probability map
                if probability is not None:

                    probability = xr.DataArray(
                        np.expand_dims(probability, axis=-1),
                        name=self.conf.experiment_type,
                        coords=image.coords,
                        dims=image.dims,
                        attrs=image.attrs
                    )

                    # Add metadata to raster attributes
                    probability.attrs['long_name'] = (
                        self.conf.experiment_type)
                    probability.attrs['model_name'] = (
                        model_filename)
                    probability = probability.transpose("band", "y", "x")

                    # Set nodata values on mask
                    nodata = probability.rio.nodata
                    probability = probability.where(image != nodata)
                    probability.rio.write_nodata(
                        self.conf.prediction_nodata,
                        encoded=True, inplace=True
                    )

                    # Save output raster file to disk
                    probability.rio.to_raster(
                        Path(output_filename).with_suffix('.prob.tif'),
                        BIGTIFF="IF_SAFER",
                        compress=self.conf.prediction_compress,
                        driver=self.conf.prediction_driver,
                        dtype='float32'
                    )
                    del probability

                # delete lock file
                try:
                    os.remove(lock_filename)
                except FileNotFoundError:
                    logging.info(f'Lock file not found {lock_filename}')
                    continue

                logging.info(f'Finished processing {output_filename}')
                logging.info(f"{(time.time() - start_time)/60} min")

            # This is the case where the prediction was already saved
            else:
                logging.info(f'{output_filename} is done or lock file exists.')

        return

    # -------------------------------------------------------------------------
    # postprocessing
    # -------------------------------------------------------------------------
    def postprocessing(
                self,
                prediction: np.ndarray,
                postprocessing_steps: list = []
            ) -> np.ndarray:

        # sieve clearing of small objects
        if 'sieve' in postprocessing_steps:
            riofeat.sieve(prediction, 800, prediction, None, 8)

        # cloud smoothing
        if 'smooth' in postprocessing_steps:
            prediction = ndimage.median_filter(prediction, size=20)

        # binary fill holes
        if 'fill' in postprocessing_steps:
            prediction = ndimage.binary_fill_holes(prediction).astype(int)

        # dilate
        if 'dilate' in postprocessing_steps:

            # get prediction and set cv2 format
            prediction = np.uint8(np.squeeze(prediction) * 255)
            _, thresh = cv2.threshold(prediction, 127, 255, 0)

            # gather contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # set output contours
            conts = np.zeros((thresh.shape[0], thresh.shape[1], 3))

            # iterate over each contour and increase size
            for c in contours:
                c = self.scale_contour(c, 1.12)  # 1.05 looks decent
                conts = cv2.drawContours(
                    conts, [c], -1, (0, 255, 0), cv2.FILLED)

            # converting mask into raster
            prediction = color.rgb2gray(conts).astype('int16')
            prediction[prediction > 0] = 1

        return prediction

    # -------------------------------------------------------------------------
    # calc_metadata
    # -------------------------------------------------------------------------
    def calc_metadata(self, prediction: np.ndarray) -> dict:

        # set metadata directory for cloud metadata
        metadata_dict = {}

        #  cloud percentage
        metadata_dict['pct_cloudcover_total'] = self.calc_cloud_percentage(
            prediction)

        return metadata_dict

    # -------------------------------------------------------------------------
    # calc_metadata
    # -------------------------------------------------------------------------
    def calc_cloud_percentage(self, prediction: np.ndarray) -> float:

        # get unique values per class
        unique, counts = np.unique(prediction, return_counts=True)
        unique_dict = dict(zip(unique, counts))

        # get cloudy pixels
        try:
            non_cloud_pixels = unique_dict[0]
        except IndexError:
            non_cloud_pixels = 0

        # get cloudy pixels
        try:
            cloud_pixels = unique_dict[1]
        except IndexError:
            cloud_pixels = 0

        # percent cloud cover
        pct_cloud_cover = round(
            100 * (cloud_pixels / (non_cloud_pixels + cloud_pixels)), 2)

        return pct_cloud_cover

    # -------------------------------------------------------------------------
    # scale_contour
    # -------------------------------------------------------------------------
    def scale_contour(self, cnt: np.ndarray, scale: float) -> np.ndarray:
        M = cv2.moments(cnt)
        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        except ZeroDivisionError:
            cx, cy = 0, 0

        cnt_norm = cnt - [cx, cy]
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        return cnt_scaled.astype(np.int32)
