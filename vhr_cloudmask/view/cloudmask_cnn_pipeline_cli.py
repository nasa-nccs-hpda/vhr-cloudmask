import sys
import time
import logging
import argparse
from vhr_cloudmask.model.pipelines.cloudmask_cnn_pipeline import \
    CloudMaskPipeline


# -----------------------------------------------------------------------------
# main
#
# python cloudmask_pipeline_cli.py -c config.yaml -d config.csv -s preprocess
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to perform CNN segmentation.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=False,
                        default=None,
                        dest='config_file',
                        help='Path to the configuration file')

    parser.add_argument('-d',
                        '--data-csv',
                        type=str,
                        required=False,
                        default=None,
                        dest='data_csv',
                        help='Path to the data configuration file')

    parser.add_argument('-s',
                        '--step',
                        type=str,
                        nargs='*',
                        required=True,
                        dest='pipeline_step',
                        help='Pipeline step to perform',
                        default=['preprocess', 'train', 'predict'],
                        choices=['preprocess', 'train', 'predict'])

    parser.add_argument('-m',
                        '--model-filename',
                        type=str,
                        required=False,
                        default=None,
                        dest='model_filename',
                        help='Path to model file')

    parser.add_argument('-o',
                        '--output-dir',
                        type=str,
                        default=None,
                        required=False,
                        dest='output_dir',
                        help='Path to output directory')

    parser.add_argument('-r',
                        '--regex-list',
                        type=str,
                        nargs='*',
                        required=False,
                        dest='inference_regex_list',
                        help='Inference regex list',
                        default=['*.tif'])

    parser.add_argument('-ib',
                        '--input-bands',
                        type=str,
                        nargs='*',
                        required=False,
                        dest='input_bands',
                        help='Inputs bands from incoming data',
                        default=None)

    parser.add_argument('-ob',
                        '--output-bands',
                        type=str,
                        nargs='*',
                        required=False,
                        dest='output_bands',
                        help='Output bands from incoming data',
                        default=None)

    parser.add_argument('-ps',
                        '--postprocessing-steps',
                        type=str,
                        nargs='*',
                        required=False,
                        dest='postprocessing_steps',
                        help='Postprocessing steps to perform',
                        default=['sieve', 'smooth', 'fill', 'dilate'],
                        choices=['sieve', 'smooth', 'fill', 'dilate'])

    args = parser.parse_args()

    # Setup timer to monitor script execution time
    timer = time.time()

    # Initialize pipeline object
    pipeline = CloudMaskPipeline(
        args.config_file,
        args.data_csv,
        args.model_filename,
        args.output_dir,
        args.inference_regex_list,
        args.input_bands,
        args.output_bands,
        args.postprocessing_steps
    )

    # Regression CHM pipeline steps
    if "preprocess" in args.pipeline_step:
        pipeline.preprocess()
    if "train" in args.pipeline_step:
        pipeline.train()
    if "predict" in args.pipeline_step:
        pipeline.predict()

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
