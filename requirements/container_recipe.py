#!/usr/bin/env python

from __future__ import print_function

import argparse
import hpccm
from hpccm.building_blocks import pip
from hpccm.primitives import baseimage

parser = argparse.ArgumentParser(description='HPCCM Tutorial')
parser.add_argument('--format', type=str, default='docker',
                    choices=['docker', 'singularity'],
                    help='Container specification format (default: docker)')
parser.add_argument('--baseimage', type=str,
                    default='nvcr.io/nvidia/tensorflow:21.12-tf2-py3',
                    help='Base container image')
parser.add_argument('--distro', type=str,
                    default='ubuntu20',
                    help='Distribution of base container image')
args = parser.parse_args()

Stage0 = hpccm.Stage()
Stage1 = hpccm.Stage()

# Start "Recipe"
Stage0 += baseimage(
    image=args.baseimage, _distro='ubuntu20')

# Install PIP packages
Stage0 += pip(
    pip='pip3',
    packages=[
        'omegaconf',
        'rasterio',
        'rioxarray',
        'xarray',
        'geopandas',
        'opencv-python',
        'opencv-python-headless',
        'opencv-contrib-python',
        'opencv-contrib-python-headless',
        'segmentation-models'
    ])

# End "Recipe"

hpccm.config.set_container_format(args.format)

print(Stage0)
