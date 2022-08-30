#!/usr/bin/env python

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

from setuptools import setup, find_packages
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='vhr_cloudmask',
    version='0.0.1',
    description='Methods for the acceleration of remote sensing satellite imagery',
    author='Jordan A. Caraballo-Vega',
    author_email='jordan.a.caraballo-vega@nasa.gov',
    zip_safe=False,
    url='https://github.com/cist-ai/vhr_cloudmask.git',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
