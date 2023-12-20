=============
vhr-cloudmask
=============

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7613207.svg
        :target: https://doi.org/10.5281/zenodo.7613207
.. image:: https://github.com/nasa-nccs-hpda/vhr-cloudmask/actions/workflows/lint.yml/badge.svg
        :target: https://github.com/nasa-nccs-hpda/vhr-cloudmask/actions/workflows/lint.yml
.. image:: https://github.com/nasa-nccs-hpda/vhr-cloudmask/actions/workflows/ci.yml/badge.svg
        :target: https://github.com/nasa-nccs-hpda/vhr-cloudmask/actions/workflows/ci.yml
.. image:: https://github.com/nasa-nccs-hpda/vhr-cloudmask/actions/workflows/dockerhub.yml/badge.svg
        :target: https://github.com/nasa-nccs-hpda/vhr-cloudmask/actions/workflows/dockerhub.yml
.. image:: https://github.com/nasa-nccs-hpda/vhr-cloudmask/actions/workflows/dockerhub.yml/badge.svg
        :target: https://github.com/nasa-nccs-hpda/vhr-cloudmask/actions/workflows/dockerhub.yml
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/psf/black
.. image:: https://coveralls.io/repos/github/nasa-nccs-hpda/vhr-cloudmask/badge.svg?branch=main
        :target: https://coveralls.io/github/nasa-nccs-hpda/vhr-cloudmask?branch=main

**Very High-Resolution Cloud Masking Framework**

Python library to perform semantic segmentation of clouds and cloud shadows using
very-high resolution remote sensing imagery by means of GPUs and CPU parallelization
for high performance and commodity base environments. 

* GitHub repo: https://github.com/nasa-nccs-hpda/vhr-cloudmask
* Documentation: https://nasa-nccs-hpda.github.io/vhr-cloudmask

**Contents**

- `Objectives`_
- `Background`_
- `Getting Started`_
- `Infrastructure`_
- `Data Locations where this Workflow has been Validated`_
- `Development Pipeline Details`_
- `Authors`_
- `Contributors`_
- `Contributing`_
- `Citations`_
- `References`_

Objectives
============

* Library to process remote sensing imagery using GPU and CPU parallelization.
* Machine learning and deep learning cloud segmentation of VHR imagery.
* Large-scale image inference.

Background
============

The detection of clouds is one of the first steps in the pre-processing of remotely sensed data.
At coarse spatial resolution (> 100 m), clouds are bright and generally distinguishable from other
landscape surfaces. At very high-resolution (< 3 m), detecting clouds becomes a significant challenge
due to the presence of smaller features, with spectral characteristics similar to other land cover types,
and thin (partially transparent) cloud forms. Furthermore, at this resolution, clouds can cover many
thousands of pixels, making both the center and boundaries of the clouds prone to pixel contamination
and variations in the spectral intensity. Techniques that rely solely on the spectral information of
clouds underperform in these situations.

In this study, we propose a multi-regional and multi-sensor deep learning approach for the detection of
clouds in very high-resolution WorldView satellite imagery. A modified UNet-like convolutional neural
network (CNN) was used for the task of semantic segmentation in the regions of Vietnam, Senegal, and
Ethiopia strictly using RGB + NIR spectral bands. In addition, we demonstrate the superiority of CNNs
cloud predicted mapping accuracy of 81–91%, over traditional methods such as Random Forest algorithms
of 57–88%. The best performing UNet model has an overall accuracy of 95% in all regions, while the 
Random Forest has an overall accuracy of 89%. We conclude with promising future research directions of 
the proposed methods for a global cloud cover implementation.

Getting Started
=================

The main recommended avenue for using vhr-cloudmask is through the publicly available set of containers
provided via this repository. If containers are not an option for your setup, follow the installation
instructions via PIP.

Downloading the Container
---------------------------

All Python and GPU depenencies are installed in an OCI compliant Docker image. You can
download this image into a Singularity format to use in HPC systems.

.. code:: python

        singularity pull docker://nasanccs/vhr-cloudmask:latest

In some cases, HPC systems require Singularity containers to be built as sandbox environments because
of uid issues (this is the case of NCCS Explore). For that case you can build a sandbox using the following
command. Depending the filesystem, this can take between 5 minutes to an hour.

.. code:: python

        singularity build --sandbox vhr-cloudmask docker://nasanccs/vhr-cloudmask:latest

If you have done this step, you can skip the Installation step since the containers already
come with all dependencies installed.

Installation
--------------

vhr-cloudmask can be installed by itself, but instructions for installing the full environments
are listed under the requirements directory so projects, examples, and notebooks can be run.

Note: PIP installations do not include CUDA libraries for GPU support. Make sure
NVIDIA libraries are installed locally in the system if not using conda.

vhr-cloudmask is available on [PyPI](https://pypi.org/project/vhr-cloudmask/).
To install vhr-cloudmask, run this command in your terminal or from inside a container:

.. code:: python

        pip install vhr-cloudmask

If you have installed vhr-cloudmask before and want to upgrade to the latest version,
you can run the following command in your terminal:

.. code:: python

        pip install -U vhr-cloudmask

Running Inference of Clouds
------------------------------

Use the following command if you need to perform inference using a regex that points
to the necessary files and by leveraging the default global model. The following is
a singularity exec command with options from both Singularity and the cloud masking
application.

Singularity options:

* '-B': mounts a filesystem from the host into the container
* '--nv': mount container binaries/devices

vhr_cloumask_cli options:

* '-r': list of regex strings to find geotiff files to predict from
* '-o': output directory to store cloud masks
* '-s': pipeline step, to generate masks only we want to predict

.. code:: python

        singularity exec --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects \
        /explore/nobackup/projects/ilab/containers/vhr-cloudmask.sif vhr-cloudmask-cli \
        -o '/explore/nobackup/projects/ilab/test/vhr-cloudmask' \
        -r '/explore/nobackup/projects/3sl/data/Tappan/Tappan16*_data.tif' '/explore/nobackup/projects/3sl/data/Tappan/Tappan15*_data.tif' \
        -s predict

To predict via slurm for a large set of files, use the following script which will start a large number
of jobs (up to your processing limit), and process the remaining files.

.. code:: python

        for i in {0..64}; do sbatch --mem-per-cpu=10240 -G1 -c10 -t05-00:00:00 -J clouds --wrap="singularity exec --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/vhr-cloudmask.sif vhr-cloudmask-cli -o '/explore/nobackup/projects/ilab/test/vhr-cloudmask' -r '/explore/nobackup/projects/3sl/data/Tappan/Tappan16*_data.tif' '/explore/nobackup/projects/3sl/data/Tappan/Tappan15*_data.tif' -s predict"; done

Infrastructure
=================

The vhr-cloudmask package is a set of CLI tools and Jupyter-based notebooks to manage and
structure the validation of remote sensing data. The CLI tools can be run from inside a container
or from any system where the vhr-cloudmask package is installed.

The main system requirements from this package are a system with GPUs to accelerate the training and
inference of imagery. If no GPU is available, the process will continue as expected but with a large
slowdown. There are no minimum system memory requirements given the sliding window procedures
implemented in the inference process.

Data Locations where this Workflow has been Validated
========================================================

The vhr-cloudmask workflow has been validated in the following study areas
using WorldView imagery. Additional areas will be included into our validation
suite as part of upcoming efforts to improve the scalability of our models.

- Senegal
- Vietnam
- Ethiopia
- Oregon
- Alaska
- Whitesands
- Siberia

Development Pipeline Details
==============================

When performing development (training a model, preprocessing, etc.), we want to run from the 
dev container so we can add the Python files to the PYTHONPATH. The following commmand is an example
command to run inference given a configuration file.

.. code:: python

        singularity exec --env PYTHONPATH="$NOBACKUP/development/tensorflow-caney:$NOBACKUP/development/vhr-cloudmask" \
        --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects \
        /explore/nobackup/projects/ilab/containers/vhr-cloudmask.sif \
        python $NOBACKUP/development/vhr-cloudmask/vhr_cloudmask/view/cloudmask_cnn_pipeline_cli.py \
        -c $NOBACKUP/development/vhr-cloudmask/projects/cloud_cnn/configs/production/cloud_mask_alaska_senegal_3sl_cas.yaml \
        -s predict

If you do not have access to modify the configuration file, or just need to perform small changes to the model selection,
the regex to the files to predict, or the output directory, manually specify the arguments to the CLI file:

.. code:: python

        singularity exec --env PYTHONPATH="$NOBACKUP/development/tensorflow-caney:$NOBACKUP/development/vhr-cloudmask" \
        --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects \
        /explore/nobackup/projects/ilab/containers/vhr-cloudmask.sif \
        python $NOBACKUP/development/vhr-cloudmask/vhr_cloudmask/view/cloudmask_cnn_pipeline_cli.py \
        -c $NOBACKUP/development/vhr-cloudmask/projects/cloud_cnn/configs/production/cloud_mask_alaska_senegal_3sl_cas.yaml \
        -o '/explore/nobackup/projects/ilab/test/vhr-cloudmask' \
        -r '/explore/nobackup/projects/3sl/data/Tappan/Tappan16*_data.tif' '/explore/nobackup/projects/3sl/data/Tappan/Tappan15*_data.tif' \
        -s predict

Authors
====================

* Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
* Caleb S. Spradlin, caleb.s.spradlin@nasa.gov
* Margaret Wooten, margaret.wooten@nasa.gov

Contributors
====================

* Andrew Weis, aweis1998@icloud.com
* Brian Lee, brianlee52@bren.ucsb.edu

Contributing
====================

Please see our [guide for contributing to vhr-cloudmask](CONTRIBUTING.md). Contributions
are welcome, and they are greatly appreciated! Every little bit helps, and credit will
always be given.

You can contribute in many ways:

Report Bugs
-------------

Report bugs at https://github.com/nasa-nccs-hpda/vhr-cloudmask/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
-------------

Look through the GitHub issues for bugs. Anything tagged with "bug" and
"help wanted" is open to whoever wants to implement it.

Implement Features
--------------------

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is
open to whoever wants to implement it.

Write Documentation
------------------------

vhr-cloudmask could always use more documentation, whether as part of the official vhr-cloudmask docs,
in docstrings, or even on the web in blog posts, articles, and such.

Submit Feedback
--------------------

The best way to send feedback is to file an issue at https://github.com/nasa-nccs-hpda/vhr-cloudmask/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions are welcome :)

Citations
============

Tutorials will be published under [Medium](https://medium.com/@jordan.caraballo/) for additional support
and development, including how to use the library or any upcoming releases.

If you find this code or methodology useful, please consider citing the following paper and/or code.

* Caraballo-Vega, J. A., Carroll, M. L., Neigh, C. S. R., Wooten, M., Lee, B., Weis, A., ... & Williams, Z. (2023).
  Optimizing WorldView-2,-3 cloud masking using machine learning approaches. Remote Sensing of Environment, 284, 113332.
* Jordan Alexis Caraballo-Vega. (2023). nasa-nccs-hpda/vhr-cloudmask: 1.2.0 (1.2.0). Zenodo. https://doi.org/10.5281/zenodo.10408125

References
============

* Raschka, S., Patterson, J., & Nolet, C. (2020). Machine learning in python: Main developments and technology trends in data science, machine learning, and artificial intelligence. Information, 11(4), 193.
* Paszke, Adam; Gross, Sam; Chintala, Soumith; Chanan, Gregory; et all, PyTorch, (2016), GitHub repository, <https://github.com/pytorch/pytorch>. Accessed 13 February 2020.
* Caraballo-Vega, J., Carroll, M., Li, J., & Duffy, D. (2021, December). Towards Scalable & GPU Accelerated Earth Science Imagery Processing: An AI/ML Case Study. In AGU Fall Meeting 2021. AGU.
