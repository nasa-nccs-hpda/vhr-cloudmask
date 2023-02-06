README.md# vhr-cloudmask

Python library to perform semantic segmentation of clouds and cloud shadows using
very-high resolution remote sensing imagery by means of GPUs and CPU parallelization
for high performance and commodity base environments. 

We are currently working on tutorials and documentations. Feel to follow this repository
for documentation updates and upcoming tutorials.

[DOI]

![CI Workflow](https://github.com/nasa-nccs-hpda/vhr-cloudmask/actions/workflows/ci.yml/badge.svg)
![CI to DockerHub ](https://github.com/nasa-nccs-hpda/vhr-cloudmask/actions/workflows/dockerhub.yml/badge.svg)
![Code style: PEP8](https://github.com/nasa-nccs-hpda/vhr-cloudmask/actions/workflows/lint.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/nasa-nccs-hpda/vhr-cloudmask/badge.svg?branch=main)](https://coveralls.io/github/nasa-nccs-hpda/vhr-cloudmask?branch=main)


## Objectives

- Library to process remote sensing imagery using GPU and CPU parallelization.
- Machine Learning and Deep Learning cloud segmentation.
- Large-scale image inference.

## Installation

vhr-cloudmask can be installed by itself, but instructions for installing the full environments
are listed under the requirements directory so projects, examples, and notebooks can be run.

Note: PIP installations do not include CUDA libraries for GPU support. Make sure
NVIDIA libraries are installed locally in the system if not using conda.

## Getting Started

``` bash
├── archives              <- Legacy code stored to historical reference
├── docs                  <- Default documentation for working with this project
├── images                <- Store project images
├── notebooks             <- Jupyter notebooks
├── examples              <- Examples for utilizing the library
├── requirements          <- Requirements for installing the dependencies
├── scripts               <- Utility scripts for analysis
├── terragpu              <- Library source code
├── README.md             <- The top-level README for developers using this project
├── CHANGELOG.md          <- Releases documentation
├── LICENSE               <- License documentation
└── setup.py              <- Script to install library
```

## Background

The detection of clouds is one of the first steps in the pre-processing of remotely sensed data. At coarse spatial resolution (> 100 m), clouds are bright and generally distinguishable from other landscape surfaces. At very high-resolution (< 3 m), detecting clouds becomes a significant challenge due to the presence of smaller features, with spectral characteristics similar to other land cover types, and thin (partially transparent) cloud forms. Furthermore, at this resolution, clouds can cover many thousands of pixels, making both the center and boundaries of the clouds prone to pixel contamination and variations in the spectral intensity. Techniques that rely solely on the spectral information of clouds underperform in these situations. In this study, we propose a multi-regional and multi-sensor deep learning approach for the detection of clouds in very high-resolution WorldView satellite imagery. A modified UNet-like convolutional neural network (CNN) was used for the task of semantic segmentation in the regions of Vietnam, Senegal, and Ethiopia strictly using RGB + NIR spectral bands. In addition, we demonstrate the superiority of CNNs cloud predicted mapping accuracy of 81–91%, over traditional methods such as Random Forest algorithms of 57–88%. The best performing UNet model has an overall accuracy of 95% in all regions, while the Random Forest has an overall accuracy of 89%. We conclude with promising future research directions of the proposed methods for a global cloud cover implementation.

## Authors

- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
- Caleb S. Spradlin, caleb.s.spradlin@nasa.gov
- Margaret Wooten, margaret.wooten@nasa.gov

## Contributors

- Andrew Weis, aweis1998@icloud.com
- Brian Lee, brianlee52@bren.ucsb.edu

## Installation

See the build [guide](requirements/README.md).

## Contributing

Please see our [guide for contributing to vhr-cloudmask](CONTRIBUTING.md).

## References

Tutorials will be published under [Medium](https://medium.com/@jordan.caraballo/) for additional support
and development, including how to use the library or any upcoming releases.

Please consider citing this when using vhr-cloudmask in a project. You can use the citation BibTeX to site
bot the software and the article:

```bibtex
@article{caraballo2023optimizing,
  title={Optimizing WorldView-2,-3 cloud masking using machine learning approaches},
  author={Caraballo-Vega, JA and Carroll, ML and Neigh, CSR and Wooten, M and Lee, B and Weis, A and Aronne, M and Alemu, WG and Williams, Z},
  journal={Remote Sensing of Environment},
  volume={284},
  pages={113332},
  year={2023},
  publisher={Elsevier}
}
```

```bibtex
TBD
```

## References

[1] Raschka, S., Patterson, J., & Nolet, C. (2020). Machine learning in python: Main developments and technology trends in data science, machine learning, and artificial intelligence. Information, 11(4), 193.

[2] Paszke, Adam; Gross, Sam; Chintala, Soumith; Chanan, Gregory; et all, PyTorch, (2016), GitHub repository, <https://github.com/pytorch/pytorch>. Accessed 13 February 2020.

[3] Caraballo-Vega, J., Carroll, M., Li, J., & Duffy, D. (2021, December). Towards Scalable & GPU Accelerated Earth Science Imagery Processing: An AI/ML Case Study. In AGU Fall Meeting 2021. AGU.
