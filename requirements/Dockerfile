# Import tensorflow-caney image
FROM tensorflow-caney:latest

# Ubuntu needs noninteractive to be forced
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get -y dist-upgrade && \
    apt-get -y install build-essential libsm6 libxext6 libxrender-dev libfontconfig1 && \
    apt-get -y autoremove && \
    rm -rf /var/cache/apt

# Pip
#RUN pip --no-cache-dir install omegaconf rasterio rioxarray xarray geopandas \
#    opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless \
#    tifffile webcolors Pillow seaborn xgboost segmentation-models && \
#    #pip uninstall -y cupy-cuda114 && pip install cupy-cuda115

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
