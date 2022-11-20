import xarray as xr
import numpy as np
import cv2 as cv
import rioxarray as rxr
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import rasterio.features as riofeat  # rasterio features include sieve filter
import rasterio
from glob import glob
from pathlib import Path
import os

data_filenames = glob('/adapt/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1/Alaska/*.tif')
output_dir = '/adapt/nobackup/projects/ilab/projects/CloudMask/products/srlite/v1.2/Alaska'
print(len(data_filenames))

os.makedirs(output_dir, exist_ok=True)

for filename in data_filenames:

    output_filename = os.path.join(output_dir, f'{Path(filename).stem}.v1.2.tif')

    image = rxr.open_rasterio(filename)
    print(image.min().values)
    data = np.squeeze(image.copy().values)
    data[data < 0] = 0

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(20,20))
    data_array = cv.morphologyEx(data, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(60,60)))#np.ones((50,50),np.uint8))
    data_array = cv.morphologyEx(data_array, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(40,40)))#np.ones((10,10),np.uint8))
    data_array = cv.dilate(data_array,cv.getStructuringElement(cv.MORPH_ELLIPSE,(50,50)), iterations = 1)
    data_array = ndimage.median_filter(data_array, size=20)

    data = data_array

    # Get metadata to save raster
    data = xr.DataArray(
            np.expand_dims(data, axis=0), #-1
                name='cloudmask',
                    coords=image.coords,
                        dims=image.dims,
                            attrs=image.attrs
                            )
    data.attrs['long_name'] = ('cloudmask')
    #data = data.transpose("band", "y", "x")

    # Set nodata values on mask
    nodata = image.rio.nodata
    data = data.where(image != nodata)

    data.rio.write_nodata(nodata, encoded=True, inplace=True)

    #data.min()

    # Save COG file to disk
    data.rio.to_raster(
            output_filename, BIGTIFF="IF_SAFER", compress='LZW',
                num_threads='all_cpus')#, driver='COG')

