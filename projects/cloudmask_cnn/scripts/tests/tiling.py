# Working with tiling

import numpy as np
import rioxarray as rxr
import matplotlib.pyplot as plt
import scipy.signal.windows as w
from tiler import Tiler, Merger
import tensorflow as tf
import segmentation_models as sm
import xarray as xr

filename = '/adapt/nobackup/people/mwooten3/Senegal_LCLUC/testForMark/5-toas/WV02_20101020_M1BS_1030010007BBFA00-toa.tif'
#filename = '/adapt/nobackup/projects/ilab/projects/Senegal/LCLUC_Senegal_Cloud/data_fixed/WV03_20150717_0-0_data.tif'
#filename = '/adapt/nobackup/projects/ilab/projects/Senegal/LCLUC_Senegal_Cloud/data_fixed/WV03_20150717_5000-0_data.tif'

model_filename = '/adapt/nobackup/projects/ilab/projects/CloudMask/SRLite/57-0.05.hdf5'
model = tf.keras.models.load_model(model_filename, custom_objects={'iou_score': sm.metrics.iou_score})
mm = 'constant'

image = rxr.open_rasterio(filename)
image = image.transpose("y", "x", "band")
print(image.shape)

tiler1 = Tiler(
            data_shape=image.shape,
            tile_shape=(256, 256, 4),
            channel_dimension=2,
            overlap=0.50,
            mode=mm,
            constant_value=1200
        )

print(image[:, :, :1].shape)

tiler2 = Tiler(
        data_shape=image[:, :, :1].shape,
        tile_shape=(256, 256, 1),
        channel_dimension=2,
        overlap=0.50,
        mode=mm,
        constant_value=1200
    )


new_shape, padding = tiler1.calculate_padding()
tiler1.recalculate(data_shape=new_shape)
tiler2.recalculate(data_shape=new_shape)
padded_image = np.pad(image, padding, mode=mm, constant_values=1200)

# Calculate and apply extra padding, as well as adjust tiling parameters
#new_shape, padding = tiler.calculate_padding()
#tiler.recalculate(data_shape=new_shape)
#padded_image = np.pad(image, padding, mode="reflect")

merger = Merger(tiler=tiler2, window="overlap-tile")#"hann")#"triang")#"barthann")#"overlap-tile")
print(tiler1)
print(tiler2)

for batch_id, batch in tiler1(padded_image, batch_size=512): #(image, batch_size=512):
    batch = model.predict(batch/10000.0)
    merger.add_batch(batch_id, 512, batch)

prediction = merger.merge(extra_padding=padding, dtype=image.dtype)
prediction = np.squeeze(np.where(prediction > 0.5, 1, 0).astype(np.int16))
print(prediction.shape, prediction.min(), prediction.max())

image = image.drop(
           dim="band",
           labels=image.coords["band"].values[1:],
           drop=True
        )

prediction = xr.DataArray(
           np.expand_dims(prediction, axis=-1),
           name='test',
           coords=image.coords,
           dims=image.dims,
           attrs=image.attrs
        )

prediction.attrs['long_name'] = ('test')
prediction = prediction.transpose("band", "y", "x")

nodata = prediction.rio.nodata
prediction = prediction.where(image != nodata)
prediction.rio.write_nodata(nodata, encoded=True, inplace=True)

prediction.rio.to_raster(
      'test.tif', BIGTIFF="IF_SAFER", compress='LZW',
      num_threads='all_cpus')#, driver='COG')
