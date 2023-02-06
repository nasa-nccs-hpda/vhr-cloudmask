import glob
import shutil
import rioxarray as rxr
from pathlib import Path

input_dir = '/adapt/nobackup/projects/ilab/projects/VHRCloudMask/Ethiopia/data'
output_dir = \
    '/adapt/nobackup/projects/ilab/projects/VHRCloudMask/Ethiopia/data_fixed'

filenames = glob.glob(f'{input_dir}/*.tif')
counter = 0
print(len(filenames))

for f in filenames:

    x = rxr.open_rasterio(f)
    xs = x.shape
    if xs[1] == 5000 and xs[2] == 5000:
        counter = counter + 1
        shutil.move(f, f'{output_dir}/{Path(f).stem}.tif')

print(counter)
