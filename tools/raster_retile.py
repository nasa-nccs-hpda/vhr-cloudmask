import os
import sys
from itertools import product
import rasterio as rio
from rasterio import windows

# in_path = '/path/to/indata/'
input_filename = sys.argv[1]

out_path = sys.argv[2]  # '/path/to/output_folder/'
output_filename = sys.argv[1].split('/')[-1][:-4] + '_{}-{}.tif'


def get_tiles(ds, width=5000, height=5000):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in offsets:
        window = windows.Window(
            col_off=col_off, row_off=row_off, width=width, height=height
        ).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


with rio.open(input_filename) as inds:
    tile_width, tile_height = 5000, 5000

    meta = inds.meta.copy()

    for window, transform in get_tiles(inds):
        print(window)
        meta['transform'] = transform
        meta['width'], meta['height'] = window.width, window.height
        outpath = os.path.join(
            out_path, output_filename.format(
                int(window.col_off), int(window.row_off)
            )
        )
        with rio.open(outpath, 'w', **meta) as outds:
            outds.write(inds.read(window=window))
