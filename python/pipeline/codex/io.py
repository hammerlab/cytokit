import os
import codex
import warnings
import os.path as osp
from collections import namedtuple
from skimage.external.tifffile import imread, imsave


FileFormats = namedtuple('FileFormats', ['raw_image', 'best_focus', 'proc_image'])
FILE_FORMATS = {
    codex.FF_V01: FileFormats(
        raw_image=osp.join('Cyc{cycle:d}_reg{region:d}', '1_{tile:05d}_Z{z:03d}_CH{channel:d}.tif'),
        best_focus=osp.join('bestFocus', 'reg{region:03d}_X{x:02d}_Y{y:02d}_Z{z:02d}.tif'),
        proc_image='reg{region:03d}_X{x:02d}_Y{y:02d}.tif'
    ),
    codex.FF_V02: FileFormats(
        raw_image=osp.join('Cyc{cycle:d}_reg{region:d}', 'C{channel:03d}_Z{z:03d}_T{cycle:03d}.tif'),
        best_focus=osp.join('bestFocus', 'R{region:03d}_X{x:03d}_Y{y:03d}_Z{z:03d}.tif'),
        proc_image='R{region:03d}_X{x:03d}_Y{y:03d}.tif'
    ),
}


def _formats():
    return FILE_FORMATS[codex.get_file_format_version()]


def read_tile(file):
    # Ignore tiff metadata warnings from skimage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return imread(file)


def save_tile(file, tile):
    if not osp.exists(osp.dirname(file)):
        os.makedirs(osp.dirname(file), exist_ok=True)
    # Save using Imagej format (this is crucial for 5D stacks)
    imsave(file, tile, imagej=True)


def get_raw_img_path(ireg, itile, icyc, ich, iz):
    return _formats().raw_image.format(cycle=icyc + 1, region=ireg + 1, tile=itile + 1, z=iz + 1, channel=ich + 1)


def get_processor_img_path(ireg, tx, ty):
    return _formats().proc_image.format(region=ireg + 1, x=tx + 1, y=ty + 1)


def get_best_focus_img_path(ireg, tx, ty, best_z):
    return _formats().best_focus.format(region=ireg + 1, x=tx + 1, y=ty + 1, z=best_z + 1)





