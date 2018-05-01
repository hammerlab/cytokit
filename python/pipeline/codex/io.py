import os
import codex
import warnings
import os.path as osp
from collections import namedtuple
from skimage.external.tifffile import imread, imsave


FileFormats = namedtuple('FileFormats', ['raw_image', 'best_focus', 'proc_image', 'expr_file'])
FILE_FORMATS = {
    codex.FF_V01: FileFormats(
        raw_image=osp.join('Cyc{cycle:d}_reg{region:d}', '1_{tile:05d}_Z{z:03d}_CH{channel:d}.tif'),
        best_focus=osp.join('bestFocus', 'reg{region:03d}_X{x:02d}_Y{y:02d}_Z{z:02d}.tif'),
        proc_image='reg{region:03d}_X{x:02d}_Y{y:02d}.tif',
        expr_file='reg{region:03d}_Expression_Compensated.txt'
    ),
    codex.FF_V02: FileFormats(
        raw_image=osp.join('Cyc{cycle:d}_reg{region:d}', 'C{channel:03d}_Z{z:03d}_T{cycle:03d}.tif'),
        best_focus=osp.join('bestFocus', 'R{region:03d}_X{x:03d}_Y{y:03d}_Z{z:03d}.tif'),
        proc_image='R{region:03d}_X{x:03d}_Y{y:03d}.tif',
        expr_file='reg{region:03d}_Expression_Compensated.txt'
    ),
    # Format for single region, single cycle Keyence experiments
    codex.FF_V03: FileFormats(
        raw_image=osp.join('Image_{tile:05d}_CH{channel:d}.tif'),
        best_focus=osp.join('bestFocus', 'reg001_X{x:02d}_Y{y:02d}_Z{z:02d}.tif'),
        proc_image='reg001_X{x:02d}_Y{y:02d}.tif',
        expr_file='reg{region:03d}_Expression_Compensated.txt'
    ),
}


def _formats():
    return FILE_FORMATS[codex.get_file_format_version()]


def save_image(file, image, **kwargs):
    """Save image array in ImageJ-compatible format"""
    if not osp.exists(osp.dirname(file)):
        os.makedirs(osp.dirname(file), exist_ok=True)
    imsave(file, image, imagej=True, **kwargs)


def read_image(file):
    # Ignore tiff metadata warnings from skimage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return imread(file)

def read_tile(file, config):
    """Read a codex_app-specific 5D image tile"""
    # When saving tiles in ImageJ compatible format, any unit length
    # dimensions are lost so when reading them back out, it is simplest
    # to conform to 5D convention by reshaping if necessary
    slices = [None if dim == 1 else slice(None) for dim in config.tile_dims]
    return imread(file)[slices]


def save_tile(file, tile):
    """Save a codex_app-specific 5D image"""
    if tile.ndim != 5:
        raise ValueError('Expecting tile with 5 dimensions but got tile with shape {}'.format(tile))
    # Save using Imagej format, otherwise channels, cycles, and z planes are 
    # all interpreted as individual slices instead of separate dimensions
    save_image(file, tile, metadata={'axes': 'TZCYX'})


def get_raw_img_path(ireg, itile, icyc, ich, iz):
    index_symlinks = codex.get_raw_index_symlinks()
    args = dict(cycle=icyc + 1, region=ireg + 1, tile=itile + 1, z=iz + 1, channel=ich + 1)
    # Remap indexes of input elements if any explicit overrides have been defined
    args = {k: index_symlinks.get(k, {}).get(v, v) for k, v in args.items()}
    return _formats().raw_image.format(**args)


def get_processor_img_path(ireg, tx, ty):
    return _formats().proc_image.format(region=ireg + 1, x=tx + 1, y=ty + 1)


def get_best_focus_img_path(ireg, tx, ty, best_z):
    return _formats().best_focus.format(region=ireg + 1, x=tx + 1, y=ty + 1, z=best_z + 1)


def get_region_expression_path(ireg):
    return _formats().expr_file.format(region=ireg + 1)





