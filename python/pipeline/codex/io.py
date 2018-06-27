import os
import codex
import warnings
import os.path as osp
import numpy as np
from skimage.external.tifffile import imread, imsave, TiffFile

# Define a list of helpful path formats (i.e. these are common and don't necessarily need to be configured
# explicitly everywhere)
FILE_FORMATS = {
    codex.FF_V01: dict(
        raw_image=osp.join('Cyc{cycle:d}_reg{region:d}', '{region:d}_{tile:05d}_Z{z:03d}_CH{channel:d}.tif'),
        best_focus=osp.join('bestFocus', 'reg{region:03d}_X{x:02d}_Y{y:02d}_Z{z:02d}.tif'),
        proc_image='reg{region:03d}_X{x:02d}_Y{y:02d}.tif',
        expr_file='reg{region:03d}_Expression_{type}.txt'
    ),
    # Proposed format (haven't seen this in the wild yet but it may be coming in the future)
    codex.FF_V02: dict(
        raw_image=osp.join('Cyc{cycle:d}_reg{region:d}', 'C{channel:03d}_Z{z:03d}_T{cycle:03d}.tif'),
        best_focus=osp.join('bestFocus', 'R{region:03d}_X{x:03d}_Y{y:03d}_Z{z:03d}.tif'),
        proc_image='R{region:03d}_X{x:03d}_Y{y:03d}.tif',
        expr_file='reg{region:03d}_Expression_{type}.txt'
    ),
    # Format for single region, single cycle Keyence experiments
    codex.FF_V03: dict(
        raw_image=osp.join('Image_{tile:05d}_CH{channel:d}.tif'),
        best_focus=osp.join('bestFocus', 'reg001_X{x:02d}_Y{y:02d}_Z{z:02d}.tif'),
        proc_image='reg001_X{x:02d}_Y{y:02d}.tif',
        expr_file='reg{region:03d}_Expression_{type}.txt'
    ),
    # Format identical to v01 except for case sensitivity on raw image region name
    codex.FF_V04: dict(
        raw_image=osp.join('Cyc{cycle:d}_Reg{region:d}', '{region:d}_{tile:05d}_Z{z:03d}_CH{channel:d}.tif'),
        best_focus=osp.join('bestFocus', 'reg{region:03d}_X{x:02d}_Y{y:02d}_Z{z:02d}.tif'),
        proc_image='reg{region:03d}_X{x:02d}_Y{y:02d}.tif',
        expr_file='reg{region:03d}_Expression_{type}.txt'
    ),
    # Another format for single region, single cycle Keyence experiments
    codex.FF_V05: dict(
        raw_image=osp.join('1_{tile:05d}_Z{z:03d}_CH{channel:d}.tif'),
        best_focus=osp.join('bestFocus', 'reg001_X{x:02d}_Y{y:02d}_Z{z:02d}.tif'),
        proc_image='reg001_X{x:02d}_Y{y:02d}.tif',
        expr_file='reg001_Expression_{type}.txt'
    )
}


def _formats():
    # Return pre-defined formats if configured formats is a string key,
    # otherwise assume the formats are specified as a dictionary compatible
    # with pre-defined path format dictionaries
    formats = codex.get_path_formats()
    if formats in FILE_FORMATS:
        return FILE_FORMATS[formats]
    return eval(formats)


def save_image(file, image, **kwargs):
    """Save image array in ImageJ-compatible format"""
    if not osp.exists(osp.dirname(file)):
        os.makedirs(osp.dirname(file), exist_ok=True)
    imsave(file, image, imagej=True, **kwargs)


def save_csv(file, df, **kwargs):
    """Save image array in ImageJ-compatible format"""
    if not osp.exists(osp.dirname(file)):
        os.makedirs(osp.dirname(file), exist_ok=True)
    df.to_csv(file, **kwargs)


def read_image(file):
    # Ignore tiff metadata warnings from skimage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return imread(file)


def read_tile(file, config):
    """Read a codex-specific 5D image tile

    Technical Note: This is a fairly complex process as it is necessary to deal with the fact that files
    saved using tifffile lose unit length dimensions.  To deal with this fact, the metadata in the image
    is parsed here to ensure that missing dimensions are added back.
    """

    # The "imagej_tags" attribute looks like this for a 5D image with no unit-length dimensions
    # and original shape cycles=2, z=25, channels=2:
    # {'ImageJ': '1.11a', 'axes': 'TZCYX', 'channels': 2, 'frames': 2, 'hyperstack': True,
    # 'images': 100, 'loop': False, 'mode': 'grayscale', 'slices': 25}
    # However, if a unit-length dimension was dropped it simply does not show up in this dict
    with TiffFile(file) as tif:
        page = tif.pages[0]
        tags = page.imagej_tags
        if tags['axes'] != 'TZCYX':
            raise ValueError(
                'Image tile at "{}" has tags indicating that it was not saved in TZCYX format.  '
                'The file should have been saved with this property explicitly set and further '
                'processing of it is likely to be incorrect'.format(file)
            )
        slices = [
            slice(None) if 'frames' in tags else None,
            slice(None) if 'slices' in tags else None,
            slice(None) if 'channels' in tags else None,
            slice(None),
            slice(None)
        ]

        return tif.asarray()[slices]
    # slices = [None if dim == 1 else slice(None) for dim in config.tile_dims]
    # return imread(file)[slices]


def save_tile(file, tile):
    """Save a codex-specific 5D image"""
    if tile.ndim != 5:
        raise ValueError('Expecting tile with 5 dimensions but got tile with shape {}'.format(tile.shape))
    # Save using Imagej format, otherwise channels, cycles, and z planes are 
    # all interpreted as individual slices instead of separate dimensions
    save_image(file, tile, metadata={'axes': 'TZCYX'})


def get_raw_img_path(ireg, itile, icyc, ich, iz):
    index_symlinks = codex.get_raw_index_symlinks()
    args = dict(cycle=icyc + 1, region=ireg + 1, tile=itile + 1, z=iz + 1, channel=ich + 1)
    # Remap indexes of input elements if any explicit overrides have been defined
    args = {k: index_symlinks.get(k, {}).get(v, v) for k, v in args.items()}
    return _formats()['raw_image'].format(**args)


FMT_PROC_IMAGE = 'proc_image'
FMT_PROC_DATA = 'proc_data'
FMT_PROC_EXEC = 'proc_exec'
FMT_CYTO_IMAGE = 'cyto_image'
FMT_CYTO_AGG = 'cyto_agg'
FMT_CYTO_STATS = 'cyto_stats'
FMT_EXTRACT_IMAGE = 'extract_image'


def get_img_path(format_key, ireg, tx, ty):
    return _formats()[format_key].format(region=ireg + 1, x=tx + 1, y=ty + 1)


def get_processor_img_path(ireg, tx, ty):
    return get_img_path(FMT_PROC_IMAGE, ireg, tx, ty)


def get_best_focus_img_path(ireg, tx, ty, best_z):
    return _formats()['best_focus_image'].format(region=ireg + 1, x=tx + 1, y=ty + 1, z=best_z + 1)


def get_best_focus_montage_path(ireg):
    return _formats()['best_focus_montage'].format(region=ireg + 1)


def get_cytometry_stats_path(ireg, tx, ty):
    return _formats()[FMT_CYTO_STATS].format(region=ireg + 1, x=tx + 1, y=ty + 1)


def get_cytometry_image_path(ireg, tx, ty):
    return get_img_path(FMT_CYTO_IMAGE, ireg, tx, ty)


def get_cytometry_agg_path(extension):
    return _formats()[FMT_CYTO_AGG].format(extension=extension)


def get_extract_image_path(ireg, tx, ty, name):
    return _formats()[FMT_EXTRACT_IMAGE].format(region=ireg + 1, x=tx + 1, y=ty + 1, name=name)


def get_processor_data_path():
    return _formats()[FMT_PROC_DATA]


def get_processor_exec_path(date):
    return _formats()[FMT_PROC_EXEC].format(date=date)


def read_raw_microscope_image(path, file_type):
    if file_type == codex.FT_GRAYSCALE:
        return read_image(path)
    elif file_type == codex.FT_KEYENCE_RGB:
        img = read_image(path)
        if img.ndim != 3:
            raise ValueError(
                'With {} file types enabled, raw image at path "{}" should have 3 dims (shape = {})'
                .format(file_type, img.shape)
            )
        # Compute image sum for each channel giving 3 item vector
        ch_sum = np.squeeze(np.apply_over_axes(np.sum, img, [0, 1]))
        if np.sum(ch_sum > 0) > 1:
            raise ValueError('Found more than one channel with information in image file "{}"'.format(path))

        # Select and return the single channel with a non-zero sum
        return img[..., np.argmax(ch_sum)]
    else:
        raise ValueError('Raw file type "{}" is not valid; should be one of {}'.format(file_type, codex.RAW_FILE_TYPES))


