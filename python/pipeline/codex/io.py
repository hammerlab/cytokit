import os
import codex
import warnings
import os.path as osp
import numpy as np
from skimage import io as sk_io
from skimage.external.tifffile import imread, imsave, TiffFile

# Define the default layout for cytokit results on disk
DEFAULT_FORMATS = dict(
    best_focus_image='best_focus/tile/R{region:03d}_X{x:03d}_Y{y:03d}_Z{z:03d}.tif',
    montage_image='montage/{name}/R{region:03d}_montage.tif',
    cyto_agg='cytometry/data.{extension}',
    cyto_image='cytometry/tile/R{region:03d}_X{x:03d}_Y{y:03d}.tif',
    cyto_stats='cytometry/statistics/R{region:03d}_X{x:03d}_Y{y:03d}.csv',
    proc_data='processor/data.json',
    proc_exec='processor/execution/{date}.json',
    proc_image='processor/tile/R{region:03d}_X{x:03d}_Y{y:03d}.tif',
    extract_image='extract/{name}/R{region:03d}_X{x:03d}_Y{y:03d}.tif'
)


def _to_pd_sep(format):
    """Convert path to platform-dependent path"""
    return format.replace('/', os.sep)


def _get_def_path_formats(raw_image_format):
    formats = {k: _to_pd_sep(v) for k, v in DEFAULT_FORMATS.items()}
    formats['raw_image'] = _to_pd_sep(raw_image_format)
    return formats


PATH_FORMATS = dict(
    keyence_single_cycle_v01=_get_def_path_formats('1_{tile:05d}_Z{z:03d}_CH{channel:d}.tif'),
    keyence_single_cycle_v02=_get_def_path_formats('1_{tile:05d}_Z{z:03d}_CH{channel:d}.jpg'),
    keyence_multi_cycle_v01=_get_def_path_formats(
        'Cyc{cycle:d}_reg{region:d}/{region:d}_{tile:05d}_Z{z:03d}_CH{channel:d}.tif')
)


def _formats():
    # Return pre-defined formats if configured formats is a string key,
    # otherwise assume the formats are specified as a dictionary compatible
    # with pre-defined path format dictionaries
    formats = codex.get_path_formats()
    if formats in PATH_FORMATS:
        return PATH_FORMATS[formats]
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
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     return sk_io.imread(file)
    return sk_io.imread(file)


def read_tile(file, config=None):
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
        if 'axes' not in tags:
            warnings.warn('ImageJ tags do not contain "axes" property (file = {}, tags = {})'.format(file, tags))
        else:
            if tags['axes'] != 'TZCYX':
                warnings.warn(
                    'Image has tags indicating that it was not saved in TZCYX format.  '
                    'The file should have been saved with this property explicitly set and further '
                    'processing of it may be unsave (file = {})'.format(file)
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
FMT_FOCUS_IMAGE = 'best_focus_image'
FMT_MONTAGE_IMAGE = 'montage_image'
FMT_CYTO_IMAGE = 'cyto_image'
FMT_CYTO_AGG = 'cyto_agg'
FMT_CYTO_STATS = 'cyto_stats'
FMT_EXTRACT_IMAGE = 'extract_image'


def get_img_path(format_key, ireg, tx, ty):
    return _formats()[format_key].format(region=ireg + 1, x=tx + 1, y=ty + 1)


def get_processor_img_path(ireg, tx, ty):
    return get_img_path(FMT_PROC_IMAGE, ireg, tx, ty)


def get_best_focus_img_path(ireg, tx, ty, best_z):
    return _formats()[FMT_FOCUS_IMAGE].format(region=ireg + 1, x=tx + 1, y=ty + 1, z=best_z + 1)


def get_cytometry_stats_path(ireg, tx, ty):
    return _formats()[FMT_CYTO_STATS].format(region=ireg + 1, x=tx + 1, y=ty + 1)


def get_cytometry_image_path(ireg, tx, ty):
    return get_img_path(FMT_CYTO_IMAGE, ireg, tx, ty)


def get_cytometry_agg_path(extension):
    return _formats()[FMT_CYTO_AGG].format(extension=extension)


def get_extract_image_path(ireg, tx, ty, name):
    return _formats()[FMT_EXTRACT_IMAGE].format(region=ireg + 1, x=tx + 1, y=ty + 1, name=name)


def get_montage_image_path(ireg, name):
    return _formats()[FMT_MONTAGE_IMAGE].format(region=ireg + 1, name=name)


def get_processor_data_path():
    return _formats()[FMT_PROC_DATA]


def get_processor_exec_path(date):
    return _formats()[FMT_PROC_EXEC].format(date=date)


def _collapse_keyence_rgb(path, img):
    # Compute image sum for each channel giving 3 item vector
    ch_sum = np.squeeze(np.apply_over_axes(np.sum, img, [0, 1]))
    if np.sum(ch_sum > 0) > 1:
        raise ValueError('Found more than one channel with information in image file "{}"'.format(path))

    # Select and return the single channel with a non-zero sum
    return img[..., np.argmax(ch_sum)]


def read_raw_microscope_image(path, file_type):
    if file_type == codex.FT_GRAYSCALE:
        img = read_image(path)
    elif file_type == codex.FT_KEYENCE_RGB:
        img = read_image(path)
        if img.ndim != 3:
            raise ValueError(
                'With {} file types enabled, raw image at path "{}" should have 3 dims (shape = {})'
                .format(file_type, path, img.shape)
            )
        img = _collapse_keyence_rgb(path, img)
    elif file_type == codex.FT_KEYENCE_REPEAT:
        img = read_image(path)
        if img.ndim != 3:
            raise ValueError(
                'With {} file types enabled, raw image at path "{}" should have 3 dims (shape = {})'
                .format(file_type, path, img.shape)
            )
        if not np.all(img[..., 0] == img[..., 0]) or not np.all(img[..., 0] == img[..., 2]):
            raise ValueError(
                'With {} file types enabled, all 3 channels are expected to be equal but they are not '
                'for image at "{}" (shape = {})'.format(file_type, path, img.shape)
            )
        img = img[..., 0]

        # TODO: REMOVE THIS!
        from skimage import transform
        from skimage import exposure
        img = transform.rescale(img, 4, mode='constant', multichannel=False, anti_aliasing=True)
        img = exposure.rescale_intensity(img, in_range=(0, 1), out_range='uint8').astype(np.uint8)
        assert img.shape == (1440, 1920)

    elif file_type == codex.FT_KEYENCE_MIXED:
        img = read_image(path)
        if img.ndim not in [2, 3]:
            raise ValueError(
                'With {} file types enabled, raw image at path "{}" should have 2 or 3 dims (shape = {})'
                .format(file_type, path, img.shape)
            )
        if img.ndim == 3:
            img = _collapse_keyence_rgb(path, img)
    else:
        raise ValueError('Raw file type "{}" is not valid; should be one of {}'.format(file_type, codex.RAW_FILE_TYPES))

    # Validate that all file types result in 2D image
    if img.ndim != 2:
        raise AssertionError(
            'Raw data file "{}" (with file type {} enabled) not processed correctly; should have resulted in '
            'single channel image but shape is {}'.format(path, file_type, img.shape))
    return img

