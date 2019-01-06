"""
IO and Path Utilities

Note that most image IO assumes ImageJ metadata handling only and that currently, this is supported via the
independent tifffile package and not the one bundled with scikit-image (which does not make the same
metadata management possible)

References:
    - scikit-image tifffile: https://github.com/scikit-image/scikit-image/blob/master/skimage/external/tifffile/tifffile.py
    - pypi tifffile: https://github.com/blink1073/tifffile/blob/master/tifffile/tifffile.py
    - SO post 1 on metadata: https://stackoverflow.com/questions/50258287/how-to-specify-colormap-when-saving-tiff-stack
    - SO post 2 on metadata: https://stackoverflow.com/questions/50948559/how-to-save-imagej-tiff-metadata-using-python?noredirect=1&lq=1
"""
import os
import cytokit
import warnings
import os.path as osp
import numpy as np
import cytokit
from cytokit.utils import ij_utils
from skimage import io as sk_io
from tifffile import imread, imsave, TiffFile


def _to_pd_sep(format):
    """Convert path to platform-dependent path"""
    return format.replace('/', os.sep)


def _get_def_path_formats(raw_image_format):
    formats = {k: _to_pd_sep(v) for k, v in cytokit.DEFAULT_PATH_FORMATS.items()}
    formats['raw_image'] = _to_pd_sep(raw_image_format)
    return formats


PATH_FORMATS = dict(
    keyence_single_cycle_v01=_get_def_path_formats('1_{tile:05d}_Z{z:03d}_CH{channel:d}.tif'),
    keyence_single_cycle_v02=_get_def_path_formats('1_{tile:05d}_Z{z:03d}_CH{channel:d}.jpg'),
    keyence_single_cycle_v03=_get_def_path_formats('1_XY01_{tile:05d}_Z{z:03d}_CH{channel:d}.tif'),
    keyence_multi_cycle_v01=_get_def_path_formats(
        'Cyc{cycle:d}_reg{region:d}/{region:d}_{tile:05d}_Z{z:03d}_CH{channel:d}.tif'),
    keyence_multi_cycle_v02=_get_def_path_formats(
        'Cyc{cycle:d}_Reg{region:d}/{region:d}_{tile:05d}_Z{z:03d}_CH{channel:d}.tif'),
    keyence_multi_cycle_v03=_get_def_path_formats(
        'Cyc{cycle:d}_reg{region:d}/Image_{tile:05d}_Z{z:03d}_CH{channel:d}.tif')
)


def _formats():
    # Return pre-defined formats if configured formats is a string key,
    # otherwise assume the formats are specified as a dictionary compatible
    # with pre-defined path format dictionaries
    formats = cytokit.get_path_formats()
    if formats in PATH_FORMATS:
        return PATH_FORMATS[formats]
    return eval(formats)


def save_image(file, image, imagej=True, **kwargs):
    """Save tif image (with default to ImageJ format)

    Args:
        file: File path to save to; will overwrite if exists and will also create directory if not present
        image: Image array to save
        kwargs: Anything compatible with tifffile.imsave (see
            https://github.com/blink1073/tifffile/blob/master/tifffile/tifffile.py)
    Examples:
        - Setting channel names within a hyperstack image:
            # Note that z and t are used to repeat names across Z and T axes
            tags = ij_utils.get_channel_label_tags(['CH1', 'CH2', 'CH3'], z=1, t=1)
            io.save_image('test.tif', image, extratags=tags)
        - Setting XYZ resolution (from https://github.com/blink1073/tifffile/issues/15#issuecomment-224715191):
            io.save_image(
                'test.tif', image,
                resolution=(0.373759, 0.373759),
                metadata={'spacing': 3.947368, 'unit': 'um'}
            )
    """
    if not osp.exists(osp.dirname(file)):
        os.makedirs(osp.dirname(file), exist_ok=True)
    imsave(file, image, imagej=imagej, **kwargs)


def save_csv(file, df, **kwargs):
    """Save image array in ImageJ-compatible format"""
    if not osp.exists(osp.dirname(file)):
        os.makedirs(osp.dirname(file), exist_ok=True)
    df.to_csv(file, **kwargs)


def _set_tiff_warning_filters():
    # Ignore warnings known to occur under normal conditions when reading files via tifffile
    warnings.filterwarnings(
        'ignore', category=UserWarning,
        message='unpack: string size must be a multiple of element size'
    )
    warnings.filterwarnings(
        'ignore', category=RuntimeWarning,
        message='py_decodelzw encountered unexpected end of stream'
    )


def read_image(file, return_metadata=False):
    with warnings.catch_warnings():
        _set_tiff_warning_filters()

        # Use skimage io if metadata not necessary
        if not return_metadata:
            return sk_io.imread(file)
        # Otherwise, read file metatdata using tifffile
        else:
            with TiffFile(file) as tif:
                res = tif.asarray()
                return res, _get_tif_metadata(tif, shape=res.shape)


def read_tile(file, return_metadata=False):
    """Read a cytokit-specific 5D image tile

    Technical Note: This is a fairly complex process as it is necessary to deal with the fact that files
    saved using tifffile lose unit length dimensions.  To deal with this fact, the metadata in the image
    is parsed here to ensure that missing dimensions are added back.
    """
    # The "imagej_metadata" attribute looks like this for a 5D image with no unit-length dimensions
    # and original shape cycles=2, z=25, channels=2:
    # {'ImageJ': '1.11a', 'axes': 'TZCYX', 'channels': 2, 'frames': 2, 'hyperstack': True,
    # 'images': 100, 'mode': 'grayscale', 'slices': 25}
    # However, if a unit-length dimension was dropped it simply does not show up in this dict
    with warnings.catch_warnings():
        _set_tiff_warning_filters()
        with TiffFile(file) as tif:
            tags = dict(tif.imagej_metadata)
            if 'axes' not in tags:
                warnings.warn('ImageJ tags do not contain "axes" property (file = {}, tags = {})'.format(file, tags))
            else:
                if tags['axes'] != 'TZCYX':
                    warnings.warn(
                        'Image has tags indicating that it was not saved in TZCYX format.  '
                        'The file should have been saved with this property explicitly set and further '
                        'processing of it may be unsafe (file = {})'.format(file)
                    )
            slices = [
                slice(None) if 'frames' in tags else None,
                slice(None) if 'slices' in tags else None,
                slice(None) if 'channels' in tags else None,
                slice(None),
                slice(None)
            ]
            res = tif.asarray()[slices]

            if return_metadata:
                return res, _get_tif_metadata(tif, shape=res.shape)
            else:
                return res


def _get_tif_metadata(tif, shape=None):
    """Extract metadata from tif file

    Args:
        tif: File object
        shape: Any multidimensional image shape with at least 2 dimensions where HW axes are last
    Returns:
        Dictionary containing relevant metadata
    """
    # At TOW, labeling information is the only helpful metadata worth passing around,
    # but this may expand in the future
    tags = dict(tif.imagej_metadata)

    # Get 1D slice labels (which correspond to different axes of 5D tiles)
    res = dict(labels=tags.get('Labels'))
    if shape is not None:
        # Provide labels in reshaped array that correspond to image index dimensions
        if len(shape) >= 3:
            # This reshaping works, for example in a 5D tile, by wrapping labels across the
            # channel dimension, then the z dimension, then the cycle dimension
            res['structured_labels'] = np.array(res['labels']).reshape(shape[:-2])
        else:
            # If image is 1D or 2D, return labels in 1D form
            res['structured_labels'] = res['labels']
    return res


def save_tile(file, tile, config=None, infer_labels=True, **kwargs):
    """Save a cytokit-specific 5D image"""
    if tile.ndim != 5:
        raise ValueError('Expecting tile with 5 dimensions but got tile with shape {}'.format(tile.shape))
    # Save with explicit axes settings otherwise channels, cycles, and z planes are
    # all interpreted as individual slices instead of separate dimensions
    if 'metadata' not in kwargs:
        kwargs['metadata'] = {}
    if 'axes' in kwargs['metadata']:
        raise ValueError('Axes should not be set explicitly in metadata when using `save_tile`')
    kwargs['metadata']['axes'] = 'TZCYX'

    # If configuration provided, add as much context as possible to saved file via metadata arguments
    if config is not None:
        # Add arguments for resolution
        resolution, meta = ij_utils.get_config_resolution_args(config)
        kwargs['resolution'] = resolution
        kwargs['metadata'].update(meta)

        # If enabled, attempt to infer and add slice names
        if infer_labels:
            tags = ij_utils.get_config_slice_label_args(config, tile.shape)
            if tags is not None:
                if 'extratags' not in kwargs:
                    kwargs['extratags'] = []
                kwargs['extratags'] += tags

    save_image(file, tile, **kwargs)


def get_raw_img_path(ireg, itile, icyc, ich, iz):
    index_symlinks = cytokit.get_raw_index_symlinks()
    args = dict(cycle=icyc + 1, region=ireg + 1, tile=itile + 1, z=iz + 1, channel=ich + 1)
    # Remap indexes of input elements if any explicit overrides have been defined
    args = {k: index_symlinks.get(k, {}).get(v, v) for k, v in args.items()}
    return _formats()['raw_image'].format(**args)


FMT_PROC_IMAGE = 'proc_image'
FMT_PROC_DATA = 'proc_data'
FMT_PROC_EXEC = 'proc_exec'
FMT_FOCUS_IMAGE = 'best_focus_image'
FMT_MONTAGE_IMAGE = 'montage_image'
FMT_ILLUM_FUNC = 'illum_func'
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


def get_illumination_function_path(ireg):
    return _formats()[FMT_ILLUM_FUNC].format(region=ireg + 1)


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
    if file_type == cytokit.FT_GRAYSCALE:
        img = read_image(path)
    elif file_type == cytokit.FT_KEYENCE_RGB:
        img = read_image(path)
        if img.ndim != 3:
            raise ValueError(
                'With {} file types enabled, raw image at path "{}" should have 3 dims (shape = {})'
                .format(file_type, path, img.shape)
            )
        img = _collapse_keyence_rgb(path, img)
    elif file_type == cytokit.FT_KEYENCE_REPEAT:
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
    elif file_type == cytokit.FT_KEYENCE_MIXED:
        img = read_image(path)
        if img.ndim not in [2, 3]:
            raise ValueError(
                'With {} file types enabled, raw image at path "{}" should have 2 or 3 dims (shape = {})'
                .format(file_type, path, img.shape)
            )
        if img.ndim == 3:
            img = _collapse_keyence_rgb(path, img)
    else:
        raise ValueError('Raw file type "{}" is not valid; should be one of {}'.format(file_type, cytokit.RAW_FILE_TYPES))

    # Validate that all file types result in 2D image
    if img.ndim != 2:
        raise AssertionError(
            'Raw data file "{}" (with file type {} enabled) not processed correctly; should have resulted in '
            'single channel image but shape is {}'.format(path, file_type, img.shape))
    return img

