from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
from cytokit.image import color as cvcolor
import numpy as np

DEFAULT_COLORS = list(cvcolor.COLORS.values())


def describe(img):
    """Return dict containing commonly referenced image metadata"""
    return dict(
        type=str(img.dtype),
        shape=img.shape,
        min=img.min(),
        max=img.max(),
        mean=img.mean(),
        bytes=img.nbytes
    )


def pad_around_center(img, shape, mode='constant', constant_values=0):
    """Pad an image to a target shape with centering

    Args:
        img: Image to pad (any number of dimensions); must have size <= `shape` in all dimensions
        shape: Target shape to pad to
        mode: Padding mode (see numpy.pad)
        constant_values: Constant value(s) used for padding when using mode 'constant' (see numpy.pad)
    """
    imgs, ts = np.array(img.shape), np.array(shape)
    if np.any(imgs > ts):
        raise ValueError(
            'Cannot pad image with shape {} to target shape {} since at least one dimension is already larger'
            .format(imgs, ts)
        )
    # Compute lower padding as half the difference in shapes (with downward rounding)
    pad_lo = (ts - imgs) // 2
    # Set upper padding as remainder
    pad_hi = ts - imgs - pad_lo
    assert np.all(pad_hi >= 0)
    # Apply padding
    return np.pad(img, list(zip(pad_lo, pad_hi)), mode=mode, constant_values=constant_values)


def crop_around_center(img, shape):
    """Crop an image to a target shape around center

    Args:
        img: Single-channel image with any number of dimensions; note that centers are calculated based on length
            along a dimension which is meaningless for RGB images
        shape: Shape of target image (which will have the same center as original image)
    """
    imgs, ts = np.array(img.shape), np.array(shape)
    if np.any(imgs < ts):
        raise ValueError(
            'Cannot crop image with shape {} to target shape {} since at least one dimension is already smaller'
            .format(imgs, ts)
        )

    crop_offset = (imgs - ts) // 2
    assert np.all(crop_offset >= 0)
    return img[[slice(start, start + length) for start, length in list(zip(crop_offset, shape))]]


def resize_image_with_crop_or_pad(img, shape, **kwargs):
    """Resize an image by cropping or padding around center (i.e. no interpolation)

    Args:
        img: Image array to resize (can have any number of dimensions but must not be scalar)
        shape: Target shape with length equal to image.ndim
        kwargs: Extra arguments for `pad_around_center` controlling padding mode and fill values
    Returns:
        Image array of same type as image but with shape `shape`
    """
    if img.ndim != len(shape):
        raise ValueError(
            'Image must have same number of dimensions as target shape (given image shape = {}, target shape = {})'
            .format(img.shape, shape)
        )
    img = pad_around_center(img, np.maximum(img.shape, shape), **kwargs)
    img = crop_around_center(img, np.minimum(img.shape, shape))
    return img


def blend_image_channels(img, mix=None, colors=None):
    """Get single RGB image by blending any number of image channels

    Args:
        img: Image in CYX or YX format (note that in YX format, this operation amounts to nothing but color conversion)
        mix: Array or list-like of length C (same as number of channels) to be multiplied by
            each channel image in the blended result
        colors: Array or list-like of shape (N, 3) where each 3 item row is an rgb color channel fraction in [0, 1].
            For example [1, 0, 0] would associated all values for a channel to the color red while [1, 0, 1] would
            indicate magenta (R + B).  The length of this list does not necessarily need to match the number of
            channels and if that there are not enough colors for each channel, they will be cycled indefinitely
    Returns:
        RGB image
    """
    if img.ndim == 2:
        img = img[np.newaxis]
    if img.ndim != 3:
        raise ValueError('Expecting  3 dimensions in image (image shape given = {})'.format(img.shape))

    colors = DEFAULT_COLORS if colors is None else colors
    nch = img.shape[0]
    ncolor = len(colors)

    # Default mixture proportions to ones
    if mix is None:
        mix = [1] * nch

    # Expect that there is a proportion for each channel
    if nch != len(mix):
        raise ValueError(
            'Number of mixture proportions must equal number of channels '
            '(image shape given = {}, mixture proportions = {})'
            .format(img.shape, mix)
        )
    mix = np.array(mix)

    nr, nc = img.shape[1], img.shape[2]
    res = np.zeros((nr, nc, 3), dtype=np.float32)

    for i in range(nch):
        # Fetch channel 2D image and reshape to YX3
        rgb = np.repeat(img[i][..., np.newaxis], repeats=3, axis=-1)
        color = colors[i % ncolor]
        if len(color) != 3:
            raise ValueError('Colors given should have size 3 in second dimension; colors given = {}'.format(colors))
        res = res + (rgb * np.array(color) * mix[i])

    # Clip floating point result to lower and upper bounds of resulting data type
    # *Note that .astype actually starts overflow values back at 0, so this is necessary
    vmin, vmax = np.iinfo(img.dtype).min, np.iinfo(img.dtype).max
    res = rescale_intensity(
        res.clip(vmin, vmax).astype(img.dtype),
        in_range='dtype', out_range=np.uint8
    ).astype(np.uint8)
    return res


def constrain_image_channels(img, dtype=None, ranges=None):
    """Constrain image channels to particular ranges

    Args:
        img: Image in CYX format
        dtype: Resulting image datatype; defaults to image dtype
        ranges: Array with shape (C, 2) or (1, 2) where second dimension denotes range lower and upper clipping values;
            A "None" lower or upper value indicates image minimum and maximum respectively; defaults to (None, None)
            for each channel
    Returns:
        Image with same shape as input and all pixel values for channels clipped to the corresponding range
    """
    if img.ndim == 2:
        img = img[np.newaxis]
    if img.ndim != 3:
        raise ValueError('Expecting  3 dimensions in image (image shape given = {})'.format(img.shape))

    if dtype is None:
        dtype = img.dtype

    if ranges is None:
        ranges = [[None, None]] * img.shape[0]
    ranges = np.array(ranges)
    if ranges.ndim == 1:
        # Stack ranges to (C, -1) -- not sure how many items are in second axis but that is checked next
        ranges = np.repeat(ranges[np.newaxis], img.shape[0], 0)

    # Validate that ranges have length equal to num image channels and 2 values for range per channel
    if ranges.shape[1] != 2:
        raise ValueError('Ranges must have length 2 in second dimension (ranges shape = {})'.format(ranges.shape))
    if ranges.shape[0] != img.shape[0]:
        raise ValueError(
            'Ranges must have length equal to number of image channels '
            '(ranges shape = {}, image shape = {}'.format(ranges.shape, img.shape)
        )

    # Apply clip to each channel and restack
    def prep_img(img, minv, maxv):
        img = img.clip(minv, maxv)
        # Check max=min to avoid division by zero warning from rescale_intensity
        if img.min() == img.max():
            img = (img * 0)
        else:
            img = rescale_intensity(img, in_range='image', out_range=dtype)
        return img.astype(dtype)

    return np.stack([
        prep_img(
            img[i],
            -np.inf if ranges[i, 0] is None else ranges[i, 0],
            np.inf if ranges[i, 1] is None else ranges[i, 1]
        )
        for i in range(img.shape[0])
    ])


def filter_label_image(img, df, config):
    """Filter label image to match objects in a data frame

    Args:
        img: Integer label image (2D)
        df: Data frame containing id of each object as well as tile x and y coordinates
        config: Experiment configuration
    """
    assert img.ndim == 2, 'Expecting 2D image but got image with shape {}'.format(img.shape)
    res = img.copy()
    nr, nc = config.region_height, config.region_width
    sr, sc = config.tile_height, config.tile_width
    for tile_y in range(nr):
        for tile_x in range(nc):
            ids = df[(df['tile_x'] == tile_x) & (df['tile_y'] == tile_y)]['id'].unique()
            rs = slice(tile_y * sr, (tile_y + 1) * sr)
            cs = slice(tile_x * sc, (tile_x + 1) * sc)
            im = res[rs, cs]
            im[~np.isin(im, ids)] = 0
    return res
