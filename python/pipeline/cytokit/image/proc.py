from cytokit.image import ops as cvops
from cytokit.image import color as cvcolor
import numpy as np
from collections import OrderedDict


class ImageProcessor(object):

    def __init__(self, n_channels, ranges=None, colors=None):
        self.n_channels = n_channels
        self.colors = colors
        self.ranges = ranges

    def run(self, img):
        assert img.shape[0] == self.n_channels, \
            'Expecting {} channels but got image with shape {}'.format(self.n_channels, img.shape)

        # print('in', img.dtype, img.shape, img.min(), img.max())
        img = cvops.constrain_image_channels(img, ranges=self.ranges, dtype=np.uint8)
        # print('mid', img.dtype, img.shape, img.min(), img.max())
        img = cvops.blend_image_channels(img, colors=self.colors)
        # print('out', img.dtype, img.shape, img.min(), img.max())
        assert img.ndim == 3 and img.shape[-1] == 3, \
            'Expecting RGB result (image shape = {})'.format(img.shape)
        return img


def get_image_processor(channels, ranges=None, colors=None):
    """ Get image processor for named ranges and colors

    Args:
        channels: Full list of channels (in order) expected to be present in images for processing
        ranges: Dictionary (order does not matter) containing channel names as keys and 2-item list-like
            values with lower and upper boundaries on channel values; if not given, channel values will
            be constrained
        colors: Dictionary (order does not matter) containing channel names as keys and 3-item list-like
            values with either color names as strings (see cytokit.image.color) or RGB fractions in [0, 1]
            (i.e. [1, 0, 0] corresponds to red); if not given, colors will be assigned arbitrarily
    """

    # Validate that a range and/or color settings are not provided for invalid channels
    def validate_arg(arg, name):
        if arg is not None:
            diff = np.setdiff1d(list(arg.keys()), channels)
            if len(diff) > 0:
                raise ValueError(
                    '{} settings given for invalid channel names; given channels = {}, channels available = {}'
                    .format(name, list(arg.keys()), channels)
                )
    validate_arg(ranges, 'Range')
    validate_arg(colors, 'Color')

    def resolve_color(channel):
        color = colors.get(channel, None)
        if color is None or isinstance(color, str):
            return cvcolor.map(color)
        else:
            if len(color) != 3 or np.array(color).min() < 0 or np.array(color).max() > 1:
                raise ValueError(
                    'Color for channel "{}" should be 3 item list-like with values in [0, 1] not "{}"'
                    .format(channel, color)
                )
            return color

    def resolve_range(channel):
        range = ranges.get(channel, [None, None])
        if len(range) != 2:
            raise ValueError(
                'Range for channel "{}" should be 2 item list-like with lower and upper boundaries not "{}"'
                .format(channel, range)
            )
        return range

    return ImageProcessor(
        len(channels),
        ranges=None if ranges is None else [resolve_range(c) for c in channels],
        colors=None if colors is None else [resolve_color(c) for c in channels],
    )
