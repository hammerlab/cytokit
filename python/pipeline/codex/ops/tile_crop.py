from codex.ops.op import CodexOp
import logging

logger = logging.getLogger(__name__)

# Note on cropping -- example settings for real data (akoya tonsil):
# - acquisition image (1920 x 1440) (cols x rows)  (images are wider than they are tall)
# - tile dims (1344 x 1008) (cols x rows)
# - tile overlap x = 576, tile overlap y = 432 (x refers to columns and y refers to rows)


def get_slice(config):
    """Get 2D slice defining crop operation appropriate for given configuration"""
    nw, nh = config.tile_width, config.tile_height
    ow, oh = config.overlap_x, config.overlap_y
    w_start, h_start = ow // 2, oh // 2
    w_stop, h_stop = w_start + nw, h_start + nh
    return [slice(h_start, h_stop), slice(w_start, w_stop)]


def apply_slice(img, crop_slice):
    """Apply cropping slice to trailing image dimensions"""
    if img.ndim < 2:
        raise ValueError('Expecting at least 2 dimensions in image (shape of given array = {})'.format(img.shape))
    slices = [slice(None, None) for _ in range(img.ndim - 2)] + crop_slice
    return img[slices]


class CodexTileCrop(CodexOp):

    def __init__(self, config):
        super(CodexTileCrop, self).__init__(config)

    def run(self, tile):
        return apply_slice(tile, get_slice(self.config))