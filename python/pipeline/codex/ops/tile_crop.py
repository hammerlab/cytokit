from codex.ops.op import CodexOp
import logging

logger = logging.getLogger(__name__)

# Note on cropping -- example settings for real data (akoya tonsil):
# - acquisition image (1920 x 1440) (cols x rows)  (images are wider than they are tall)
# - tile dims (1344 x 1008) (cols x rows)
# - tile overlap x = 576, tile overlap y = 432 (x refers to columns and y refers to rows)

class CodexTileCrop(CodexOp):

    def __init__(self, config):
        super(CodexTileCrop, self).__init__(config)

    def run(self, tile):
        # Tile should have shape (cycles, z, channel, height, width)
        nw, nh = self.config.tile_width(), self.config.tile_height()

        ow, oh = self.config.overlap_x(), self.config.overlap_y()
        w_start, h_start = ow // 2, oh // 2
        w_stop, h_stop = w_start + nw, h_start + nh

        return tile[:, :, :, h_start:h_stop, w_start:w_stop]
