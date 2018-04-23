from codex.ops.op import CodexOp
from os import path as osp
from codex import io as codex_io
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CodexTileGenerator(CodexOp):

    def __init__(self, config, data_dir, region_index, tile_index):
        super(CodexTileGenerator, self).__init__(config)
        self.data_dir = data_dir
        self.region_index = region_index
        self.tile_index = tile_index

    def run(self):
        ncyc, nz, nch = self.config.n_cycles, self.config.n_z_planes, self.config.n_channels_per_cycle

        # Tile should have shape (cycles, z, channel, height, width)
        img_cyc = []
        for icyc in range(ncyc):
            img_ch = []
            for ich in range(nch):
                img_z = []
                for iz in range(nz):
                    img_path = codex_io.get_raw_img_path(self.region_index, self.tile_index, icyc, ich, iz)
                    img_z.append(codex_io.read_tile(osp.join(self.data_dir, img_path)))
                img_ch.append(np.stack(img_z, 0))
            img_cyc.append(np.stack(img_ch, 1))
        tile = np.stack(img_cyc, 0)

        return tile
