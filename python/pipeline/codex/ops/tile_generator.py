
from codex.ops.op import CodexOp
from skimage import io
from os import path as osp
import numpy as np


class CodexTileGenerator(CodexOp):

    def __init__(self, config, data_dir, region_index, tile_index):
        super(CodexTileGenerator, self).__init__(config)
        self.data_dir = data_dir
        self.region_index = region_index
        self.tile_index = tile_index

    def run(self):

        ncyc, nx, ny, nz, nch = self.config.tile_dims()

        # Tile should have shape (cycles, z, channel, height, width)
        img_cyc = []
        for icyc in range(ncyc):
            img_ch = []
            for ich in range(nch):
                img_z = []
                for iz in range(nz):
                    img_fmt = osp.join('Cyc{}_reg{}', '1_{0:05d}_Z{0:03d}_CH{}.tif')
                    img_name = img_fmt.format(
                        icyc + 1, self.region_index + 1, self.tile_index + 1,
                        iz + 1, ich + 1
                    )
                    img = io.imread(osp.join(self.data_dir, img_name))
                    img_z.append(img)
                img_ch.append(np.stack(img_z, 0))
            img_cyc.append(np.stack(img_ch, 1))
        tile = np.stack(img_cyc, 0)

        return tile
