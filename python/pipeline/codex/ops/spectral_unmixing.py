import os.path as osp
import numpy as np
import pandas as pd
from collections import OrderedDict
from codex.ops import op as codex_op
from codex import io as codex_io
from codex.function import data as function_data
from codex.cytometry.cytometer import DEFAULT_CELL_INTENSITY_PREFIX
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
import logging
logger = logging.getLogger(__name__)


class SpectralUnmixing(codex_op.CodexOp):

    def __init__(self, config):
        super().__init__(config)

        # cyc, ch = self.config.get_channel_coordinates(target_channel)
        self.channels = config.channel_names
        self.features = [DEFAULT_CELL_INTENSITY_PREFIX + c for c in self.channels]

        self.data = None
        self.data_saved = False

    def get_decomposition_model(self, region_index, region_data):
        logger.debug('Building spectral unmixing model for region %s', region_index)
        X = region_data[self.features].values

        # See: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3832632/
        W = np.random.rand(*X.shape)

        # TODO: Implement exponential weights based on wavelength
        H = (np.eye(X.shape[1], dtype='float') + .1) * (10. / 11.)
        return NMF(max_iter=1000, init='custom').fit(X, W=W, H=H)

    def get_decomposition_coefs(self, est):
        return pd.DataFrame(est.components_, columns=self.features, index=self.features)

    def prepare_region_data(self, output_dir):
        if self.data is not None:
            return

        df = function_data.get_cytometry_data(output_dir, self.config, mode='all')
        n = len(df)
        if df is None or n == 0:
            raise ValueError('Cytometry data cannot be empty in order to use it for spectral unmixing')

        self.data = {}
        for region_index, region_data in df.groupby('region_index'):
            est = self.get_decomposition_model(region_index, region_data)
            coefs = self.get_decomposition_coefs(est)
            self.data[region_index] = (est, coefs)

    def save_region_data(self, output_dir):
        if self.data is None:
            raise ValueError('Region data cannot be saved until `prepare_region_data` is called')
        if self.data_saved:
            return None
        for region_index, (est, coefs) in self.data.items():
            # Stack 2D data frame into rows, cols and values
            coefs = coefs.stack().reset_index()
            coefs.columns = ['row', 'column', 'value']
            for i, r in coefs.iterrows():
                self.record(r.to_dict())
        self.data_saved = True

    def _run(self, tile, tile_indices):
        # Get unmixing model for this region
        est = self.data[tile_indices.region_index][0]

        # Get tile type information prior to correction
        dtype = tile.dtype
        dinfo = np.iinfo(dtype)

        # TODO:
        # - Decide what to do with z dimension
        # - Need to get X matrix with pixel per row
        # - Edit tile in place

        # Reshape image to rows/cols:
        # shape = img.shape
        # X = img.reshape((-1, len(features)))
        # Y = decomp.transform(X)
        # return Y.reshape(shape)

        # Clip to range of original data type and convert
        tile = tile.clip(dinfo.min, dinfo.max).astype(dtype)

        return tile

    def save(self, tile_indices, output_dir, tile):
        path = codex_io.get_processor_img_path(
            tile_indices.region_index, tile_indices.tile_x, tile_indices.tile_y)
        codex_io.save_tile(osp.join(output_dir, path), tile, config=self.config)


