import os.path as osp
import numpy as np
import pandas as pd
from collections import OrderedDict
from cytokit.ops import op as cytokit_op
from cytokit import io as cytokit_io
from cytokit.function import data as function_data
from cytokit.cytometry.cytometer import DEFAULT_CELL_INTENSITY_PREFIX
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
from cytokit import SEED
import logging
logger = logging.getLogger(__name__)


def _to_rc(a):
    """Reshape a 5D tile to a single 2D matrix with channels in columns"""
    # (ncyc, nz, nch, nh, nw) -> (nz, nh, nw, ncyc, nch)
    a = np.transpose(a, (1, 3, 4, 0, 2))

    # (nz, nh, nw, ncyc, nch) -> (nz, nh, nw, C)
    a = np.reshape(a, a.shape[:3] + (-1,))

    # (nz, nh, nw, C) -> (R, C)
    return np.reshape(a, (-1, a.shape[-1]))


def _from_rc(a, shape):
    """Reshape 2D matrix with channel columns back into 5D tile"""
    # (R, C) -> (nz, nh, nw, C)
    a = np.reshape(a, tuple(shape[i] for i in [1, 3, 4]) + (-1,))

    # (nz, nh, nw, C) -> (nz, nh, nw, ncyc, nch)
    a = np.reshape(a, a.shape[:-1] + (shape[0], shape[2]))

    # (nz, nh, nw, ncyc, nch) -> (ncyc, nz, nch, nh, nw)
    return np.transpose(a, (3, 0, 4, 1, 2))


def get_default_crosstalk_coefficients(n):
    """Generate default cross-talk coefficient initialization matrix

    Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3832632/

    Args:
        n: Number of dimensions
    Returns:
        Square matrix of shape (n, n)
    """
    a = np.eye(n)
    for c in range(n):
        for r in range(c+1, n):
            a[r, c] = 2**(c-r)
            a[c, r] = 1/2**n
    return a


class SpectralUnmixing(cytokit_op.CytokitOp):

    def __init__(self, config, crosstalk_coefficients=None):
        super().__init__(config)

        params = config.spectral_unmixing_params
        self.crosstalk_coefficients = params.get('crosstalk_coefficients', crosstalk_coefficients)

        # cyc, ch = self.config.get_channel_coordinates(target_channel)
        self.channels = config.channel_names
        self.features = [DEFAULT_CELL_INTENSITY_PREFIX + c for c in self.channels]

        if len(self.channels) < 2:
            raise ValueError(
                'Spectral unmixing is only possible with at least 2 channels (channels configured = {})'
                .format(self.channels)
            )

        self.data = None
        self.saved_regions = set()

    def get_decomposition_model(self, region_index, region_data):
        logger.debug('Building spectral unmixing model for region %s', region_index)
        X = region_data[self.features].values

        # See: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3832632/
        np.random.seed(SEED)
        W = np.random.rand(*X.shape)

        # Use default crosstalk coefficient starting point only if not provided explicitly
        if self.crosstalk_coefficients is None:
            H = get_default_crosstalk_coefficients(X.shape[1])
        else:
            H = np.array(self.crosstalk_coefficients, dtype=float)

        # Normalize column sums to 1
        H = H / H.sum(axis=0)

        return NMF(max_iter=10000, init='custom', random_state=SEED).fit(X, W=W, H=H)

    def get_decomposition_coefs(self, est):
        return pd.DataFrame(est.components_, columns=self.features, index=self.features)

    def prepare_region_data(self, output_dir):
        if self.data is not None:
            return
        logger.info('Preparing spectral unmixing models')

        df = function_data.get_cytometry_data(output_dir, self.config, mode='all')
        n = len(df)
        if df is None or n == 0:
            raise ValueError('Cytometry data cannot be empty in order to use it for spectral unmixing')

        self.data = {}
        for region_index, region_data in df.groupby('region_index'):
            est = self.get_decomposition_model(region_index, region_data)
            coefs = self.get_decomposition_coefs(est)
            self.data[region_index] = (est, coefs)

    def _record_coefs(self, tile_indices):
        if tile_indices.region_index in self.saved_regions:
            return

        # Fetch unmixing model data for this region
        est, coefs = self.data[tile_indices.region_index]

        # Stack 2D data frame into rows, cols and values and record as operation measurement data
        coefs = coefs.stack().reset_index()
        coefs.columns = ['row', 'column', 'value']
        for i, r in coefs.iterrows():
            self.record(r.to_dict())

        self.saved_regions.add(tile_indices.region_index)

    def _run(self, tile, tile_indices):

        if self.data is None:
            raise ValueError('Operation cannot be run without first calling `prepare_region_data`')
        self._record_coefs(tile_indices)

        # Get unmixing model for this region
        est = self.data[tile_indices.region_index][0]

        # Get tile information prior to correction
        dtype = tile.dtype
        dinfo = np.iinfo(dtype)
        shape = tile.shape

        # Reshape tile to (R, C) where R = nz * nh * nw and C = ncyc * nch, get transformation
        # and then reshape back to 5D tile as (ncyc, nz, nch, nh, nw)
        tile = _from_rc(est.transform(_to_rc(tile)), shape)

        # Ensure that resulting array shape was not modified
        assert tile.shape == shape, \
            'Corrected tile has wrong shape (expected shape = {}, actual shape = {})'.format(shape, tile.shape)

        # Clip to range of original data type and convert
        tile = tile.clip(dinfo.min, dinfo.max).astype(dtype)

        return tile

    def save(self, tile_indices, output_dir, tile):
        # Overwrite the original preprocessed tile with corrected version
        path = cytokit_io.get_processor_img_path(
            tile_indices.region_index, tile_indices.tile_x, tile_indices.tile_y)
        cytokit_io.save_tile(osp.join(output_dir, path), tile, config=self.config)
        return path


