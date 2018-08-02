import os.path as osp
import numpy as np
import pandas as pd
from codex.ops import op as codex_op
from codex import io as codex_io
from codex.cytometry.cytometer import DEFAULT_CHANNEL_PREFIX
from codex.function import data as function_data
from sklearn.ensemble import GradientBoostingRegressor
import logging
logger = logging.getLogger(__name__)

DEFAULT_FILTER_RANGE = [.1, .9]
DEFAULT_FILTER_FEATURES = ['cell_diameter']
SEED = 5512


class IlluminationCorrection(codex_op.CodexOp):

    def __init__(self, config, max_cells=100000, n_estimators=100, filter_range=DEFAULT_FILTER_RANGE, filter_features=None):
        super().__init__(config)

        params = config.illumination_correction_params
        self.target_channel = params['channel']
        self.target_feature = DEFAULT_CHANNEL_PREFIX + self.target_channel
        self.filter_range = params.get('filter_range', filter_range)
        self.max_cells = params.get('max_cells', max_cells)
        self.n_estimators = params.get('n_estimators', n_estimators)
        if 'filter_features' in params:
            self.filter_features = params['filter_features'] or []
        elif filter_features is not None:
            self.filter_features = filter_features
        else:
            # Default to the nuclear channel intensity and cell size as filters
            self.filter_features = [self.target_feature] + DEFAULT_FILTER_FEATURES

        if self.filter_range is None or len(self.filter_range) != 2:
            raise ValueError(
                'Must provide filter range as 2 item list (given = {})'
                .format(self.filter_range)
            )
        for v in self.filter_range:
            if not 0 <= v <= 1:
                raise ValueError(
                    'Filter range percentile values must be in [0, 1] (given = {})'
                    .format(self.filter_range)
                )

        self.data = None
        self.data_saved = False

    def get_filter_masks(self, df):
        """Get masks for each filtered feature

        Args:
            df: Cytometry dataframe
        Returns:
            DataFrame with column for each filter feature and values equal to boolean mask
                (where true means that the record is WITHIN the desired range)
        """
        # Compute low/high thresholds as dataframe like:
        #     feat_1 feat_2
        # p_lo    .1     10
        # p_hi   5.3    100
        ranges = df[self.filter_features].quantile(q=self.filter_range)

        # Stack masks horizontally as dataframe
        return pd.concat([
            df[c].between(ranges[c].iloc[0], ranges[c].iloc[1])
            for c in ranges
        ], axis=1)

    def prepare_region_data(self, output_dir):
        if self.data is not None:
            return
        # Use whatever cytometry data was generated, whether it was for best
        # z planes, all planes, or a specific one
        df = function_data.get_cytometry_data(output_dir, self.config, mode='all')
        n = len(df)
        if df is None or n == 0:
            raise ValueError('Cytometry data cannot be empty in order to use it for illumination correction')

        self.data = {}
        for region_index, region_data in df.groupby('region_index'):
            est = self.get_illumination_model(region_index, region_data)
            img = self.get_illumination_image(est)
            self.data[region_index] = (img, est)

    def get_illumination_model(self, region_index, df):
        # Restrict cell data to only records matching the given filters
        n = len(df)
        df = df[self.get_filter_masks(df).all(axis=1).values].copy()

        # If necessary, downsample modeling data to improve performance
        if len(df) > self.max_cells:
            df = df.sample(n=self.max_cells, random_state=SEED)

        if len(df) == 0:
            raise ValueError('Cytometry data empty after application of feature filters')
        logger.debug(
            'Building illumination model based on data for region %s (%s records originally, %s '
            'records after cell feature filtering)',
            region_index, n, len(df)
        )

        # Extract spatial cell features and prediction target as the intensity of the nuclear marker
        X, y = df[['ry', 'rx']], df[self.target_feature]
        if np.isclose(y.mean(), 0):
            raise ValueError(
                'Average nuclear channel intensity for region {} (across {} cells) is ~0, '
                'making illumination correction impossible'
                .format(region_index, len(df))
            )
        y = y / y.mean()

        # Fit regression model used to represent illumination surface
        est = GradientBoostingRegressor(n_estimators=self.n_estimators)
        return est.fit(X, y)

    def get_illumination_image(self, est):
        """Get an illumination image by predicting the intensity at each pixel across a region

        Args:
            est: Illumination model
        Returns:
            A 2D float32 array with shape equal (region height * tile height, region width * tile width)
        """
        # Get whole region shape as rows, cols
        r, c = (
            self.config.region_height * self.config.tile_height,
            self.config.region_width * self.config.tile_width
        )
        ii = np.transpose([np.repeat(np.arange(r), c), np.tile(np.arange(c), r)])
        X = pd.DataFrame(ii, columns=['ry', 'rx'])
        y = est.predict(X)
        return y.reshape((r, c)).astype(np.float32)

    def save_region_data(self, output_dir):
        if self.data is None:
            raise ValueError('Region data cannot be saved until `prepare_region_data` is called')
        if self.data_saved:
            return None
        path = None
        for region_index, (img, est) in self.data.items():
            path = osp.join(output_dir, codex_io.get_illumination_image_path(region_index))
            codex_io.save_image(path, img)
        self.data_saved = True
        return osp.dirname(path or '')

    def _run(self, tile, tile_indices):
        # Get illumination image for this region
        img = self.data[tile_indices.region_index][0]

        # Extract matching patch in illumination image
        r, c = tile_indices.tile_y * self.config.tile_height, tile_indices.tile_x * self.config.tile_width
        img = img[r:(r + self.config.tile_height), c:(c + self.config.tile_width)]

        # Get tile type information prior to multiplication
        dtype = tile.dtype
        dinfo = np.iinfo(dtype)

        # Divide correction across HW dimensions of tile (must first add leading dims to match 5D tile)
        tile = tile / img[np.newaxis, np.newaxis, np.newaxis, :, :]

        # Clip to range of original data type and convert
        tile = tile.clip(dinfo.min, dinfo.max).astype(dtype)

        return tile, img

    def save(self, tile_indices, output_dir, illum_data):
        # Overwrite the original preprocessed tile
        path = codex_io.get_processor_img_path(tile_indices.region_index, tile_indices.tile_x, tile_indices.tile_y)
        codex_io.save_tile(osp.join(output_dir, path), illum_data[0])
        return path

