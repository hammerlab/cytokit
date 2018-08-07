import os.path as osp
import numpy as np
import pandas as pd
from collections import OrderedDict
from codex.ops import op as codex_op
from codex import io as codex_io
from codex.cytometry.cytometer import DEFAULT_CHANNEL_PREFIX
from codex.function import data as function_data
from sklearn.linear_model import HuberRegressor, Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from skimage import transform
import logging
logger = logging.getLogger(__name__)

# filter_params:
# percentile_range: [.01, .99]
# features: [DAPI, cell_size]
# max_cells: 250000
# feature_params: {tile: {degree: 2}, region: {degree: 2}}

DEFAULT_FILTER_PARAMS = dict(percentile_range=[.01, .99], features=['cell_size'], max_cells=250000)
DEFAULT_FEATURE_PARAMS = dict(tile=dict(degree=2), region=dict(degree=2))
DEFAULT_MODEL_PARAMS = dict(type='ls')
DEFAULT_PREDICTION_PARAMS = dict(step_size=10)
MODELS = {
    'ls': {'factory': lambda args: LinearRegression(**args)},
    'huber': {'factory': lambda args: HuberRegressor(**args)},
    'ridge': {'factory': lambda args: Ridge(**args)},
    'gbr': {'factory': lambda args: GradientBoostingRegressor(random_state=SEED, **args)},
    'knn': {'factory': lambda args: KNeighborsRegressor(**args)}
}
SEED = 5512


# max_cells=250000
# step_size = 10

def _get_params(key, params, defaults):
    # Copy default values
    res = dict(defaults)
    # If params present, overwrite default values
    if key in params:
        res.update(params[key])
    return res


def get_prediction_features(tile_shape, region_shape, step=1):
    step_size = self.prediction_params['step_size']
    r, c = shape
    ri, ci = np.arange(0, r, step_size), np.arange(0, c, step_size)

    # Produce 2D array with 2 cols, ry and rx
    X_region = np.transpose([np.repeat(ri, len(ci)), np.tile(ci, len(ri))])
    X_tile = np.mod()

    pass

class IlluminationCorrection(codex_op.CodexOp):

    def __init__(
            self, config,
            filter_params=DEFAULT_FILTER_PARAMS,
            feature_params=DEFAULT_FEATURE_PARAMS,
            model_params=DEFAULT_MODEL_PARAMS,
            prediction_params=DEFAULT_PREDICTION_PARAMS,
            overwrite_tile=False):
        super().__init__(config)

        params = config.illumination_correction_params

        # Get mapping of source to target channel names, and sort by source channel
        self.channel_mapping = pd.Series(params['channel_mapping']).sort_index()

        # Extra parameters relevant to each step
        self.filter_params = _get_params('filter_params', filter_params)
        self.feature_params = _get_params('feature_params', feature_params)
        self.model_params = _get_params('model_params', model_params)
        self.prediction_params = _get_params('prediction_params', prediction_params)

        # Extract flag indicating whether or not to overwrite original tiles with corrected results
        self.overwrite_tile = params.get('overwrite_tile', overwrite_tile)

        ############################
        # Configuration Validation #
        ############################

        if len(self.filter_params['percentile_range']) != 2:
            raise ValueError(
                '`percentile_range` in filter_params must be 2 item list (given = {})'
                .format(self.filter_params['percentile_range'])
            )
        for v in self.filter_params['percentile_range']:
            if not 0 <= v <= 1:
                raise ValueError(
                    '`percentile_range` value in filter_params values must be in [0, 1] (given = {})'
                    .format(v)
                )

        if self.prediction_params['step_size'] < 1:
            raise ValueError(
                '`step_size` in prediction_params must be > 1 (given = {})'
                .format(self.prediction_params['step_size'])
            )
        if self.model_params['type'] not in MODELS:
            raise ValueError(
                '`type` in model_params type must be one of [] (given = "{}")'
                .format(self.model_params['type'], list(MODELS.keys()))
            )

        self.data = None
        self.data_saved = False

    def _get_filter_masks(self, df, features):
        """Get masks for each filtered feature

        Args:
            df: Cytometry dataframe
            features: List of features to perform percentile filtering on
        Returns:
            DataFrame with column for each filter feature and values equal to boolean mask
                (where true means that the record is WITHIN the desired range)
        """
        # Compute low/high thresholds as dataframe like:
        #     feat_1 feat_2
        # p_lo    .1     10
        # p_hi   5.3    100
        ranges = df[features].quantile(q=self.filter_params['percentile_range'])

        # Stack masks horizontally as dataframe
        return pd.concat([
            df[c].between(ranges[c].iloc[0], ranges[c].iloc[1])
            for c in ranges
        ], axis=1)

    def _get_filter_features(self, df):
        feats = []
        for feat in self.filter_params['features']:
            # Test membership for feature as provided
            if feat in df:
                feats.append(feat)
                continue
            # Test membership when prefixed by channel markers
            ch_feat = DEFAULT_CHANNEL_PREFIX + feat
            if ch_feat in df:
                feats.append(ch_feat)
                continue
            raise ValueError('Feature "{}" not present in cytometry data'.format(feat))


    def _get_prediction_features(self):


        step_size = self.prediction_params['step_size']
        r, c = shape
        ri, ci = np.arange(0, r, step_size), np.arange(0, c, step_size)

        # Produce 2D array with 2 cols, ry and rx
        X = np.transpose([np.repeat(ri, len(ci)), np.tile(ci, len(ri))])

        pass

    def _prepare_prediction_features(self, df):
        pass

    def _estimate_model(self, X, y):
        # Fit regression model used to represent illumination surface
        args = self.model_params.get('args', {})
        est = MODELS[self.model_params['type']]['factory'](args)
        return est.fit(X, y)

    def get_illumination_models(self, region_index, df):
        n = len(df)
        ests = OrderedDict()
        for channel in self.channel_mapping.index:

            # Set list of features to filter on
            features = self._get_filter_features(df)

            # Restrict cell data to only records matching the given filters
            dfm = df[self._get_filter_masks(df, features).all(axis=1).values].copy()

            # If necessary, downsample modeling data to improve performance
            if len(dfm) > self.filter_params['max_cells']:
                dfm = dfm.sample(n=self.filter_params['max_cells'], random_state=SEED)

            if len(dfm) == 0:
                raise ValueError(
                    'Cytometry data empty after application of feature filters for channel {}'.format(channel))

            logger.debug(
                'Building illumination model for region %s, channel "%s" using %s cells (%s originally)',
                region_index, channel, len(dfm), n
            )

            # Extract spatial cell features and prediction target
            X, y = dfm[['ry', 'rx']], dfm[DEFAULT_CHANNEL_PREFIX + channel]
            if np.isclose(y.mean(), 0):
                raise ValueError(
                    'Average {} channel intensity for region {} (across {} cells) is ~0, '
                    'making illumination correction impossible'
                    .format(region_index, channel, len(df))
                )
            y = y / y.mean()

            # Fit regression model used to represent illumination surface
            ests[channel] = self._estimate_model(X, y)
        return ests

    def _estimate_image(self, est, shape):
        step_size = self.prediction_params['step_size']
        r, c = shape
        ri, ci = np.arange(0, r, step_size), np.arange(0, c, step_size)

        # Produce 2D array with 2 cols, ry and rx
        X = np.transpose([np.repeat(ri, len(ci)), np.tile(ci, len(ri))])

        y = est.predict(X)

        # Some models may intentionally produce nans for pixels too far away from foreground
        # objects so set those values to 1 as this will lead to no change in correction
        y = np.where(np.isnan(y), 1., y)

        # Reshape to image shape
        y = y.reshape((len(ri), len(ci)))

        # Upsample if necessary
        if step_size > 1:
            y = transform.resize(y, shape, mode='constant', order=1, anti_aliasing=False, preserve_range=False)

        assert y.shape == shape, 'Expecting result with shape {} but was {}'.format(shape, y.shape)
        return y.astype(np.float32)

    def get_illumination_images(self, ests):
        """Get an illumination image by predicting the intensity at each pixel across a region

        Args:
            ests: Illumination models
        Returns:
            Dictionary mapping source channels to a 3D float32 array with shape equal to
            (region height * tile height, region width * tile width)
        """
        # Get whole region shape as rows, cols
        r, c = (
            self.config.region_height * self.config.tile_height,
            self.config.region_width * self.config.tile_width
        )

        imgs = OrderedDict()
        for channel, est in ests.items():
            imgs[channel] = self._estimate_image(est, (r, c))

        if len(imgs) > 0:
            img = list(imgs.values())[0]
            logger.debug('Resulting illumination image array shape = %s (dtype = %s)', img.shape, img.dtype)

        return imgs

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
            ests = self.get_illumination_models(region_index, region_data)
            imgs = self.get_illumination_images(ests)
            self.data[region_index] = (imgs, ests)

    def save_region_data(self, output_dir):
        if self.data is None:
            raise ValueError('Region data cannot be saved until `prepare_region_data` is called')
        if self.data_saved:
            return None
        path = None
        for region_index, (imgs, ests) in self.data.items():
            # Stack 2D images on first axis to give 3D array
            img = np.stack(list(imgs.values()), 0)
            assert img.ndim == 3, 'Expecting 3D array, got shape {}'.format(img.shape)
            path = osp.join(output_dir, codex_io.get_illumination_function_path(region_index))
            codex_io.save_image(path, img)

        self.data_saved = True
        return osp.dirname(path or '')

    def _run(self, tile, tile_indices):
        # Get illumination image for this region
        imgs = self.data[tile_indices.region_index][0]

        # Get tile type information prior to multiplication
        dtype = tile.dtype
        dinfo = np.iinfo(dtype)

        # Determine starting offsets for tile slicing
        r, c = tile_indices.tile_y * self.config.tile_height, tile_indices.tile_x * self.config.tile_width

        # Loop through channel mapping and apply each adjustment
        tile = tile.astype(np.float32)
        for source_channel, target_channel in self.channel_mapping.to_dict().items():

            logger.debug('Applying correction from source channel "%s" to target "%s"', source_channel, target_channel)

            # Extract matching patch in illumination image
            img = imgs[source_channel][r:(r + self.config.tile_height), c:(c + self.config.tile_width)]

            # If application is across all channels, do single matrix division
            if target_channel == 'all':
                # Divide correction across HW dimensions of tile (must first add leading dims to match 5D tile)
                tile = tile / img[np.newaxis, np.newaxis, np.newaxis, :, :]

            # Otherwise, find target channel's coordinates within
            # tile and apply adjustment only to that slice
            else:
                cyc, ch = self.config.get_channel_coordinates(target_channel)
                tile[cyc, :, ch, :, :] = tile[cyc, :, ch, :, :] / img[np.newaxis, :, :]

        # Clip to range of original data type and convert
        tile = tile.clip(dinfo.min, dinfo.max).astype(dtype)

        return tile

    def save(self, tile_indices, output_dir, tile):
        if self.overwrite_tile:
            # Overwrite the original preprocessed tile with corrected version
            path = codex_io.get_processor_img_path(
                tile_indices.region_index, tile_indices.tile_x, tile_indices.tile_y)
            codex_io.save_tile(osp.join(output_dir, path), tile, config=self.config)
        else:
            # Save corrected tile in separate location
            path = codex_io.get_illumination_image_path(
                tile_indices.region_index, tile_indices.tile_x, tile_indices.tile_y)
            codex_io.save_tile(osp.join(output_dir, path), tile, config=self.config)
        return path

